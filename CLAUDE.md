# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a deep learning project that predicts the **Dst (Disturbance Storm Time) geomagnetic index** 6 hours into the future using solar wind data. It serves as an early warning system for geomagnetic storms.

## Running the Pipeline

All code lives in `SolarWindLSTM.ipynb`. Run it cell-by-cell in Jupyter:

```bash
jupyter notebook SolarWindLSTM.ipynb
```

Or execute the full pipeline non-interactively:

```bash
jupyter nbconvert --to notebook --execute SolarWindLSTM.ipynb
```

Saved model weights write to `solar_bilstm_model.pth` whenever validation loss improves.

## Data

Raw data lives in `archive/`:
- `solar_wind.csv` — solar wind parameters (Bx, By, Bz, Bt, speed, density, temperature, etc.) at sub-hourly resolution
- `labels.csv` — Dst index values
- `sunspots.csv` — monthly smoothed sunspot number
- All files use a `(period, timedelta)` multi-index; periods are named event windows (e.g., `train_a`, `train_b`)

Cell `bb365e9d` (A1) auto-downloads the full MagNet dataset from Zenodo (record `12694950`) if fewer than 6 periods are detected. Safe to re-run.

## Architecture & Pipeline

**Data pipeline** (`load_and_preprocess` → `create_sequences`):
1. Load CSVs, set `(period, timedelta)` multi-index
2. Aggregate to hourly **mean + std** for 14 solar-wind variables (paper feature set)
3. Merge smoothed sunspot number → **29 features total** (`feature_set='paper'`). Optional `feature_set='full'` adds 8 derived features (energy, rolling means, dyn_pressure) → 37 features.
4. Create sliding windows: 48-hour input → 6-hour target vector
5. **Per-period 70/20/10 chronological train/val/test split**; `StandardScaler` fit only on training data

**Model** (`SolarAttentionLSTM`):
- 2-layer BiLSTM (hidden=128 per direction, dropout=0.3, bidirectional=True)
- Custom attention layer that weights each of the 48 input timesteps
- Output: concatenation of attention context vector + last LSTM hidden state → linear → 6 Dst predictions

**Training**:
- Two-tier asymmetric weighted MSE: moderate storms (Dst < −20 nT) 5×, intense storms (Dst < −50 nT) 15×
- Adam optimizer, lr=0.001, weight_decay=1e-5
- Gradient clipping max_norm=1.0
- Early stopping with patience=15 on validation loss

## Key Hyperparameters

```python
SEQ_LEN          = 48     # Input window (hours)
FORECAST_HORIZON = 6      # Prediction horizon (hours)
HIDDEN_DIM       = 128
NUM_LAYERS       = 2
DROPOUT          = 0.3
BIDIRECTIONAL    = True
BATCH_SIZE       = 64
LEARNING_RATE    = 0.001
TRAIN_SPLIT      = 0.7
VAL_SPLIT        = 0.2
TEST_SPLIT       = 0.1
EPOCHS           = 50
FEATURE_SET      = 'paper'  # 29 features; 'full' for 37
```

## Performance (current best model, Phase A — 29 features, per-period 70/20/10 split)

**BiLSTM base (full val set, t+6):**
- RMSE: 11.5731 nT
- Pearson r: 0.7755
- R²: 0.5822

**BiLSTM + GRU correction (full val set, t+6):** RMSE 11.2267 nT (+0.35 nT vs base)

**Held-out test set (never seen during training or early stopping, t+6):**
- RMSE: 11.2597 nT | Pearson r: 0.8247 | R²: 0.6780
- Intense storm RMSE: 44.64 nT (N=387)

**OMNI-augmented BiLSTM (Phase B Step 4 — REJECTED):** single-seed-42 looked great (t+6 11.14, intense 40.01) but 5-seed paired verify showed no robust gain (intense Δ −1.73±3.79, aggregate Δ −0.147±0.370 — aug worse). Single-seed mirage. Base stays primary. See Step 4 section.

**1-min OMNI-augmented BiLSTM (Phase D — ADOPTED 2026-05-25): FIRST lever to robustly beat base.** Real `OMNI_HRO_1MIN` (true hourly mean+std, not Step-4's 3h proxy), 15 storms → 1389 seqs / 877 intense (2.7× Step 4's 329). 5-seed paired held-out test t+6: **aggregate 11.42→10.94 (Δ +0.48±0.28), intense 44.46→36.10 (Δ +8.36±3.39)** — all 5 seeds positive on both. Adopted checkpoint `solar_bilstm_omni1m_model.pth`. **Intense storm RMSE cut 19%.** See Phase D section + `results.md`.

**Storm-conditional t+6 (val set, BiLSTM base):**
| Condition | N | RMSE (nT) | R² |
|---|---|---|---|
| Quiet (Dst ≥ −20) | 23840 | 9.31 | −0.17 |
| Moderate (−50 to −20) | 3440 | 14.20 | −2.42 |
| Intense (Dst < −50) | 662 | 38.48 | +0.02 |

**Note on apparent regression vs old 8.45 nT:** Phase A switched to a per-period 70/20/10 split (vs old global 80/20) and the full Zenodo dataset (vs partial archive). The new val set contains ~4.5× more intense storms (145 → 662) and the test set has 387 — RMSE numerator naturally grew. Pearson r (0.72 → 0.78) and R² (0.41 → 0.58) both **improved**, indicating better variance capture. Direct cross-Phase RMSE comparison is invalid.

## A3 — Comparison to TriQXNet (Jahin et al. 2025, arXiv:2407.06658v3)

| Metric | Ours | TriQXNet |
|---|---|---|
| RMSE t+1 (val) | 10.14 nT | — |
| RMSE t+1 (held-out test) | 9.82 nT | — |
| RMSE t0/t+1 avg | n/a (no t0) | 9.27 nT |

We lag ~0.5–0.9 nT at t+1. **Confirmed from the paper (full extraction): same dataset (MagNet 2021, ACE+DSCOVR), same 29 features (14 vars × mean+std + smoothed_ssn), same hourly mean+std aggregation, same 70/20/10 split.** Real differences = input window (their 128h vs our 48h), horizon (their t0+t+1 vs our t+1…t+6), arch (3-pipeline classical + 4-qubit quantum vs our BiLSTM+attn), and they add conformal UQ + XAI + 10-fold CV.

**TriQXNet's own gaps (grounded in their numbers — full table in `PROGRESS_REPORT.md` §1):**
- **9.27 is quiet-dominated + t0-inflated.** Their extreme-event case study: t0 RMSE **20.33 nT**, t+1 **20.86 nT** — error ~doubles on the storms that matter. Only ONE extreme event evaluated; no storm-conditional table.
- **1h horizon only**; t0 is trivial nowcasting that pulls the average down.
- **Quantum branch marginal/fragile:** 9.65→9.27 (~4%, 0.38 nT), non-monotonic in qubits (4q 9.75, 6q 9.91, 8q 9.70), simulator-only.
- **UQ at t+1 only** — no error-growth-with-horizon.

**Our complementary position:** we do t+1…t+6 and report the full storm-conditional table honestly (intense t+6 44.64 nT — bad but measured); we lack UQ (→ Phase C). Note: Step 3 proved the t+1 gap is NOT primarily window length (96/128h closed only ~0.2 nT). Full head-to-head in `PROGRESS_REPORT.md`.

## A4 — Benchmark Sweep (5 seeds × 6 architectures, plain MSE, full val set)

| Architecture | t+1 | t+6 |
|---|---|---|
| LSTM | 10.524±0.115 | 11.844±0.058 |
| Stacked BiLSTM | 9.971±0.207 | 11.438±0.108 |
| BiLSTM+BiGRU | 9.907±0.093 | 11.433±0.070 |
| CNN+BiLSTM | 9.892±0.141 | 11.307±0.047 |
| **Conv1DNet** | **9.935±0.151** | **11.182±0.148** ← best t+6 |
| SolarAttentionLSTM | 10.121±0.305 | 11.441±0.228 |

Saved to `benchmark_table.csv`. All within 1–2σ. Conv1DNet best at t+6 with lowest absolute RMSE; CNN+BiLSTM has lowest variance. **SolarAttentionLSTM has the highest variance (±0.228)** — attention adds noise, not skill, at this dataset size.

## Phase 3 & 4 (ensemble layers — honest held-out test eval, Step 1, 2026-05-23)

**Headline: ensemble layers do NOT beat plain BiLSTM on the held-out test set. The earlier sub-7 nT numbers were a val-subset artifact.** Alphas selected on val-subset (last 20% of val), applied unchanged to held-out test (no leakage).

| Config | val-subset t+6 (old "headline") | **held-out test t+6 (honest)** |
|---|---|---|
| BiLSTM base | 7.03 | **11.26** ← best aggregate |
| + LGBM blend | 6.96 | 11.49 (−0.23 worse) |
| + GRU correction | 7.07 | 11.31 (−0.05 worse) |
| + GRU + LGBM combined | 7.02 | 11.56 (−0.30 worse) |

LGBM-alone test t+6 = 12.63 nT (worse than BiLSTM at every step).

**Storm-conditional on held-out test (t+6):**
| Bin | N | BiLSTM | +GRU | +Combined |
|---|---|---|---|---|
| Quiet (≥−20) | 11900 | 7.94 | 7.56 | **7.30** ← ensemble helps |
| Moderate (−50..−20) | 1687 | 12.16 | 12.86 | 13.02 ← hurts |
| Intense (<−50) | 387 | 44.64 | 46.27 | 49.45 ← hurts most |

**Why val-subset lied:** the last 20% of val is a quiet stretch (few intense storms), so RMSE there (~7 nT) is far below the storm-rich held-out test (~11 nT). Comparing 6.96 vs full-val 11.57 implied a 40% error cut — false.

**Why aggregate worse despite quiet gain:** GRU/LGBM shift error from quiet → storms. Intense bin degradation (44→49 nT, squared, N=387) outweighs the quiet improvement (7.94→7.30, N=11900) in the MSE numerator. Wrong direction — storms are exactly where skill is needed.

**Conclusion:** BiLSTM-alone (test t+6 = 11.26 nT) is the honest best aggregate. Ensemble stack adds complexity, no aggregate skill. GRU still useful as a quiet-time bias corrector if a regime-split deployment is built, but not as a blanket layer. Real lever is architecture (Step 2: Conv1DNet) or data (Step 4: OMNI), not post-hoc stacking.

## Persistence Baseline (post-Phase-A)

| Step | Persistence | BiLSTM | BiLSTM+GRU | Skill(+GRU) |
|---|---|---|---|---|
| t+1 | 4.03 | 10.14 | 9.73 | −141.1% |
| t+2 | 6.60 | 10.11 | 9.78 | −48.1% |
| t+3 | 8.38 | 10.33 | 10.08 | −20.3% |
| t+4 | 9.69 | 10.71 | 10.48 | −8.2% |
| t+5 | 10.78 | 11.11 | 10.87 | −0.9% |
| t+6 | 11.71 | 11.57 | 11.23 | **+4.1%** |

Storm-conditional at t+6 (BiLSTM+GRU vs persistence):
- Quiet: 8.47 vs 7.69 nT (−10.2% skill, worse)
- Moderate: 14.67 vs 15.74 nT (+6.8%)
- Intense: 40.25 vs 48.66 nT (**+17.3%**)

Same qualitative story as pre-Phase-A: model only wins at t+6 and only meaningfully on intense storms. Per-horizon best strategy: persistence for t+1–t+5, BiLSTM+GRU for t+6.

---

## Phase C — Uncertainty Quantification (conformal prediction, 2026-05-25)

Post-hoc split-conformal on the trained BiLSTM (**no retrain**), library `crepes` 0.9.0 (same as TriQXNet). Calibrate on **full val residuals** (fit=train / calibrate=val / evaluate=test). Cell `phase_c_uq` (after A3 cell `37d6b8b0`); writes `conformal_intervals.csv` + `conformal_band_plot.png`. Our value-add over TriQXNet (UQ at t+1 only): **per-horizon t+1…t+6 + per-storm-bin + storm-adaptive (Mondrian)**.

**Marginal per-horizon coverage(%) / width(nT), held-out test:**
| Step | cov@90 | w@90 | cov@95 | w@95 | cov@99 | w@99 |
|---|---|---|---|---|---|---|
| t+1 | 92.6 | 29.4 | 96.8 | 40.0 | 99.3 | 68.8 |
| t+6 | 93.0 | 33.5 | 97.0 | 44.5 | 99.3 | 75.4 |

Coverage ≈ nominal (slightly conservative). **Width grows monotonically t+1→t+6** (40.0→44.5 @95%) = error-growth-with-lead-time, which TriQXNet's t+1-only UQ cannot show.

**Per-storm-bin coverage @95% (t+6), Marginal vs Mondrian (bins = PREDICTED severity):**
| Bin (by actual) | N | marg cov% | marg w | mond cov% | mond w |
|---|---|---|---|---|---|
| Quiet (≥−20) | 11900 | 98.6 | 44.5 | 97.7 | 36.5 |
| Moderate (−50..−20) | 1687 | 93.1 | 44.5 | 91.7 | 54.9 |
| Intense (<−50) | 387 | **64.3** | 44.5 | **82.9** | 107.5 |

**Marginal undercovers intense storms (64.3% vs 95%)** — constant width sized by quiet-dominated calibration. **Mondrian recovers most (→82.9%)** by widening storm bands (44.5→107.5 nT) + tightening quiet (→36.5). Residual gap = **missed onsets** (model predicts quiet, storm hits → row in narrow quiet bin; Mondrian-by-predicted can't fix what the model doesn't see coming). Same storm-scarcity ceiling as Phase B, now in interval form. **Conclusion:** UQ shipped, honest + per-horizon; biggest TriQXNet gap closed at zero retrain cost.

---

## Progress Log

### Phase E — Completed: 10-fold blocked CV + paired significance tests (2026-06-01)
Added the significance protocol TriQXNet has and we lacked (they use 10-fold CV paired t-tests; we had only 5-seed mean±std). Cell `cv_significance` after `omni1m_multiseed`. **Leakage-safe blocked folds** (NOT plain random 10-fold): 139,713 pooled sequences → 10 contiguous chronological blocks/period; fold-train drops a 54-seq (`SEQ_LEN+FORECAST_HORIZON`) guard band each side of the test block so no train window shares raw hours with the test block. Per fold: refit scaler, train base + aug (fold-train + 1389 1-min OMNI seqs) at the **same seed → paired**, storm-weighted loss + early stop. Metric = t+6 RMSE on identical fold-test. 20 trainings, ~4.3 h on RTX 4060. Outputs `cv_significance_folds.csv` + `cv_significance_tests.csv`.
- **Results (paired-t / Wilcoxon across 10 folds, meanΔ>0 = first model wins, lower RMSE better):**
  - agg `aug−base` +0.260 nT, p=0.105 / W=0.232 — **ns** (matches the documented tradeoff: aug ≈flat on aggregate).
  - intense `aug−base` **+3.011 nT, p=0.0536 / W=0.0488** — **aug significant on Wilcoxon**, paired-t just misses 0.05. Aug ≤ base intense in 8/10 folds. **Phase D adoption holds under a stricter protocol.**
  - agg `base−persist` −0.975 nT, **p=0.016** — persistence beats model on aggregate (significant); confirms quiet-dominated story.
  - intense `base−persist` **+5.118 nT, p=0.0069** — model beats persistence on intense storms (highly significant, the metric that matters).
- **Why the intense aug gain (+3.01) is smaller than the held-out 5-seed +8.36:** CV folds are heterogeneous, train on smaller/shifted data per fold, and per-fold intense N ranges 231–727 — CV is the more conservative estimator. Direction robust; magnitude compresses. CV RMSEs also run higher than the fixed held-out test (storm-rich blocks, less train data/fold) — not directly comparable to the held-out split by design.
- **Decision: significance protocol now matches TriQXNet's bar.** Headline claims that clear it: model >> persistence on intense (p=0.007) and aug > base on intense (Wilcoxon p=0.049). Aug-aggregate and persistence-aggregate are honest expected non-wins. No checkpoint touched (CV writes only CSVs). Full tables in `results.md` §"Phase E". **Cell inserted + executed standalone via the dependency-chain runner (env conda `dnn`); outputs are in the CSVs, not yet embedded in the in-notebook cell — re-run `cv_significance` in Jupyter to populate cell outputs.**

### Phase D-2 — Completed: 25+ storms tested — REGRESSED, NOT adopted. 15-storm stays primary (2026-05-26)
Tested the Phase D "more headroom" follow-up: expanded `omni1m_build` STORMS_1M from 15 → **27** major storms (Dst<−100, 2000–2024). Added 12 events (aug2000 −234, apr2001 −271, sep2002 −176, jul2004 −170, jan2005 −103, sep2005 −139, mar2012 −145, mar2013 −132, dec2015 −166, mar2023 −163, apr2023 −213, oct2024 −333) — all DST-verified to carry real intense hours (24–110 each) before running. Re-ran `omni1m_multiseed` (5-seed paired, same base seeds → base reproduced exactly: agg 11.418±0.222, intense 44.46±1.03).
- **27-storm aug held-out test t+6:** aggregate 11.077±0.447 (Δ **+0.341±0.275**); intense (N=387) 38.89±3.75 (Δ **+5.57±3.68**). All 5 seeds positive on both — so 27-storm still beats base.
- **But WORSE than the 15-storm adopted model on both metrics:** intense Δ +5.57 vs **+8.36** (−2.8 nT), aggregate Δ +0.341 vs **+0.48** (−0.14 nT). Aug variance also up (intense std 3.75 vs 3.39, agg 0.447 vs 0.28). More storms = noisier, not better.
- **Why:** the 12 additions span solar-min years (jan/sep2005, mar2012/13 — smoothed_ssn 40–80) + 2023/24, and several are weaker (Dst −103 to −176) vs the curated 15 that included Gannon −406. Heterogeneous solar-cycle/instrument provenance + diluted extreme tail = noise ≈ signal at the margin. Same Step-4 lesson re-confirmed: **quantity ≠ quality for the storm tail; curation beats count.**
- **Decision: REJECT 27-storm. Keep 15-storm `solar_bilstm_omni1m_model.pth` adopted.** Multiseed writes nothing to disk → adopted checkpoint untouched, no action needed. Cell reverted to the 15-storm list (notebook reproduces the adopted model). The storm-scarcity lever has a ceiling — *the right 15* beat *27 mixed*. Do NOT re-attempt blind storm-count expansion; if revisited, curate for extreme Dst + matched solar-cycle phase, not raw count.

### Phase C (aug) — Completed: Conformal UQ recalibrated on the adopted 1-min OMNI model (2026-05-26)
Follow-up to Phase D: recalibrated split-conformal on the aug model (`solar_bilstm_omni1m_model.pth`), **no retrain**. New cell `phase_c_uq_aug` after `phase_c_uq`. Self-contained: refits StandardScaler on base-train+`X_omni1m` (deterministic → reproduces the aug training scaler), loads both checkpoints, infers val+test for base & aug, calibrates on each model's val residuals, A/Bs intervals on held-out test. Outputs `conformal_intervals_aug.csv`, `conformal_base_vs_aug_bins.csv`.
- **Intense @95% t+6:** marginal coverage base 64.3 → aug **73.4%** (near-equal width 44.5→45.6 nT = residuals genuinely tighter); Mondrian coverage base 82.9 → aug **90.4%** (near nominal). Aug sees onsets better → more intense rows land in the predicted-intense bin → missed-onset gap shrinks. Mondrian intense width grows 107.5→122.4 (intense calibration bin now holds more genuine storm residuals — not a regression).
- **Cost:** +~1 nT aggregate width at t+6; moderate coverage dips 91.7→89.5%. Mirrors the point-forecast tradeoff. Net: clear UQ win on storms (the largest base-model UQ gap), small give-back elsewhere.
- **Verdict:** Phase D point-forecast win carries into uncertainty. Use aug-model UQ for storm-warning deployment. Verified via standalone exec of cells 1/3/4/5/6/7 + omni1m_build + the cell logic (env conda `dnn`); cell inserted, not yet run in-notebook (outputs empty until user runs). Full tables in `results.md` §"Phase C (aug)".

### Phase D — Completed: 1-min OMNI augmentation ADOPTED — first robust storm-RMSE win (2026-05-25)
Re-did the rejected Step-4 OMNI augmentation with the fix Step 4's own post-mortem prescribed: **real 1-min `OMNI_HRO_1MIN`** (true hourly mean+std over ~60 samples/hr) instead of the 3h rolling-std proxy, and **15 storms** instead of 5. Two cells `omni1m_build` → `omni1m_multiseed` after `omni_multiseed`. Adopted checkpoint `solar_bilstm_omni1m_model.pth` (seed-42 final; base `solar_bilstm_model.pth` retained for A/B, not overwritten).
- **Data:** 15 major storms (Dst<−100) 2000–2024 incl. Gannon 2024 (−406). Features from `OMNI_HRO_1MIN` 1-min → true hourly mean+std for 14 vars; angles theta/phi derived at 1-min then aggregated (matches MagNet convention); label = `DST1800` (Kyoto Dst, same index as base). 2184 OMNI hours → **1389 sequences / 877 intense** (Step-4 proxy: 791 / 329). Train intense 3321 → 4198 (+26%).
- **5-seed paired held-out test t+6:** aggregate base 11.42±0.22 → aug 10.94±0.27 (**Δ +0.48±0.28**); intense (N=387) base 44.46±1.03 → aug 36.10±3.08 (**Δ +8.36±3.39**); moderate flat (Δ −0.19±1.04). **All 5 seeds positive** on intense (+5.4 to +13.7) and aggregate. Adopt rule (intense Δ>0 AND |Δ|>std AND agg not worse) met on every count.
- **Why it worked where Step 4 failed:** (1) true sub-hourly std replaces the proxy that injected noise; (2) 2.7× the intense sequences; (3) DST1800 label matches base provenance. The Step-4 post-mortem predicted exactly this fix — confirmed.
- **Tradeoff (per-step A/B, adopted seed-42 ckpt):** aug shifts skill toward long horizons + storms. **Wins t+4–t+6 and intense** (t+6 11.26→10.50, intense 44.64→30.94); **loses t+1–t+3 and quiet/moderate** (quiet t+6 7.94→8.51, moderate 12.16→13.54). Crossover ~t+4. Not a free lunch — but the right trade: persistence already beats the model at t+1–t+5, so the model's job is t+6 storm warning, exactly where aug wins. Deploy: base/persistence for short+quiet, omni1m-aug for t+6 storms.
- **Decision: ADOPT.** Intense storm RMSE cut **19%** (44.5→36.1 nT), the metric that matters, with aggregate t+6 also improved (robust +0.48±0.28) — the storm-data-scarcity ceiling moved for the first time. **Follow-ups:** ~~recalibrate Phase C conformal UQ on the aug model~~ ✅ DONE 2026-05-26 (Phase C (aug) above — intense coverage 64→73% marginal / 83→90% Mondrian @95%); consider overwriting primary `.pth` once downstream eval re-run on the aug model; ~~25+ storms for more headroom~~ ❌ TESTED & REJECTED 2026-05-26 — 27 storms regressed vs 15 (intense Δ +5.57 vs +8.36), see Phase D-2 above.

### Phase C — Completed: Conformal UQ added (2026-05-25)
Conformal prediction intervals on the BiLSTM primary, `crepes` 0.9.0, calibrated on full val residuals, evaluated on held-out test. New cell `phase_c_uq` after `37d6b8b0` (self-contained: needs `all_preds`/`all_actuals` from cell 9 + `test_preds_arr`/`test_actuals_arr` from cell 25). Three views: marginal per-horizon (coverage ≈ nominal, width 40.0→44.5 nT t+1→t+6 @95%), per-storm-bin (intense undercovers 64.3% @95%), Mondrian storm-adaptive by predicted severity (intense 64.3→82.9%, residual gap = missed onsets). Outputs `conformal_intervals.csv`, `conformal_band_plot.png`. No retrain; primary `.pth` untouched. env = conda `dnn` (torch 2.9.1); `crepes` installed there. Verified via standalone exec of cells 1/3/4/5/6/7 + inference + the cell (cell not yet executed in-notebook — outputs empty until user runs it). Full tables in `results.md` + Performance §Phase C above. **Next: 10-fold CV paired t-tests, then real sub-hourly OMNI (`OMNI_HRO_1MIN`).**

### Phase B Step 4 — Completed: OMNI augmentation REJECTED — single-seed mirage (2026-05-24)
Training-only data augmentation from 5 OMNI storms. Initially looked like the first lever to pass; **5-seed verification reversed it.**
- 3 cells `omni_download` → `omni_features` → `omni_augment` + verify cell `omni_multiseed`, after `b3aaa3d6`. Option B (29-feature paper format). NASA CDAWeb `OMNI2_H0_MRG1HR`, vars `*1800`. OMNI hourly → mean = hourly value, std = 3h rolling-std proxy; angles derived; smoothed_ssn hardcoded SILSO v2.
- +791 sequences (329 intense) into train only; intense 3321→3650 (+10%). Val/test base-only, scaler refit. `solar_bilstm_augmented_model.pth` = single-seed-42 ablation artifact only.
- **Single seed 42 looked great:** test t+6 11.26→11.14, intense 44.64→40.01 (+4.63). **5-seed paired re-run killed it:** intense Δ −1.73±3.79 (aug worse), aggregate Δ −0.147±0.370 (aug worse). Only seed 42 won; 43–46 all hurt intense (−1.3 to −4.9). Aug also tripled intense variance (std 1.03→3.52).
- **Decision: REJECT.** BiLSTM-base (11.26) stays primary. Do not promote. Data-scarcity ceiling real but 791 proxy-feature OMNI sequences don't clear it (too few; rolling-std/hardcoded-ssn proxies + cross-dataset provenance inject noise ≈ signal). **Process lesson: multi-seed before declaring any storm-metric win — intense N=387 gives ±4 nT seed noise.**

### Phase B Step 3 — Completed: input window 48h→128h REJECTED (2026-05-24)
Input-window ablation. SEQ_LEN ∈ {48, 96, 128}, BiLSTM primary.
- Cell `seq_ablation_step3` added after `conv1d_plainmse`. Self-contained: rebuilds per-period 70/20/10 split at each window, BATCH=32 uniform (window = sole variable), seed 42, storm-weighted loss, patience 15. Saves `solar_bilstm_seq{48,96,128}.pth` (primary `solar_bilstm_model.pth` untouched).
- **Finding:** 128h does NOT beat current primary. In-cell (BATCH=32) test t+6: 48h 11.43, 96h 11.92, 128h 11.29. Existing primary (48h, BATCH=64) = 11.26, lower than all three — batch-size delta exceeds window gain.
- **No storm benefit:** 128h intense t+6 46.24 ≈ 48h 46.20, both worse than primary's 44.64. Moderate worsens with window (12.20→12.97). Only quiet improves (7.94→7.60) — 91% of data, not where skill matters.
- **96h non-monotonic:** best t+1 (9.62), worst t+6 (11.92) + worst intense (50.78). Discarded.
- **TriQXNet gap not window-driven:** best t+1 9.62 closes only ~0.2 nT of the 0.87 nT gap.
- **Decision:** 48h stays primary. Checkpoints are ablation artifacts only. Do NOT re-attempt longer windows. Next lever = data (Step 4 OMNI).

### Phase B Step 2 — Completed: Conv1DNet REJECTED (2026-05-24)
Conv1DNet promotion attempt. Tested both losses on held-out test.
- Cells `conv1d_step2` (weighted 5×/15×) + `conv1d_plainmse` (plain MSE, A4 config) added after `3b7398e3`. Save `solar_conv1d_model.pth`, `solar_conv1d_plainmse.pth` (BiLSTM `.pth` untouched).
- Fixed bug: cell var `F = X_train_scaled.shape[2]` shadowed module-global `import torch.nn.functional as F` → `F.softmax` in `SolarAttentionLSTM.forward` hit int. Renamed → `N_FEAT`.
- **Finding:** Conv1DNet loses to BiLSTM at every test step under both losses (test t+6: BiLSTM 11.26, Conv wtd 11.66, Conv plain 11.71). Only win: weighted Conv1D on intense (42.92 vs 44.64), costs aggregate.
- **A4 win was val-only.** Val t+6 Conv1D 11.49 < BiLSTM 11.57 (matches A4), but does NOT transfer to held-out test. Treat A4 table as val-ranking only.
- **Decision:** BiLSTM primary (test t+6 11.26). Conv1D rejected, ablation only. Do not re-attempt. Next lever = data (Step 3 window / Step 4 OMNI), not architecture.

### Phase B Step 1 — Completed (2026-05-23)
Honest held-out test evaluation of ensemble layers.
- Added test-set eval blocks to cells `1cb094ba` (Phase 3) and `17143447` (Combined). BiLSTM inference on `test_loader`, meta-features from `X_test_raw`, LGBM/GRU applied, alphas from val-subset (no leakage).
- **Finding:** ensemble does not beat plain BiLSTM on held-out test (blend 11.49, +GRU 11.31, combined 11.56 vs BiLSTM 11.26 nT at t+6). Sub-7 nT val-subset numbers were a quiet-tail artifact.
- GRU/LGBM trade quiet-time error (7.94→7.30) for worse storm error (intense 44.6→49.5) — wrong direction.
- **Decision:** BiLSTM-alone (test t+6 11.26 nT) is the honest baseline. Ensemble dropped as blanket layer. Next lever = architecture (Conv1DNet) or data (OMNI).

### Phase A — Completed (2026-05-22)
Parity work toward TriQXNet-style evaluation protocol.
- Feature set expanded 15 → **29 (paper)**: 14 vars × (mean+std) + smoothed_ssn
- Split changed from global 80/20 → **per-period 70/20/10**; new held-out test set
- A1: Zenodo auto-download added (`bb365e9d`)
- A3: t+1 comparison to TriQXNet + held-out test eval (`37d6b8b0`)
- A4: 5-seed × 6-architecture benchmark sweep (`a3c0b28f`, `8cb1d904`, `478748ae`, `95fd2fda`)
- Full retrain from scratch (old `.pth` checkpoints architecturally incompatible — `input_dim` mismatch)
- **Results:** val t+6 RMSE 11.5731 nT, Pearson r 0.7755, R² 0.5822 (BiLSTM base). Test t+6 RMSE 11.26 nT. Conv1DNet wins A4 at t+6 (11.18 nT). See sections above for full tables.
- Saves to `solar_bilstm_model.pth`, `gru_corrector.pth`, `benchmark_table.csv`, `results.md`

### Phase 1 — Completed (2026-04-26)
- Added full 6-step inference collection (`all_preds`, `all_actuals`)
- Added per-step RMSE/Pearson r/R² and storm-conditional RMSE tables
- Pre-Phase-A results: RMSE 8.85 nT, Pearson r 0.696, R² 0.358 (t+6) on old 80/20 split

### Phase 2 — Completed (2026-04-27)
- BiLSTM upgrade: `SEQ_LEN` 24→48h, `HIDDEN_DIM` 64→128, `DROPOUT` 0.4→0.3, `BIDIRECTIONAL=True`
- Two-tier weighted loss (5× moderate, 15× intense), gradient clipping, patience 15
- Pre-Phase-A results: RMSE 8.4531 nT, Pearson r 0.7201, R² 0.4141 (t+6) on old 80/20 split

### Phase 3 — Completed (2026-04-27)
- LightGBM stacked on BiLSTM predictions; one regressor per step
- Pre-Phase-A: blend t+6 RMSE 8.00 nT (last 20% of val), best α ~0.33

### Combined Ensemble — Completed (2026-04-27)
- BiLSTM → GRU correction → LGBM blend
- Pre-Phase-A: t+6 RMSE 8.3234 nT (last 20% of val), best α 0.60

### Phase 4 — Completed (2026-04-27)
- `ResidualGRU` (1-layer GRU hidden=32 → Linear 32→6) trained on training-set residuals via `shuffle=False` loader
- Pre-Phase-A: full-val t+6 RMSE 7.775 nT, delta +0.68 nT vs LSTM alone
- Key finding: improvement is mean-bias correction (residual mean +3.15 nT), not complex structure

---

## Key Observations & Negative Results (DO NOT REPEAT)

### More OMNI storms (27 vs 15) does NOT help — curation beats count (2026-05-26)

Phase D-2 tested expanding the adopted 1-min OMNI augmentation from 15 → 27 storms (12 DST-verified Dst<−100 additions). 5-seed paired re-run (base reproduced exactly): 27-storm aug beats base but **regresses vs the 15-storm adopted model** — intense t+6 Δ +5.57±3.68 vs the 15-storm +8.36±3.39 (−2.8 nT), aggregate Δ +0.341±0.275 vs +0.48±0.28, and aug variance rose (intense std 3.39→3.75). The 12 additions span solar-min years (smoothed_ssn 40–80) + 2023/24 and several are weaker (Dst −103 to −176) vs the curated 15 (incl. Gannon −406); heterogeneous solar-cycle/instrument provenance + diluted extreme tail injects noise ≈ signal at the margin. **The storm-scarcity lever has a ceiling: the right 15 beat 27 mixed.** Do NOT re-attempt blind storm-count expansion. If revisited, curate for extreme Dst + matched solar-cycle phase, not raw count. Cell `omni1m_build` reverted to the 15-storm list. Full numbers: Progress Log §Phase D-2 + `results.md`.

### Longer input window (96h/128h) does not help — 48h is the right window (2026-05-24)

Step 3 ablation (cell `seq_ablation_step3`) tested SEQ_LEN ∈ {48, 96, 128} on the held-out test set, pure ablation (window = sole variable, BATCH=32 uniform, seed 42). **128h does not beat the 48h primary**: in-cell test t+6 is 48h 11.43 / 96h 11.92 / 128h 11.29, but the existing 48h primary at BATCH=64 is 11.26 — lower than all three. The batch-size effect alone outweighs any window gain. 128h is also worse on intense storms (46.24 vs primary 44.64) and on moderate (monotonically worse with window); its only gain is the quiet bin (91% of data, where persistence already wins). 96h is non-monotonic — best t+1 but worst t+6 and worst intense. **The TriQXNet t+1 gap (0.87 nT) is NOT a window-length artifact** — the best window-tuned t+1 (9.62) closes only ~0.2 nT. Do NOT re-attempt longer windows; the lever is data (intense scarcity), not input length. Checkpoints `solar_bilstm_seq{48,96,128}.pth` are ablation artifacts only.

### OMNI augmentation does not robustly help — single-seed mirage (2026-05-24)

Adding 791 sequences (329 intense) from 5 OMNI storms to train (29-feature paper format, 3h rolling-std proxy for std cols, hardcoded SILSO-v2 smoothed_ssn, derived field angles) **does not robustly improve storm RMSE.** Seed 42 alone showed test t+6 intense 44.64→40.01 (−4.63 nT) and looked like the first Phase B lever to pass — but a 5-seed paired re-run (cell `omni_multiseed`) gave intense Δ **−1.73±3.79 nT** and aggregate Δ **−0.147±0.370 nT**, both *worse* on average with std swamping the mean. Only seed 42 won; seeds 43–46 all hurt intense (−1.3 to −4.9). Augmentation also tripled intense-bin variance (std 1.03→3.52). Likely causes: too few sequences vs a 98k train set, proxy-feature distribution mismatch (rolling-std ≠ true sub-hourly std; monthly ssn hardcoded), and cross-dataset (OMNI vs MagNet) provenance — noise ≈ signal. **Do not re-attempt with the hourly-proxy approach.** If revisited, use 1-min `OMNI_HRO_1MIN` aggregated to *real* mean+std and many more storms. **[UPDATE 2026-05-25: done in Phase D — exactly this fix (real 1-min mean+std, 15 storms) ADOPTED, intense Δ +8.36±3.39 nT robust across 5 seeds. The proxy was the problem, not OMNI itself. See Phase D in Progress Log.]** **Reusable lesson: ALWAYS multi-seed before declaring a storm-metric win — intense N≈387 gives ±1–3.5 nT seed-to-seed RMSE noise, enough to fake a ±4 nT "improvement" on one seed.** This nearly produced a false ADOPTED in the docs.

### A4 architecture ranking is VAL-only — not test-truth (2026-05-24)

A4 benchmark sweep (cell `95fd2fda`) ranked 6 architectures on the **val set**. Conv1DNet "won" t+6 there (11.18 vs SolarAttentionLSTM 11.44). Step 2 promoted Conv1DNet and evaluated on the **held-out test set**: it LOST to BiLSTM at every step under both weighted and plain-MSE loss (test t+6 11.66 / 11.71 vs BiLSTM 11.26). Val ranking did not transfer. **Do not pick the primary architecture from the A4 table — it is a val-ranking only. Any architecture decision must be confirmed on the held-out test set.** Conv1DNet is rejected; do not re-attempt its promotion. See Step 2 in Next Steps for full tables.

### Storm Bias Investigation (2026-04-27)

BiLSTM has condition-dependent bias (measured pre-Phase-A; pattern likely similar post-Phase-A):
- Quiet: −2.89 nT (overshoots)
- Moderate: +4.07 nT (undershoots magnitude)
- Intense: +10.55 nT (severely undershoots)
- Per-step bias grows +0.95 → +2.23 nT t+1→t+6

**Three fixes tried — all failed:**

1. **Linear recalibration** — fit `actual = a*pred + b` per step. Train/val patterns differ; one transform cannot fix both. Helped quiet (+0.88 nT), hurt intense (−3.53 nT).
2. **Directional asymmetric loss** — 2.5× extra penalty when overpredicting during storms (effective 12.5× moderate, 37.5× intense). Result: overall RMSE 8.45→9.54 nT, quiet destabilised, intense improvement marginal.
3. **Oversampling** — moderate 4×, intense 15× in training. Result: everything worse (overall 8.45→9.67 nT, intense 20.69→27.01 nT). 15× repetition of 145 samples = severe overfitting.

**Conclusion:** Storm performance is data-scarcity-bound. Loss reweighting and oversampling cannot bridge it. Requires more training data or physics-informed prior.

### Persistence Baseline Analysis (2026-04-28, updated 2026-05-22)

BiLSTM is worse than persistence at t+1–t+5; only beats persistence at t+6. Model's real value is intense-storm detection (+17.3% to +29.3% skill on intense events depending on Phase). Aggregate RMSE penalised by being worse than persistence during quiet times (~91% of data).

### Hybrid Switching Experiment (2026-04-28, re-run 2026-05-22)

Condition-dependent switching (persistence when `last_dst ≥ threshold`, BiLSTM+GRU otherwise). Grid-searched thresholds −5 to −40 nT. Pre-Phase-A best threshold −40 nT gave 7.98 nT (worse than pure 7.78 nT). Post-Phase-A best threshold −35 nT gave 11.58 nT (vs pure 11.23 nT) — still worse. Root cause: storm onset timing mismatch — a storm peaking at Dst = −60 nT at t+6 often has `last_dst = −25 nT` at prediction time, routed to persistence which is terrible during onset.

**Conclusion:** For real-time deployment, output persistence for t+1–t+5 and BiLSTM+GRU for t+6. No threshold needed — switchover is purely by horizon. Do NOT tune thresholds further; the mismatch is structural.

### Physics-Informed Feature Engineering (2026-04-28)

Added 5 solar wind coupling functions (`bz_south`, `vBs`, `clock_angle`, `epsilon`, `newell`) → 20 features. Pre-Phase-A t+6 RMSE 8.6117 nT (worse by −0.16 nT). Worse at every step and every storm bin.

**Verdict:** Physics couplings are derived nonlinear transforms of raw inputs the BiLSTM already learns implicitly. Explicit redundant features add collinearity, not signal. Do not add.

---

## Next Steps — Plan (added 2026-05-23)

### ✅ PHASE B COMPLETE (2026-05-24) — all four levers evaluated, three rejected.
- **Step 1** (ensemble honest eval) — DONE. Ensemble does not beat BiLSTM-base on held-out test.
- **Step 2** (Conv1DNet arch) — REJECTED. Won on val, lost on test.
- **Step 3** (96/128h window) — REJECTED. Longer window no help.
- **Step 4** (OMNI augmentation) — REJECTED. Single-seed mirage; 5-seed verify killed it.
- **Step 5** (doc pass) — DONE. results.md + CLAUDE.md + `PROGRESS_REPORT.md` updated.

**Net Phase B finding:** BiLSTM-base (test t+6 11.26 nT) stays primary. Architecture, window, and post-hoc stacking are NOT the lever; storm-data scarcity is the ceiling. Do not re-attempt any Phase B lever (see Negative Results).

### ✅ PHASE C — Uncertainty Quantification — COMPLETE (2026-05-25)
Conformal prediction intervals added to the BiLSTM primary (`crepes` 0.9.0, calibrate on full val residuals, evaluate on held-out test). Cell `phase_c_uq`. See Performance §"Phase C" + `results.md` for full tables. Headline: coverage ≈ nominal, **width grows t+1→t+6** (40.0→44.5 nT @95%); marginal undercovers intense (64.3% @95%), **Mondrian-by-predicted-severity recovers to 82.9%** (residual gap = missed onsets). Biggest TriQXNet gap closed, zero retrain.

**→ Next levers:**
- ~~**10-fold CV paired t-tests** for significance (match TriQXNet's bar).~~ ✅ **DONE — Phase E, 2026-06-01.** Leakage-safe blocked 10-fold. Intense `aug−base` +3.01 nT (Wilcoxon p=0.049, sig); intense `base−persist` +5.12 nT (p=0.007, sig). Aggregate aug-gain ns (matches tradeoff). Phase D adoption holds under the stricter protocol. See Progress Log §Phase E + `results.md`.
- ~~**Real sub-hourly OMNI** (1-min `OMNI_HRO_1MIN`, true mean+std, many more storms)~~ ✅ **DONE — Phase D, ADOPTED 2026-05-25.** Intense t+6 44.5→36.1 nT (Δ +8.36±3.39, 5-seed robust); aggregate 11.42→10.94. The storm-scarcity lever worked. Next within this lever: more storms (25+) + recalibrate UQ on aug model.
- **(Optional) normalized conformal** with a difficulty estimator (e.g. recent solar-wind volatility) instead of discrete Mondrian bins — smoother adaptive width, may close more of the intense undercoverage.

---

#### Phase B archive (historical — do not re-execute; steps below are all closed)
Post-Phase-A state established. Phase B = honest re-evaluation + architectural sweep using A4 winners. Steps 1→4 were executed in order; outcomes recorded in each step header and the Progress Log.

### Step 1 — Fix Phase 3 / Combined ensemble evaluation — ✅ DONE (2026-05-23)

**Outcome.** Held-out test eval added to cells `1cb094ba` and `17143447`. Result: ensemble layers do NOT beat plain BiLSTM on the held-out test (blend 11.49, +GRU 11.31, combined 11.56 vs BiLSTM 11.26). The sub-7 nT numbers were a val-subset artifact (quiet tail of val). See `## Phase 3 & 4` section for full tables. **Decision:** BiLSTM-alone is the honest baseline; do not ship the ensemble as a blanket layer. Proceed to Step 2.

**Problem (original).** Cells `1cb094ba` (Phase 3 LGBM) and `17143447` (Combined BiLSTM+GRU+LGBM) currently split the val set 80/20 and report RMSE on the inner 20%. With the new per-period split (cell `0d486255`), the val set is 27 942 sequences and the LGBM-test subset is 5 589 — small, and not the **held-out test set** (which is 13 974 sequences from cell `0d486255`'s `test_loader`).

This produces apples-to-oranges numbers:
- Reported Phase 3 t+6 = 6.96 nT (last 20% of val)
- Reported Combined t+6 = 7.09 nT (last 20% of val)
- Reported BiLSTM base t+6 = 11.57 nT (full val)

Reader naturally compares 7.09 vs 11.57 and concludes the ensemble cut error by 39 %. False — the splits differ.

**Fix.**
1. In cell `1cb094ba`: keep LGBM training on a chronological split *within val* (must, since `all_preds`/`all_actuals` live in val) — but **also evaluate the trained LGBM and the blend on `test_loader`**. Requires generating BiLSTM predictions on the test set first: replicate the inference block from cell `af6ee96c` against `test_loader` → `test_preds`, `test_actuals`. Build matching meta-features from `X_test_raw` and call `lgbm.predict` per step.
2. Same change in cell `17143447`: extend GRU correction inference to `test_loader`, then feed GRU-corrected test predictions into the LGBM stack, then blend.
3. Report **two** tables per cell: (a) the existing val-subset table labelled clearly, and (b) the new held-out test table — this is the headline number for any paper/PR/comparison.

**Variables available** (set by cell `0d486255` post-Phase-A): `test_loader`, `X_test_raw`, `y_test`, `X_test_scaled`. If absent in current state, the cell was edited without re-running — re-run `0d486255` first.

**Expected outcome.** Honest Phase 3 / Combined t+6 numbers on held-out test set. Best guess: 10.5–11.0 nT (modest improvement over BiLSTM-alone test t+6 = 11.26 nT). If sub-10 nT on test, that is genuinely strong; if higher than 11.26 nT, the ensemble is overfitting val.

**Files touched.** `SolarWindLSTM.ipynb` only.

### Step 2 — Conv1DNet as new base model — ❌ REJECTED (2026-05-24). BiLSTM stays primary.

**Outcome.** Conv1DNet loses to BiLSTM on held-out test under BOTH losses, at every step. A4's Conv1D t+6 win (11.18 vs 11.44) was a **val-only artifact** — does not transfer to held-out test. Cells `conv1d_step2` (weighted loss) + `conv1d_plainmse` (plain MSE, A4 config) added after `3b7398e3`. Checkpoints `solar_conv1d_model.pth`, `solar_conv1d_plainmse.pth` (BiLSTM `.pth` untouched).

**Held-out test, per-step RMSE (nT):**
| Step | BiLSTM | Conv1D wtd | Conv1D plain |
|---|---|---|---|
| t+1 | **9.82** | 10.28 | 9.97 |
| t+2 | **9.66** | 10.24 | 9.76 |
| t+6 | **11.26** | 11.66 | 11.71 |

**Storm-conditional test t+6 (nT):**
| Bin | N | BiLSTM | Conv wtd | Conv plain |
|---|---|---|---|---|
| Quiet (≥−20) | 11900 | **7.94** | 8.71 | 8.06 |
| Moderate (−50..−20) | 1687 | **12.16** | 12.97 | 12.84 |
| Intense (<−50) | 387 | 44.64 | **42.92** | 47.24 |

**Findings.**
1. BiLSTM wins aggregate test t+6 (11.26) regardless of Conv1D loss choice. Plain-MSE Conv1D (11.71) is even worse than weighted (11.66) at t+6.
2. Only place any Conv1D wins: weighted-loss Conv1D on intense (42.92 vs 44.64) — but costs quiet (7.94→8.71) and aggregate. Plain-MSE Conv1D is worst on intense (47.24).
3. Val told a different story (Conv1D val t+6 11.49 < BiLSTM 11.57) — consistent with A4. **The A4 benchmark sweep ranked architectures on val only; that ranking does not hold on held-out test.** Treat A4 table as val-ranking, not test-truth.

**Decision: BiLSTM primary (test t+6 11.26 nT). Conv1DNet rejected — ablation only.** Do NOT re-attempt Conv1D promotion. GRU-on-Conv1D residuals abandoned. Next lever = data (window length / OMNI), not architecture.

**Why (original rationale — see Findings above for why it didn't pan out).** A4 benchmark (cell `95fd2fda`, `benchmark_table.csv`):

| Arch | t+6 |
|---|---|
| Conv1DNet | **11.182±0.148** |
| CNN+BiLSTM | 11.307±0.047 |
| SolarAttentionLSTM (ours) | 11.441±0.228 |

Conv1DNet wins by 0.26 nT at t+6 with 1/6 the variance of SolarAttentionLSTM. CNN+BiLSTM has the **lowest** variance (±0.047) — possibly more reliable in practice. Attention adds noise, not skill, at this dataset size.

**What to do.**
1. **Promote Conv1DNet (or CNN+BiLSTM) to the main training cell** `8a4080ce`. The class is already defined in cell `a3c0b28f` — import / reuse it. Replace `SolarAttentionLSTM(...)` construction with `Conv1DNet(input_dim=..., output_dim=FORECAST_HORIZON)` (verify constructor signature in `a3c0b28f`).
2. **Train with the same storm-weighted loss** (5× moderate, 15× intense), gradient clipping, patience=15. The A4 sweep used plain MSE for fairness — weighted loss should improve intense-storm RMSE further.
3. Save checkpoint as `solar_conv1d_model.pth` (do not overwrite `solar_bilstm_model.pth`). Decision: which checkpoint becomes the primary will depend on test-set numbers from step 1 (run step 1 first against old BiLSTM, then re-run step 1 against the new Conv1DNet).
4. **Rerun the full eval stack** against the new checkpoint: cells `af6ee96c` (inference), `446d08ce`/`3b7398e3` (per-step + storm tables), `317731c2` (GRU correction — note: GRU was tuned on BiLSTM residuals; retrain it on Conv1DNet residuals), `1cb094ba`/`17143447` (LGBM, Combined — using the fixed test-set eval from step 1).
5. **A/B comparison table** in `results.md`: BiLSTM base vs Conv1DNet base, both on the held-out test set, per-step + storm-conditional.

**Risk.** Conv1DNet may not benefit from the storm-weighted loss the way BiLSTM does — CNNs respond differently to class reweighting. If aggregate RMSE worsens, fall back to plain MSE for Conv1DNet only.

**Decision criterion.** Adopt Conv1DNet as primary if test-set t+6 RMSE is ≥0.2 nT better than BiLSTM **and** intense-storm RMSE is not worse. Otherwise keep BiLSTM, report Conv1DNet as ablation only.

### Step 3 — Increase input window 48h → 128h — ❌ REJECTED (2026-05-24). 48h stays primary.

**Outcome.** Window increase does NOT beat the existing 48h primary. Ablation cell `seq_ablation_step3` (inserted after `conv1d_plainmse`) trains BiLSTM at SEQ_LEN ∈ {48, 96, 128} — pure ablation, BATCH=32 uniform (window = sole variable, avoids 128h OOM), seed 42, storm-weighted loss, per-period 70/20/10 split. Saves `solar_bilstm_seq{48,96,128}.pth` (primary `solar_bilstm_model.pth` untouched).

**Held-out test, per-step RMSE (nT):**
| Step | 48 | 96 | 128 |
|---|---|---|---|
| t+1 | 9.80 | **9.62** | 9.73 |
| t+2 | 9.64 | 9.76 | **9.62** |
| t+3 | 9.88 | 10.18 | **9.80** |
| t+4 | 10.36 | 10.75 | **10.23** |
| t+5 | 10.89 | 11.38 | **10.77** |
| t+6 | 11.43 | 11.92 | **11.29** |

**Storm-conditional test t+6 (nT):**
| Bin | N | 48 | 96 | 128 |
|---|---|---|---|---|
| Quiet (≥−20) | 11900 | 7.94 | 7.85 | **7.60** |
| Moderate (−50..−20) | 1687 | **12.20** | 12.59 | 12.97 |
| Intense (<−50) | 387 | **46.20** | 50.78 | 46.24 |

**Findings.**
1. **128h does not beat the current primary.** In-cell BATCH=32, so the fair in-cell baseline is 48h = 11.43 (t+6). But the existing primary `solar_bilstm_model.pth` (48h, BATCH=64) is test t+6 **11.26** — lower than all three in-cell runs, including 128h (11.29). The batch-size delta alone exceeds the window gain. 128h vs current primary = **+0.03 nT worse aggregate**.
2. **No storm benefit.** Intense at t+6: 48h 46.20 ≈ 128h 46.24 (flat); current primary intense was 44.64 → 128h is **worse on intense**. Moderate gets monotonically worse with window (12.20→12.97). 128h's only gain is the quiet bin (7.94→7.60) — but quiet = 91% of data where persistence already wins; not where skill is needed.
3. **96h is non-monotonic** — best t+1 (9.62) but **worst** t+6 (11.92) and worst intense (50.78). Discard outright.
4. **Window is not the TriQXNet lever.** Gap was 0.87 nT (our 10.14 vs 9.27 at t+1). Best window-tuned t+1 = 9.62 closes only ~0.2 nT, far short of the full gap. The remaining gap is data (intense scarcity) / training setup, not input length.

**Decision criterion** (adopt if test t+6 clearly better AND intense not worse): 128h fails both — 11.29 ≥ 11.26 and intense 46.24 > 44.64. **REJECT. 48h stays primary.** Checkpoints kept as ablation artifacts only; do NOT re-attempt longer windows. Next lever = data (Step 4 OMNI).

**Original rationale (did not pan out — see Findings).** A3 (cell `37d6b8b0`): our t+1 = 10.14 nT, TriQXNet t+1 = 9.27 nT (gap 0.87 nT). TriQXNet uses 128h input window vs our 48h — window length was the most obvious uncontrolled variable. Pure ablation expected t+1 to drop 0.3–0.7 nT; actual drop ~0.2 nT, and aggregate/storm metrics did not improve over the 48h primary.

### Step 4 — OMNI data augmentation — ❌ REJECTED (2026-05-24). Single-seed mirage; no robust gain. BiLSTM-base stays primary.

**Outcome.** OMNI augmentation does **NOT** robustly help. The single-seed-42 result (intense −4.63 nT, "first lever to pass") was a lucky draw — a 5-seed paired re-run shows aug is **worse on average** on both intense and aggregate, with variance that swamps the mean. Multi-seed verification (cell `omni_multiseed`) caught the false positive before promotion. Implemented via **Option B** (29-feature paper format). Three cells `omni_download` → `omni_features` → `omni_augment` + verify cell `omni_multiseed`, inserted after `b3aaa3d6`. Checkpoint `solar_bilstm_augmented_model.pth` is an **ablation artifact only** (single-seed-42); primary `solar_bilstm_model.pth` untouched and stays the primary.

**What was built.** 5 OMNI storms (Halloween 2003 −383, Bastille 2000 −300, St Patrick 2015 −234, Sep 2017 −148, Aug 2018 −176) from NASA CDAWeb `OMNI2_H0_MRG1HR` (vars suffixed `*1800`; JSON path `CDF[0].cdfVariables.variable[].cdfVarData.record[].value[0]`). OMNI already hourly → mean cols = the hourly value, std cols = **3h rolling-std proxy** (intra-hour std undefined at 1 obs/hr). `bx_gsm`=`bx_gse`; angles derived `theta=degrees(asin(Bz/Bt))`, `phi=(degrees(atan2(By,Bx))+360)%360`. `smoothed_ssn` hardcoded per storm month (SILSO v2). 1056 OMNI hours → 791 sequences (329 intense); train intense 3321 → 3650 (+10%). Val/test base-only (no leakage); scaler refit on augmented train.

**Multi-seed held-out test t+6 (5 seeds, paired — same seed trains base & aug):**
| Metric | Base | Aug | Delta (mean±std) |
|---|---|---|---|
| Aggregate | 11.418±0.222 | 11.565±0.315 | **−0.147 ± 0.370** |
| Intense (N=387) | 44.46±1.03 | 46.19±3.52 | **−1.73 ± 3.79** |

**Per-seed intense Δ:** seed42 **+4.63** (the only win) · seed43 −4.90 · seed44 −1.31 · seed45 −4.03 · seed46 −3.03.

**Findings.**
1. **Single-seed result was cherry-picked by chance.** Only seed 42 helped; the other four all *hurt* intense (−1.3 to −4.9 nT). Paired mean intense Δ = −1.73 nT (aug worse), std 3.79 >> |mean| → not significant, wrong sign.
2. **Aug destabilizes intense prediction** — intense RMSE std 1.03 → 3.52 (3.4×). Adding 791 OMNI sequences makes the storm tail *more* seed-sensitive, not less.
3. **Decision rule fails on both counts:** mean delta < 0 (intense AND aggregate) and |mean| << std. REJECT.
4. **Why it likely fails:** (a) only 791 sequences / 329 intense — too few to shift a 98k-sequence train set robustly; (b) OMNI proxies (3h rolling-std for the 14 std features, hardcoded monthly ssn) are a distribution mismatch vs MagNet's true sub-hourly std — injecting noise the scaler then spreads across all features; (c) OMNI Dst/solar-wind provenance differs from MagNet (different instruments/processing). Net: noise ≈ signal.

**Decision: REJECT. BiLSTM-base (test t+6 11.26 nT) stays primary.** Do NOT promote the augmented checkpoint; do NOT re-run downstream eval against it. The data-scarcity ceiling is real but **this** augmentation (5 OMNI storms, proxy features) does not clear it. Possible future lever: more storms + true sub-hourly OMNI (1-min `OMNI_HRO_1MIN`) aggregated to real mean+std, not the hourly proxy. Not attempted.

**Process note: always multi-seed before declaring a storm-metric win.** Intense N=387 makes single-seed RMSE high-variance (±1–3.5 nT seed-to-seed); a single run can show ±4 nT swings that are pure noise. This nearly produced a false ADOPTED.

### Step 5 — Documentation pass (final, after steps 1–4)

After each step:
1. Update `results.md` with the new numbers (cell `bb012d7a` writes it but currently only logs the base model — extend it to write ensemble + Conv1DNet + augmented rows).
2. Update CLAUDE.md's `Performance` section.
3. Add a Progress Log entry for the step.
4. Delete this Next Steps block once all four steps are complete (or trim to the unfinished items).

### Decision log (read before starting a step)

- **Why not transformer?** Pre-Phase-A transformer experiment hit 8.27 nT vs BiLSTM 8.45 nT — marginal. Conv1DNet/CNN+BiLSTM are simpler, faster, and won the A4 sweep. Revisit only if Conv1DNet + 128h window + OMNI augmentation still trails TriQXNet at t+1.
- **Why not retrain SolarAttentionLSTM with bigger data?** A4 shows its variance is structurally high (±0.228). Attention is the wrong inductive bias for this dataset size. Promoting Conv1DNet is cheaper than fixing attention.
- **Why not directly target TriQXNet?** Corrected 2026-05-24 after full paper extraction: it is the **same dataset/features/split** as ours (not RTSW-vs-cleaned as previously assumed), and like us it forecasts from solar wind **without prior Dst**. Their 9.27 nT is a t0/t+1 average (t0 = trivial nowcast) and is quiet-dominated — on extreme events their error is t0 20.33 / t+1 20.86 nT. **The t+1 gap is NOT window length** (Step 3: 96/128h closed only ~0.2 nT) and NOT architecture (A4 sweep: all within noise). The remaining gap is the 3-pipeline ensemble + the t0-inflated metric, neither worth chasing. Better use of effort = the capabilities they have and we lack (UQ → Phase C) and the capability we have and they lack (multi-hour horizon + honest storm-conditional eval). Full analysis: `PROGRESS_REPORT.md` (supersedes `18_05_2026.md` notes).
- **Persistence-vs-model hybrid is permanently dead.** Two grid searches (pre- and post-Phase-A) both selected the model only for ~5 % of samples, both lost on intense storms. The timing mismatch is structural. Do not re-attempt. See § "Hybrid Switching Experiment".

---

## ~~Pending~~ SUPERSEDED: OMNI Data Augmentation — DONE 2026-05-24

**Status: IMPLEMENTED & ADOPTED via Option (a).** See `### Phase B Step 4` (Progress Log + Next Steps) for the real cells and results. The live cells are `omni_download` / `omni_features` / `omni_augment` (after `b3aaa3d6`) — they emit the **29-feature paper format** (mean = hourly value, std = 3h rolling-std proxy, derived angles, hardcoded SILSO v2 smoothed_ssn). The template below is the original 15-feature draft, kept only as historical reference — **do not use it**; it predates the per-period split and the 29-feature pipeline.

**Original draft notes (historical):** the cells below build a **15-feature** OMNI DataFrame and won't match the current **29-feature paper set**. Either (a) re-derive OMNI features as `(mean, std)` per hour to match the 29-feature pipeline [← what was done], or (b) switch the notebook to `FEATURE_SET='full'` (37 features inc. the original 15 derived ones) which is a superset.

**Why:** Every algorithmic improvement hit the data-scarcity ceiling. More storm data is the only remaining lever for intense-storm RMSE improvement.

**Where to insert:** Add 3 new cells after cell `b3aaa3d6` (hybrid baseline).

### Data source
**NASA CDAWeb REST API** — no extra packages, just `requests`:
```
https://cdaweb.gsfc.nasa.gov/WS/cdasr/1/dataviews/sp_phys/datasets/OMNI2_H0_MRG1HR/data/{START}T000000Z,{END}T235959Z/BX_GSE,BY_GSE,BZ_GSE,BY_GSM,BZ_GSM,Magnitude,Proton_Density,flow_speed,T,DST1800/?format=json
```
Returns JSON with `CDF[0].Variables` list (each with `Name` and `Values[0]`).

**OMNI variable → our column mapping:**
- `BX_GSE`→`bx_gse`, `BY_GSE`→`by_gse`, `BZ_GSE`→`bz_gse`
- `BY_GSM`→`by_gsm`, `BZ_GSM`→`bz_gsm`, `Magnitude`→`bt`
- `Proton_Density`→`density`, `flow_speed`→`speed`, `T`→`temperature`, `DST1800`→`dst`

**Fill values → NaN:**
`BX/BY/BZ/Magnitude` ≥9999.0 | `Proton_Density` ≥999.0 | `flow_speed` ≥99999.0 | `T` ≥9999999.0 | `DST1800` ≥99999

### Storm periods (Dst min < −100 nT)
| `period` name | start | end | Dst min |
|---|---|---|---|
| `omni_halloween_2003` | `20031025` | `20031105` | −422 nT |
| `omni_bastille_2000`  | `20000712` | `20000718` | −301 nT |
| `omni_stpatrick_2015` | `20150315` | `20150322` | −222 nT |
| `omni_sep2017`        | `20170904` | `20170912` | −142 nT |
| `omni_aug2018`        | `20180822` | `20180829` | −174 nT |

### Cell 1 — Download + parse OMNI

```python
import requests, json

CDAWEB = "https://cdaweb.gsfc.nasa.gov/WS/cdasr/1/dataviews/sp_phys/datasets/OMNI2_H0_MRG1HR/data"
OMNI_VARS = "BX_GSE,BY_GSE,BZ_GSE,BY_GSM,BZ_GSM,Magnitude,Proton_Density,flow_speed,T,DST1800"

STORM_PERIODS = {
    'omni_halloween_2003': ('20031025', '20031105'),
    'omni_bastille_2000':  ('20000712', '20000718'),
    'omni_stpatrick_2015': ('20150315', '20150322'),
    'omni_sep2017':        ('20170904', '20170912'),
    'omni_aug2018':        ('20180822', '20180829'),
}

COL_MAP = {'BX_GSE':'bx_gse','BY_GSE':'by_gse','BZ_GSE':'bz_gse',
           'BY_GSM':'by_gsm','BZ_GSM':'bz_gsm','Magnitude':'bt',
           'Proton_Density':'density','flow_speed':'speed','T':'temperature','DST1800':'dst'}

def fetch_omni(start, end):
    url = f"{CDAWEB}/{start}T000000Z,{end}T235959Z/{OMNI_VARS}/?format=json"
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.json()

def parse_omni_to_df(raw, period_name):
    variables = {v['Name']: np.array(v['Values'][0], dtype=float)
                 for v in raw['CDF'][0]['Variables'] if v['Name'] in COL_MAP}
    n = len(list(variables.values())[0])
    df = pd.DataFrame(index=range(n))
    df['period']    = period_name
    df['timedelta'] = pd.to_timedelta(np.arange(n), unit='h')
    for omni_var, our_col in COL_MAP.items():
        vals = variables[omni_var].copy()
        if our_col in ('bx_gse','by_gse','bz_gse','by_gsm','bz_gsm','bt'):
            vals[vals >= 9999.0] = np.nan
        elif our_col == 'density':       vals[vals >= 999.0]     = np.nan
        elif our_col == 'speed':         vals[vals >= 99999.0]   = np.nan
        elif our_col == 'temperature':   vals[vals >= 9999999.0] = np.nan
        elif our_col == 'dst':           vals[vals >= 99999]     = np.nan
        df[our_col] = vals
    return df

sw_dfs = []
for name, (start, end) in STORM_PERIODS.items():
    print(f'Downloading {name} ...', end=' ', flush=True)
    raw = fetch_omni(start, end)
    df  = parse_omni_to_df(raw, name)
    n_intense = (df['dst'] < -50).sum()
    print(f'{len(df)} h | intense: {n_intense}')
    sw_dfs.append(df)

omni_df = pd.concat(sw_dfs, ignore_index=True)
print(f'\nTotal: {len(omni_df)} h | intense (Dst<-50): {(omni_df["dst"]<-50).sum()}')
```

### Cell 2 — Feature derivation (TODO: rewrite for 29-feature paper set)

Original cell built the 15 derived features (energy, rolling means, dyn_pressure). For the current 29-feature paper set, OMNI data must instead be aggregated as `(mean, std)` per hour for each of the 14 solar-wind variables and joined with smoothed_ssn. Simplest path: switch notebook to `FEATURE_SET='full'` (superset including the 15 derived features), then the original cell below works unchanged.

```python
# Original 15-feature derivation — only valid if FEATURE_SET='full' is used
omni_indexed = omni_df.set_index(['period', 'timedelta'])
grp = omni_indexed.groupby('period')
omni_indexed['energy']       = omni_indexed['speed'] * omni_indexed['bz_gse']
omni_indexed['bz_3h']        = grp['bz_gse'].transform(lambda x: x.rolling(3,  min_periods=1).mean())
omni_indexed['speed_3h']     = grp['speed'].transform(lambda x: x.rolling(3,  min_periods=1).mean())
omni_indexed['bz_6h']        = grp['bz_gse'].transform(lambda x: x.rolling(6,  min_periods=1).mean())
omni_indexed['speed_6h']     = grp['speed'].transform(lambda x: x.rolling(6,  min_periods=1).mean())
omni_indexed['bz_12h']       = grp['bz_gse'].transform(lambda x: x.rolling(12, min_periods=1).mean())
omni_indexed['speed_12h']    = grp['speed'].transform(lambda x: x.rolling(12, min_periods=1).mean())
omni_indexed['dyn_pressure'] = omni_indexed['density'] * omni_indexed['speed']**2
omni_indexed = omni_indexed.interpolate(method='linear', limit_direction='both').ffill().bfill()

X_omni, y_omni = create_sequences(omni_indexed[df_data.columns], SEQ_LEN, FORECAST_HORIZON)
print(f'OMNI sequences: {X_omni.shape}  intense targets: {(y_omni[:,-1] < -50).sum()}')

# Augmented train = base train + all OMNI; val/test unchanged
split_idx     = int(len(X_all) * TRAIN_SPLIT)
X_aug_train   = np.concatenate([X_all[:split_idx], X_omni])
y_aug_train   = np.concatenate([y_all[:split_idx], y_omni])
X_aug_val     = X_all[split_idx:int(len(X_all)*(TRAIN_SPLIT+VAL_SPLIT))]
y_aug_val     = y_all[split_idx:int(len(y_all)*(TRAIN_SPLIT+VAL_SPLIT))]

scaler_aug    = StandardScaler()
N_tr, T, F   = X_aug_train.shape
X_aug_train_sc = scaler_aug.fit_transform(X_aug_train.reshape(-1,F)).reshape(N_tr,T,F)
X_aug_val_sc   = scaler_aug.transform(X_aug_val.reshape(-1,F)).reshape(len(X_aug_val),T,F)

aug_train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_aug_train_sc), torch.FloatTensor(y_aug_train)), batch_size=BATCH_SIZE, shuffle=True)
aug_val_loader   = DataLoader(TensorDataset(torch.FloatTensor(X_aug_val_sc),   torch.FloatTensor(y_aug_val)),   batch_size=BATCH_SIZE, shuffle=False)

aug_model     = SolarAttentionLSTM(input_dim=F, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
                                    dropout=DROPOUT, bidirectional=BIDIRECTIONAL).to(device)
aug_optimizer = torch.optim.Adam(aug_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

best_aug_val, aug_patience, aug_counter = float('inf'), 15, 0
for epoch in range(EPOCHS):
    aug_model.train(); bl = []
    for X_b, y_b in aug_train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        aug_optimizer.zero_grad()
        loss_el = (aug_model(X_b) - y_b)**2
        w = torch.ones_like(y_b); w[y_b < -20] = 5.0; w[y_b < -50] = 15.0
        (torch.mean(loss_el * w)).backward()
        torch.nn.utils.clip_grad_norm_(aug_model.parameters(), max_norm=1.0)
        aug_optimizer.step(); bl.append(torch.mean(loss_el * w).item())
    aug_model.eval(); vl = []
    with torch.no_grad():
        for X_v, y_v in aug_val_loader:
            vl.append(nn.MSELoss()(aug_model(X_v.to(device)), y_v.to(device)).item())
    avg_v = np.mean(vl)
    if avg_v < best_aug_val:
        best_aug_val = avg_v; aug_counter = 0
        torch.save(aug_model.state_dict(), 'solar_bilstm_augmented_model.pth')
    else:
        aug_counter += 1
        if aug_counter >= aug_patience: break
```

### Cell 3 — Evaluate augmented vs base

```python
aug_model.load_state_dict(torch.load('solar_bilstm_augmented_model.pth', map_location=device))
aug_model.eval()
ap, aa = [], []
with torch.no_grad():
    for X_v, y_v in aug_val_loader:
        ap.append(aug_model(X_v.to(device)).cpu().numpy()); aa.append(y_v.numpy())
aug_preds   = np.vstack(ap); aug_actuals = np.vstack(aa)

print('--- OMNI Augmentation: Base vs Augmented BiLSTM ---')
rows = []
for s in range(FORECAST_HORIZON):
    rb = np.sqrt(mean_squared_error(all_actuals[:,s], all_preds[:,s]))
    ra = np.sqrt(mean_squared_error(aug_actuals[:,s], aug_preds[:,s]))
    rows.append({'Step':f't+{s+1}','Base RMSE':round(rb,4),'Aug RMSE':round(ra,4),'Delta':round(rb-ra,4)})
print(pd.DataFrame(rows).to_string(index=False))

dst_t6 = aug_actuals[:,-1]
bins = {'Quiet':dst_t6>=-20,'Moderate':(dst_t6<-20)&(dst_t6>=-50),'Intense':dst_t6<-50}
for lbl, mask in bins.items():
    rb = np.sqrt(mean_squared_error(all_actuals[mask,-1], all_preds[mask,-1]))
    ra = np.sqrt(mean_squared_error(aug_actuals[mask,-1], aug_preds[mask,-1]))
    print(f'{lbl:<10} N={mask.sum():>5}  Base={rb:.4f}  Aug={ra:.4f}  Δ={rb-ra:+.4f}')
```

### Critical notes
- `X_all` and `y_all` are the full base sequence arrays from cell `0d486255`.
- Val and test sets stay base-data-only — OMNI goes training-only.
- After running, update this section with results and delete the OMNI block.
