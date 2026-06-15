# CLAUDE.md

Guidance for Claude Code this repo. Instructions OVERRIDE defaults.

## Project

Deep learning: predict **Dst geomagnetic index 6h ahead** from solar wind data. Early warning for storms. All code in `SolarWindLSTM.ipynb`.

## Run

```bash
jupyter notebook SolarWindLSTM.ipynb          # cell-by-cell
jupyter nbconvert --to notebook --execute SolarWindLSTM.ipynb   # full pipeline
```
Env = conda `dnn` (torch 2.9.1, `crepes` 0.9.0, lightgbm). Best weights write `solar_bilstm_model.pth` on val-loss improve.

## Data (`archive/`)

- `solar_wind.csv` (Bx/By/Bz/Bt/speed/density/temp…), `labels.csv` (Dst), `sunspots.csv` (monthly smoothed ssn). All `(period, timedelta)` multi-index; periods = named event windows.
- Cell `bb365e9d` (A1) auto-downloads full MagNet dataset from Zenodo `12694950` if <6 periods. Safe re-run.

## Pipeline (`load_and_preprocess` → `create_sequences`)

1. Load CSVs, `(period, timedelta)` index.
2. Hourly **mean+std** for 14 solar-wind vars (paper feature set).
3. + smoothed_ssn → **29 features** (`feature_set='paper'`). `'full'` adds 8 derived → 37.
4. Sliding window: 48h input → 6h target vector.
5. **Per-period 70/20/10 chronological** split; `StandardScaler` fit train only. Held-out test never seen training/early-stop.

## Model `SolarAttentionLSTM`

2-layer BiLSTM (hidden=128/dir, dropout=0.3, bidirectional) + attention over 48 timesteps → concat(context, last hidden) → linear → 6 Dst preds.

## Training

Two-tier asymmetric weighted MSE: Dst<−20 → 5×, Dst<−50 → 15×. Adam lr=1e-3 wd=1e-5, grad clip 1.0, early stop patience=15 on val loss.

## Hyperparameters

```python
SEQ_LEN=48; FORECAST_HORIZON=6; HIDDEN_DIM=128; NUM_LAYERS=2
DROPOUT=0.3; BIDIRECTIONAL=True; BATCH_SIZE=64; LEARNING_RATE=0.001
TRAIN_SPLIT=0.7; VAL_SPLIT=0.2; TEST_SPLIT=0.1; EPOCHS=50; FEATURE_SET='paper'
```

## Current models & performance

**Three adopted checkpoints (deploy by regime + horizon):**
- `solar_bilstm_model.pth` — BiLSTM base. **Primary fallback.** Held-out test t+6: RMSE 11.26 nT, r 0.8247, R² 0.6780; intense (N=387) 44.64 nT.
- `solar_bilstm_omni1m_model.pth` — base + 1-min OMNI augmentation (Phase D). **Primary t+6 storm warning.** Held-out test t+6: aggregate 10.94, **intense 36.10 nT (−19% vs base)**. 5-seed paired Δ: aggregate +0.48±0.28, intense +8.36±3.39 (all 5 seeds positive). Crossover ~t+4; loses t+1–t+3/quiet (t+6 quiet 7.94→8.51).
- `solar_bilstm_idx_model.pth` — base + **Kp/ap/AE geomag-index INPUTS** (Phase J, next-step #4). **Primary t+1–t+3 storm warning.** 32 feat (paper-29 + lagged Kp/ap/AE). Checkpoint test (seed42): agg 9.83, intense 36.70, **t+1 intense 28.01 nT (−21% vs base 35.49)**, t+6 intense 45.49 (no t+6 gain). 5-seed paired Δ vs base (rebuilt, same windows): t+1 intense **+4.84±1.53 (all 5 +, t-p .002)**, all-horiz intense +2.80±1.17 (t-p .006), agg +0.55±0.14. Gain **front-loaded** (decays to ns by t+6) — complements OMNI-aug (owns t+6). Scaler `idx_scaler.pkl`. ⚠️ Deploy needs **near-real-time** AE/Kp/ap (quicklook): gain halves by ~3h index staleness, gone by ~6h.

**UQ:** Mondrian conformal (by predicted severity) = shipped storm-adaptive intervals. Aug-model recalibrated (Phase C aug): intense @95% t+6 coverage 90.4% (base marginal 64.3%). Use aug-model UQ storm deployment. Marginal width grows t+1→t+6 (40.0→44.5 nT @95%) = honest error-growth-with-lead, which TriQXNet's t+1-only UQ can't show.

**Storm-conditional t+6 (base, held-out test):** Quiet (≥−20, N=11900) 7.94 | Moderate (−50..−20, N=1687) 12.16 | Intense (<−50, N=387) 44.64.

**Persistence:** beats model t+1–t+5 aggregate; model wins t+6, mainly intense (+17% skill). Base deploy: persistence t+1–t+5, OMNI-aug t+6. Switchover by horizon only — no threshold (hybrid dead, see below). **Storm-warning refinement (Phase J):** for intense storms, `idx`-model beats persistence AND base at t+1–t+3 (t+1 intense 28.0 vs base 35.5) when near-real-time indices available → idx t+1–t+3, OMNI-aug t+6.

**vs TriQXNet (arXiv:2407.06658v3):** same dataset/features/split. Our t+1 10.14 vs their 9.27 (their number is t0/t+1 avg, t0=trivial nowcast, quiet-dominated; on extremes they hit t0 20.33/t+1 20.86). Gap NOT window (Step 3) nor arch (A4) — their 3-pipeline ensemble + t0-inflated metric, not worth chasing. We add multi-horizon + honest storm-conditional eval + per-horizon UQ they lack.

## Significance (Phase E — leakage-safe blocked 10-fold paired)

- intense `aug−base` +3.01 nT (Wilcoxon p=0.049 **sig**, paired-t p=0.054). Phase D holds under stricter protocol.
- intense `base−persist` +5.12 nT (p=0.007 **sig**) — model >> persistence on storms.
- agg `aug−base` ns (p=0.105); agg `base−persist` −0.98 (p=0.016, persistence wins aggregate) — both expected, quiet-dominated.
- CV intense N/fold 231–727; CV RMSE runs higher than fixed held-out (storm-rich blocks, less train/fold) — not directly comparable by design.

## STATUS: Phases A–K COMPLETE. Phase J (Kp/ap/AE inputs) ADOPTED — first new-modality win. Phase K (IG interpretability) DONE — all 5 next-steps closed.

Phase J (next-step #4) = first external-data lever to beat base on held-out test (leakage-verified, multi-seed). Remaining open levers: SHAP (#5), SYM-H target, solar imagery. Everything else ruled out (see DO NOT REPEAT).

### Phase J record — date-recovery + geomag indices (2026-06-13)
MagNet base periods are **anonymized** (relative timedelta). Recovered absolute dates by Dst-fingerprint cross-correlation vs OMNI hourly Dst (`_date_recovery_full.py` → `_magnet_dates.json`): train_a 1998-02-16, train_c 2004-05-01 (RMSE ~0 exact lock), train_b 2013-06-01 (RMSE 5.6, unambiguous 2nd-best gap). Enabled merging OMNI Kp/ap/AE into base periods incl held-out test. **Leakage guard:** indices live in input window only (`create_sequences`) AND cadence-lagged to last completed interval before t (AE 1h, Kp/ap 3h). **Leak-robustness verified:** lag-decay sweep (`geomag_index_lag_decay.csv`) shows t+1 intense gain decays SMOOTHLY with extra index staleness (+0h +7.5 → +1h +5.9 → +3h +4.1 → +6h ~0), no cliff → real magnetosphere-autocorrelation signal, not boundary label-leak.

### Pending-cell record — CLOSED 2026-06-11 (option 1: embed, no rerun)
4 pending cells (`cv_significance`, `phase_c_uq`, `phase_c_uq_aug`, `phase_f_normalized`) had empty in-notebook outputs (computed standalone, env `dnn`). Result tables now embedded as markdown cells right after each (ids `*_results`); CSVs remain source of truth. Code cells still carry `EXECUTION PENDING` header + no kernel rerun (cv_significance = ~4.3h GPU, reproduces same CSV). Backup `SolarWindLSTM.ipynb.bak`.

## Options to proceed (model done; pick by goal)

**Bucket 1 — housekeeping: DONE** (results embedded above). Optional leftover: reword `EXECUTION PENDING` headers → "results embedded below"; or full kernel rerun to populate real outputs (incl 4.3h CV).

**Bucket 2 — new-data levers (curated-storm sub-option now CLOSED, see DO NOT REPEAT #10; remaining options below):**
- New input modality (not a transform of existing inputs — those rejected #9): solar imagery (SDO/AIA), CME catalog, L1 real-time feed.
- Higher-cadence target: SYM-H instead of hourly Dst → more storm-resolution labels.

**Bucket 3 — productionize / publish (work is paper-shaped):**
- Real-time inference: persistence t+1–t+5, base+aug t+6, Mondrian UQ bands.
- Writeup — novel vs TriQXNet = multi-horizon + honest storm-conditional + per-horizon UQ.

Recommendation: storm-RMSE lever exhausted (15-storm Phase D stays adopted) → ship/finish via bucket 3, or bucket 2 new-modality/SYM-H if pursuing accuracy further.

## Next steps (from deep-research-report cross-check — recommended order)

Accuracy lever exhausted; these add value via operational framing + honest eval + new modality (the actual differentiation vs TriQXNet). Ordered cheapest→biggest. Items below are NEW (survived DO NOT REPEAT check); report's transformer/ensemble/loss-reweight proposals all collide with rejected list — skip them.

1. **Storm-detection eval metrics** — ✅ DONE (Phase G, cell `storm_detect_metrics` + `_results`, held-out TEST, CSV `storm_detection_metrics.csv`). POD/FAR/CSI/HSS/BIAS + per-horizon + ROC at Dst<−50/−80/−100. Result: model **under-forecasts intensity** on extremes (BIAS 0.67@−80, 0.54@−100 → low FAR 0.16/0.04 but lower POD than persistence 0.57/0.52 vs 0.72/0.69); at −50 persistence edges CSI (0.53 vs 0.52). Per-horizon honest skill decay POD 0.695→0.669 t+1→t+6. Confirms conservative-alarm regime (data-scarcity, #8) and motivates #2 (tunable prob threshold). NB: notebook Phase-1 inference cell uses val_loader (early-stop set) — Phase G re-runs checkpoint on test_loader for clean hold-out.
2. **Multi-task storm-classifier head** — ✅ DONE (Phase H, cells `mtl_model_def`/`mtl_train`/`mtl_eval` + `_results`, checkpoint `solar_bilstm_mtl_model.pth`, adopted base untouched). Shared encoder + reg head + clf head (P(Dst<−50) per horizon), joint loss = base weighted MSE + 200·BCE. Held-out test: **AUC 0.983, Brier 0.012, ECE 0.003** (well-calibrated — capability Phase G lacked). Reg cost small (agg t+6 11.26→11.58; intense 44.64→46.14 within seed noise; moderate improved 12.16→11.56). **Win:** tunable prob threshold — at τ=0.30 POD 0.713/CSI 0.573/HSS 0.721 beats persistence (0.695/0.533) AND deterministic base (0.669/0.519). Resolves Phase G's fixed-conservative-bias finding. Adoptable as storm-alert layer. (Single seed OK: headline is calibration/AUC; intense RMSE got worse not better, so no false-win risk.) Extension: add −80/−100 heads.
3. **Robustness / degradation tests** — ✅ DONE (Phase I, cell `robustness_tests` + `_results`, CSV `robustness_metrics.csv`). No retrain; base checkpoint on perturbed test inputs. **Meets <1.5× target at 10% dropout** (agg 1.02–1.10×, intense 1.05–1.11×). Noise asymmetry: σ=2 noise → agg 1.89× but intense ~1.0× (storm signal robust, quiet fragile). **Magnetometer outage critical: intense 2.05× (44.6→91.5 nT)** — Bz drives Dst; plasma outage mild (1.03×) → prioritize mag redundancy. (Single-satellite mapped to instrument-block outage since MagNet data is pre-merged, no per-satellite source.) Novel vs literature.
4. **Geomagnetic-index INPUTS (Kp/Ap/AE)** — ✅ DONE & ADOPTED (Phase J, checkpoint `solar_bilstm_idx_model.pth`, cells `phase_j_build`/`phase_j_results`, CSVs `geomag_index_ab.csv`/`geomag_index_lag_decay.csv`). New modality, NOT a transform → escapes #9. Required date-recovery (anonymized periods, see Phase J record). Leakage handled: input-window-only + cadence-lag + verified by smooth lag-decay (no cliff). **5-seed win, all positive:** t+1 intense +4.84±1.53 (−21%, t-p .002), all-horiz intense +2.80±1.17, agg +0.55±0.14. Front-loaded — owns t+1–t+3 storm warning, complements OMNI-aug (t+6). Deploy caveat: needs near-real-time indices. Extension: SYM-H/ASY indices, or +6h-lagged-only variant for latency-robust deploy.
5. **Interpretability (SHAP / integrated gradients)** — ✅ DONE (Phase K, cells `phase_k_shap`/`phase_k_results`, script `_shap_ig.py`, CSVs `shap_feature_importance.csv`/`shap_time_profile.csv`, plot `shap_ig.png`). Hand-rolled IG (shap/captum absent), base model, held-out TEST (13974 seqs, 529 storm), baseline=scaled-zero, M=32 steps. **Top features:** bz_gsm_mean (0.161), speed_mean (0.161), theta_gsm_mean (0.154), bt_mean (0.109). **On storm rows:** bt_mean (0.545) + bz_gsm_mean (0.511) + speed_mean (0.306) dominate — southward-Bz/field-magnitude/speed = exactly the Dst-driving physics. **Recency:** mean|IG| t=−1 (0.310) / t=−48 (0.025) = 12.4×, last 6h carry bulk (echoes Phase J autocorrelation). **Validates #9:** model already concentrates on Bz/Bt/speed that the rejected coupling transforms (vBs/clock_angle/epsilon/Newell) recombine → they were redundant, not under-used. Interpretability gap closed, paper-friendly.

Maybe (bigger lift): SYM-H target (bucket 2, higher-cadence labels); physics baseline (Burton formula) comparison to contextualize ML gain.

## DO NOT REPEAT (tested & rejected)

1. **More OMNI storms (27 vs 15)** — regressed (intense Δ +5.57 vs adopted +8.36), noisier. Curation > count; additions span solar-min/weaker storms. Don't blind-expand count; if revisited curate extreme Dst + matched solar-cycle phase. (Phase D-2)
2. **Longer input window (96/128h)** — 48h primary (11.26) beats all; batch-size delta > window gain; worse on intense/moderate. Gap not window-driven (best t+1 closes only 0.2 nT). (Step 3)
3. **OMNI aug with hourly proxy (3h rolling-std, 5 storms)** — single-seed-42 mirage; 5-seed killed it (intense Δ −1.73±3.79). Proxy std + hardcoded ssn = noise. Real 1-min mean+std (Phase D) was fix. (Step 4)
4. **Conv1DNet / other archs as primary** — A4 ranking VAL-only, did NOT transfer to test. Conv1D lost every test step both losses. Confirm any arch on held-out test. (Step 2)
5. **Ensemble stack (LGBM blend / GRU correction)** as blanket layer — doesn't beat BiLSTM-base on held-out test; trades quiet gain for worse storms (intense 44→49). Sub-7 nT was val-subset (quiet-tail) artifact. GRU only useful as quiet-time bias corrector in regime-split deploy. (Step 1)
6. **Normalized conformal (volatility sigma)** — partial (intense 64.3→72.9%) but Mondrian wins (82.9%) and worse on quiet. Keep Mondrian. If revisited: Mondrian bins of normalized residuals, don't substitute. (Phase F)
7. **Persistence-vs-model hybrid switching** — permanently dead. Two grid searches both picked model ~5% of samples, both lost on intense. Onset timing mismatch structural. Switch by horizon only.
8. **Loss reweighting / oversampling for storm bias** — all failed (linear recalib, directional asymmetric loss, 4×/15× oversampling). Storm perf data-scarcity-bound, not loss-bound.
9. **Physics-coupling features** (bz_south, vBs, clock_angle, epsilon, newell) — −0.16 nT worse; redundant nonlinear transforms BiLSTM already learns. Don't add.
10. **Curated extreme storms (18 vs 15, all Dst<−200, ssn-matched)** — added Sep1998/Oct1999/Apr2023 to adopted 15-list, 5-seed paired vs base: intense Δ +2.93±3.39 (1/5 seed negative), aggregate Δ +0.081±0.292 (2/5 negative) — both worse than adopted 15-storm (intense +8.36±3.39 all-positive, agg +0.48±0.28). Even strict Dst<−200 + ssn-matched curation regresses vs 15-list. Confirms #1: 15-storm set is a local optimum, don't expand. Closes bucket-2 curated-storm sub-option.

**Process rule: ALWAYS multi-seed before declaring storm-metric win.** Intense N≈387 → ±1–3.5 nT seed noise, enough to fake ±4 nT improvement on one seed (nearly caused false ADOPT).

## Key cells

Pipeline: `0d486255` (split/exec) · `bc9878f0` (model def) · `8a4080ce` (train) · `af6ee96c` (inference) · `446d08ce`/`3b7398e3` (per-step + storm tables) · `37d6b8b0` (A3 TriQXNet) · `bb012d7a` (writes `results.md`).
Adopted aug: `omni1m_build` (15-storm list, true hourly mean+std from `OMNI_HRO_1MIN`, label DST1800) → `omni1m_multiseed`.
UQ/sig: `phase_c_uq`, `phase_c_uq_aug`, `cv_significance`, `phase_f_normalized`.
Storm-detection (Phase G, next-step #1): `storm_detect_metrics` (+ `_results` md) → `storm_detection_metrics.csv`/`.png`. Self-contained, re-runs checkpoint on test_loader.
Multi-task classifier (Phase H, next-step #2): `mtl_model_def`/`mtl_train`/`mtl_eval` (+ `_results` md) → `solar_bilstm_mtl_model.pth`, `mtl_regression_ab.csv`, `mtl_prob_threshold_sweep.csv`, `mtl_storm_classifier.png`. Code cells carry EXECUTION PENDING (computed standalone env dnn, ~train+eval).
Robustness (Phase I, next-step #3): `robustness_tests` (+ `_results` md) → `robustness_metrics.csv`/`.png`. No retrain, base checkpoint on perturbed test inputs.
Geomag-index inputs (Phase J, next-step #4): `phase_j_build` (date-recovery + Kp/ap/AE merge, EXECUTION PENDING) + `phase_j_results` md → `solar_bilstm_idx_model.pth`/`idx_scaler.pkl`, `geomag_index_ab.csv`/`geomag_index_lag_decay.csv`. Standalone scripts: `_date_recovery_full.py`, `_fetch_indices.py`, `_build_features_indices.py`, `_ab_train_indices.py`, `_lag_sweep.py`, `_finalize_phase_j.py`. Caches `_magnet_dates.json`, `_omni_dst_1995_2020.pkl`, `_omni_indices_1998_2019.pkl`, `_seq_base.npz`/`_seq_aug.npz`.
Interpretability (Phase K, next-step #5): `phase_k_shap` (IG code, EXECUTION PENDING — computed standalone via `_shap_ig.py`) + `phase_k_results` md → `shap_feature_importance.csv`, `shap_time_profile.csv`, `shap_ig.png`. Hand-rolled IG, cuDNN disabled for eval-mode RNN backward, base checkpoint on held-out test.
Ablation artifacts only (rejected, don't promote): `conv1d_step2`, `conv1d_plainmse`, `seq_ablation_step3`, `omni_download`/`omni_features`/`omni_augment`/`omni_multiseed`.

Full historical tables: `results.md`, `PROGRESS_REPORT.md`.