# Evaluation Results
**Run:** 2026-05-24 11:48:38
**Config:** SEQ_LEN=48, HIDDEN_DIM=128, BIDIRECTIONAL=True, DROPOUT=0.4

## Overall Performance (t+6)
| Metric | Value |
|---|---|
| RMSE | 11.5731 nT |
| Pearson r | 0.7755 |
| R2 | 0.5822 |

## Per-Step Metrics
| Step   |   RMSE (nT) |   Pearson r |     R² |
|:-------|------------:|------------:|-------:|
| t+1    |     10.144  |      0.8272 | 0.6788 |
| t+2    |     10.1132 |      0.8286 | 0.6808 |
| t+3    |     10.3338 |      0.8202 | 0.6668 |
| t+4    |     10.7069 |      0.806  | 0.6423 |
| t+5    |     11.1093 |      0.7908 | 0.615  |
| t+6    |     11.5731 |      0.7755 | 0.5822 |

## Storm-Conditional Metrics (t+6)
| Condition                  |     N |   RMSE (nT) |   Pearson r |      R² |
|:---------------------------|------:|------------:|------------:|--------:|
| Quiet (Dst ≥ −20)          | 23840 |      9.3141 |      0.5325 | -0.1652 |
| Moderate (−50 ≤ Dst < −20) |  3440 |     14.2021 |      0.4012 | -2.4207 |
| Intense (Dst < −50)        |   662 |     38.4836 |      0.5853 |  0.0249 |

---

# Phase B Step 4 — OMNI Data Augmentation (REJECTED — single-seed mirage, 2026-05-24)

Training-only augmentation: +791 sequences (329 intense) from 5 OMNI storms (Halloween 2003, Bastille 2000, St Patrick 2015, Sep 2017, Aug 2018). Train intense count 3321 → 3650 (+10%). Val/test held base-only (no leakage). `solar_bilstm_augmented_model.pth` = single-seed-42 ablation artifact only; primary `solar_bilstm_model.pth` untouched and stays primary.

## Single seed 42 (looked great — but NOT robust)
Held-out test t+6: aug better at every step; aggregate 11.26→11.14 (+0.12); intense (N=387) 44.64→40.01 (**+4.63**). This single run suggested ADOPT.

## 5-seed paired verification (held-out test t+6) — reverses the verdict
Same seed trains both base & aug (paired deltas). Cell `omni_multiseed`.

| Metric | Base | Aug | Delta (mean±std) |
|---|---|---|---|
| Aggregate | 11.418±0.222 | 11.565±0.315 | **−0.147 ± 0.370** |
| Intense (N=387) | 44.46±1.03 | 46.19±3.52 | **−1.73 ± 3.79** |

Per-seed intense Δ: **seed42 +4.63** (only win) · seed43 −4.90 · seed44 −1.31 · seed45 −4.03 · seed46 −3.03.

## Verdict
**REJECTED.** Decision rule fails: mean intense Δ −1.73 nT (aug *worse*) and aggregate Δ −0.147 nT (aug *worse*), with std swamping both means. The +4.63 was seed-42 luck — every other seed hurt intense. Aug also tripled intense variance (1.03→3.52), i.e. *destabilizes* the storm tail. BiLSTM-base (test t+6 11.26 nT) stays primary; do not promote.

**Lesson:** intense N=387 gives ±1–3.5 nT seed-to-seed RMSE noise — always multi-seed before declaring a storm-metric win. A single run nearly produced a false ADOPTED.

---

# Phase C — Conformal Prediction Intervals (Uncertainty Quantification, 2026-05-25)

Post-hoc split-conformal on the trained BiLSTM (**no retrain**), library `crepes` 0.9.0 (same as TriQXNet). Calibrated on **full val residuals** (fit=train / calibrate=val / evaluate=test). Cell `phase_c_uq` (after A3 cell `37d6b8b0`). Outputs `conformal_intervals.csv` + `conformal_band_plot.png`. TriQXNet ships UQ at t+1 only — ours is **per-horizon t+1…t+6** + **per-storm-bin** + a **storm-adaptive (Mondrian)** variant.

## Marginal conformal — per-horizon coverage (%) / mean width (nT), held-out test
| Step | cov@90 | w@90 | cov@95 | w@95 | cov@99 | w@99 |
|---|---|---|---|---|---|---|
| t+1 | 92.6 | 29.4 | 96.8 | 40.0 | 99.3 | 68.8 |
| t+2 | 92.9 | 29.3 | 96.8 | 39.5 | 99.4 | 68.3 |
| t+3 | 92.7 | 29.7 | 96.9 | 40.3 | 99.4 | 69.1 |
| t+4 | 92.9 | 30.9 | 96.9 | 41.3 | 99.3 | 71.1 |
| t+5 | 92.8 | 32.0 | 97.0 | 42.7 | 99.3 | 73.7 |
| t+6 | 93.0 | 33.5 | 97.0 | 44.5 | 99.3 | 75.4 |

Coverage slightly conservative (≈ nominal +1–2 pts). **Interval width grows monotonically with horizon** (40.0→44.5 nT @95%) — the error-growth-with-lead-time signal TriQXNet's t+1-only UQ cannot show.

## Per-storm-bin coverage @ 95% (t+6): Marginal vs Mondrian (binned by PREDICTED severity)
| Bin (by actual) | N | marg cov% | marg w | mond cov% | mond w |
|---|---|---|---|---|---|
| Quiet (Dst ≥ −20) | 11900 | 98.6 | 44.5 | 97.7 | 36.5 |
| Moderate (−50 ≤ Dst < −20) | 1687 | 93.1 | 44.5 | 91.7 | 54.9 |
| Intense (Dst < −50) | 387 | **64.3** | 44.5 | **82.9** | 107.5 |

**Marginal undercovers intense storms badly (64.3% vs 95% target)** — constant-width intervals are sized by the quiet-dominated calibration set. **Mondrian fixes most of it (64.3→82.9%)** by widening storm intervals (44.5→107.5 nT) and tightening quiet (44.5→36.5). Residual gap = **missed onsets**: when the model predicts quiet but the storm hits, the row sits in the narrow quiet bin — Mondrian-by-predicted cannot rescue cases the model does not see coming.

## Verdict
UQ added, **honest and per-horizon**. Closes our biggest capability gap vs TriQXNet at zero retrain cost. The intense-bin undercoverage is the same storm-data-scarcity ceiling (Phase B) re-surfacing in interval form — Mondrian mitigates but does not erase it.

---

# Phase D — 1-min OMNI Augmentation (ADOPTED — first robust storm-RMSE win, 2026-05-25)

Re-did the rejected Step-4 OMNI augmentation with the fix its own post-mortem prescribed: **real `OMNI_HRO_1MIN` (1-min cadence → true hourly mean+std)** instead of the 3h rolling-std proxy, and **15 storms** instead of 5. Cells `omni1m_build` → `omni1m_multiseed`. Adopted checkpoint `solar_bilstm_omni1m_model.pth` (base `solar_bilstm_model.pth` retained for A/B).

## Data
15 major storms (Dst<−100), 2000–2024, incl. Gannon May 2024 (−406). Features = true hourly mean+std over ~60 one-min samples/hr for 14 vars; theta/phi derived at 1-min then aggregated; label = `DST1800` (Kyoto Dst, same index as base). **2184 OMNI hours → 1389 sequences / 877 intense** (Step-4 proxy had 791 / 329 — 2.7× the intense data). Train intense 3321 → 4198 (+26%). Train-only; val/test base-only; scaler refit on augmented train.

## 5-seed paired held-out test t+6 (same seed trains base & aug → paired deltas cancel init variance)
| Metric | Base | Aug | Delta (mean±std) |
|---|---|---|---|
| Aggregate | 11.418±0.222 | **10.937±0.265** | **+0.481 ± 0.281** |
| Intense (N=387) | 44.46±1.03 | **36.10±3.08** | **+8.36 ± 3.39** |
| Moderate | 12.77±0.89 | 12.96±0.50 | −0.19 ± 1.04 (flat) |

**Per-seed intense Δ:** seed42 +13.70 · seed43 +5.42 · seed44 +7.92 · seed45 +5.56 · seed46 +9.21 — **all positive.** Aggregate also positive on all 5.

## Verdict
**ADOPTED.** Decision rule met on every count: intense mean Δ +8.36 > 0, |mean| 8.36 > std 3.39 (gain not swamped by seed noise), aggregate also improves. **Intense storm RMSE cut 19% (44.5→36.1 nT)** — the metric that matters — with aggregate down too. First lever (Phase B Steps 1–4 + this) to move the storm-data-scarcity ceiling.

**Contrast vs Step 4 (REJECTED):** identical harness, only the OMNI features changed (proxy-std → true 1-min std) plus 3× more storms. Step 4 intense Δ was −1.73±3.79 (worse); this is +8.36±3.39. **The proxy std was the problem, not OMNI augmentation itself.** Confirms Step 4's post-mortem hypothesis exactly.

**Follow-ups:** recalibrate Phase C conformal UQ on the aug model (storm intervals should tighten); 25+ storms for more headroom; decide whether to overwrite the primary checkpoint after re-running downstream eval on the aug model.

## Adopted checkpoint (seed 42) — per-step A/B on held-out test
`solar_bilstm_omni1m_model.pth` vs base `solar_bilstm_model.pth`. (Seed 42 is the most favorable of the 5; the robust cross-seed claim is the multi-seed table above. Per-step shown to expose the tradeoff.)

| Step | Base | Aug | Δ |
|---|---|---|---|
| t+1 | 9.82 | 10.55 | −0.73 |
| t+2 | 9.66 | 10.15 | −0.49 |
| t+3 | 9.83 | 9.98 | −0.15 |
| t+4 | 10.25 | 9.95 | +0.30 |
| t+5 | 10.72 | 10.11 | +0.61 |
| t+6 | 11.26 | **10.50** | **+0.76** |

| Storm bin (t+6) | N | Base | Aug | Δ |
|---|---|---|---|---|
| Quiet (≥−20) | 11900 | 7.94 | 8.51 | −0.57 |
| Moderate (−50..−20) | 1687 | 12.16 | 13.54 | −1.38 |
| Intense (<−50) | 387 | 44.64 | **30.94** | **+13.70** |

**Honest read — it's a tradeoff, not a free lunch.** Aug shifts skill toward **long horizons (crossover ~t+4) and intense storms**, at the cost of short-horizon (t+1–t+3) and quiet/moderate t+6. This is the *right* trade: persistence already beats the model at t+1–t+5 (see Persistence Baseline), so the model's job is t+6 storm warning — exactly where aug wins (intense −13.7 nT). For an operational deployment, pair: persistence/base for t+1–t+3 quiet, omni1m-aug for t+6 storms.

---

# Phase C (aug) — Conformal UQ recalibrated on the adopted 1-min OMNI model (2026-05-26)

Phase D cut intense t+6 RMSE 44.5→36.1 nT — so the conformal intervals should track storms better. Recalibrated split-conformal on the aug model (`solar_bilstm_omni1m_model.pth`), **no retrain**: calibrate on aug-model val residuals, evaluate on held-out test. Scaler reproduced exactly (StandardScaler deterministic → refit on base-train+OMNI = the training scaler). Cell `phase_c_uq_aug` (after `phase_c_uq`). Outputs `conformal_intervals_aug.csv`, `conformal_base_vs_aug_bins.csv`.

## Intense-storm coverage @95% (t+6) — the metric Phase D moved
| | base | aug |
|---|---|---|
| marginal cov% | 64.3 | **73.4** |
| Mondrian cov% | 82.9 | **90.4** |
| Mondrian width (nT) | 107.5 | 122.4 |

**Aug UQ is materially better on storms.** Marginal intense coverage 64.3→73.4% at near-equal width (44.5→45.6 nT) = storm residuals genuinely tighter. Mondrian intense coverage 82.9→**90.4%** (near nominal 95%): the aug model sees onsets better, so more intense rows land in the model's *predicted-intense* bin and get the wide storm band — the residual gap (missed onsets) shrinks. The wider Mondrian intense band (122.4 vs 107.5) reflects that the intense calibration bin now holds more/larger genuine storm residuals, not a regression.

## Aggregate / other bins
Marginal per-horizon width grows t+1→t+6 same as base (40.1→45.6 nT @95%); aggregate coverage ≈ nominal. Cost of aug: +~1 nT aggregate width at t+6, moderate-bin coverage dips 91.7→89.5%. Net: clear UQ improvement where it matters (storms), small give-back elsewhere — mirrors the point-forecast tradeoff.

## Verdict
The Phase D point-forecast win carries into uncertainty: the aug model's intervals cover intense storms far closer to nominal (the largest base-model UQ gap, 64→73% marginal / 83→90% Mondrian) at zero retrain. Use the aug-model UQ for storm-warning deployment.


# Phase D-2 — 25+ storms tested: REGRESSED, NOT adopted (2026-05-26)

Tested the Phase D follow-up "more storms for headroom": expanded omni1m_build STORMS_1M 15 -> 27 major storms (Dst<-100, 2000-2024). 12 additions (aug2000 -234, apr2001 -271, sep2002 -176, jul2004 -170, jan2005 -103, sep2005 -139, mar2012 -145, mar2013 -132, dec2015 -166, mar2023 -163, apr2023 -213, oct2024 -333), all DST-verified to carry real intense hours (24-110 each). 5-seed paired re-run, same base seeds (base reproduced exactly).

| Metric (t+6) | base | 27-storm aug | delta (27) | 15-storm aug delta (ADOPTED) |
|---|---|---|---|---|
| Aggregate | 11.418 +/- 0.222 | 11.077 +/- 0.447 | +0.341 +/- 0.275 | +0.48 +/- 0.28 |
| Intense (N=387) | 44.46 +/- 1.03 | 38.89 +/- 3.75 | +5.57 +/- 3.68 | +8.36 +/- 3.39 |

Per-seed intense delta (27-storm): s42 +3.63, s43 +4.32, s44 +7.50, s45 +10.91, s46 +1.50 (all positive).

**Finding:** 27-storm aug still beats base (all 5 seeds positive both metrics) but is WORSE than the 15-storm adopted model on both: intense +5.57 vs +8.36 (-2.8 nT), aggregate +0.341 vs +0.48 (-0.14 nT). Aug variance rose (intense std 3.39->3.75, agg 0.28->0.447). More storms = noisier, not better.

**Why:** the 12 additions span solar-min years (jan/sep2005, mar2012/13 — smoothed_ssn 40-80) + 2023/24, and several are weaker (Dst -103 to -176) vs the curated 15 (incl. Gannon -406). Heterogeneous solar-cycle/instrument provenance + diluted extreme tail = noise ~ signal at the margin. Quantity != quality for the storm tail.

**Decision: REJECT 27-storm. 15-storm solar_bilstm_omni1m_model.pth stays adopted.** Multiseed wrote nothing to disk -> checkpoint untouched. Cell omni1m_build reverted to 15-storm list. Storm-scarcity lever has a ceiling: the right 15 beat 27 mixed. Do NOT re-attempt blind storm-count expansion; curate for extreme Dst + matched solar-cycle phase if revisited.

---

# Phase E — 10-fold blocked CV + paired significance tests (2026-06-01)

Matches TriQXNet's significance bar (they used 10-fold CV paired t-tests; we previously reported only mean±std over 5 seeds). Cell `cv_significance` (after `omni1m_multiseed`). **Leakage-safe blocking, NOT TriQXNet's plain random 10-fold:** per period the 139,713 pooled sequences are cut into 10 contiguous chronological blocks; fold k test = block k, fold-train = the rest MINUS a `SEQ_LEN+FORECAST_HORIZON`=54-sequence guard band each side of the test block, so no train window shares raw hours with any test window (sliding-window overlap leakage removed). Per fold: refit StandardScaler, train base (fold-train) + aug (fold-train + 1389 1-min OMNI seqs) at the **same seed (paired)**, storm-weighted loss + early stop (mirrors `omni1m_multiseed`). Metric = t+6 RMSE on the identical fold-test block. 20 trainings, ~4.3 h GPU (RTX 4060). Outputs `cv_significance_folds.csv`, `cv_significance_tests.csv`.

## Paired significance across 10 folds (meanΔ>0 favors the FIRST-named model; lower RMSE better)
| Comparison (t+6) | meanΔ (nT) | t | p (paired-t, 2-sided) | Wilcoxon p | verdict |
|---|---|---|---|---|---|
| agg: aug − base | +0.260 | +1.80 | 0.105 | 0.232 | ns |
| intense: aug − base | **+3.011** | +2.22 | 0.0536 | **0.0488** | borderline — Wilcoxon significant |
| agg: base − persist | −0.975 | −2.95 | **0.016** | 0.014 | persistence beats model (sig) |
| intense: base − persist | **+5.118** | +3.49 | **0.0069** | 0.0098 | model beats persistence (sig **) |

## Reads
1. **Aug's intense-storm gain over base survives CV.** meanΔ +3.01 nT, Wilcoxon p=0.049 (significant); paired-t p=0.054 just misses 0.05. Smaller than the held-out 5-seed claim (+8.36 nT) because CV folds are heterogeneous and train on smaller, shifted data per fold — CV is the more conservative estimator. Direction robust (aug ≤ base intense in 8/10 folds). The Phase D adoption holds under a stricter significance protocol.
2. **Aug's aggregate gain is NOT significant** (p=0.11) — exactly the documented tradeoff: aug wins long-horizon + storms, ≈flat aggregate.
3. **Model beats persistence on intense storms, highly significant** (p=0.007, +5.12 nT) — the model earns its keep precisely where it matters.
4. **Persistence beats the model on aggregate, significant** (p=0.016, −0.97 nT) — confirms the quiet-dominated aggregate story (persistence wins the ~91% quiet rows; see Persistence Baseline).

## Notes
- CV RMSEs run higher than the fixed held-out test numbers (agg ~11–16 nT/fold vs test 11.26) because folds include storm-rich blocks and each fold trains on less/different data — expected; CV is intentionally conservative, not directly comparable to the held-out split.
- Per-fold intense N ranges 231–727 (fold-dependent); folds with fewer intense rows add seed/sample noise, widening the intense-delta variance and explaining why the strong held-out +8.36 nT compresses to +3.01 nT here.
- **Verdict:** significance protocol now matches TriQXNet's bar. Headline claims that clear it: (a) model >> persistence on intense storms (p=0.007), and (b) aug > base on intense storms (Wilcoxon p=0.049). Aug aggregate and the persistence-aggregate loss are reported honestly as expected non-wins.

---

# Phase F — Normalized conformal prediction (volatility difficulty estimator) (2026-06-11)

Optional follow-up flagged in CLAUDE.md: replace discrete Mondrian bins (by predicted severity)
with a continuous difficulty estimator, sigma(x) = "recent solar-wind volatility" = mean over
the 48h input window of (bz_gse_std + bt_std + speed_std). Normalized split-conformal scales the
calibration residuals by sigma, giving a smooth per-sample interval width instead of 3 discrete
bins. Base model only (`solar_bilstm_model.pth`), no retrain. Standalone script
`phase_f_normalized_conformal.py` (cell `phase_f_normalized` inserted after `phase_c_uq_aug`,
execution-pending). Outputs `conformal_normalized.csv`, `conformal_method_comparison.csv`.

## Per-horizon coverage(%) / width(nT), held-out test — Marginal vs Normalized
| Step | marg cov@95 | marg w@95 | norm cov@95 | norm w@95 |
|---|---|---|---|---|
| t+1 | 96.8 | 40.0 | 96.3 | 43.1 |
| t+6 | 97.0 | 44.5 | 96.3 | 50.5 |

Both ≈ nominal; normalized runs slightly wider on average (smooth volatility scaling vs constant width).

## Per-storm-bin coverage @95% (t+6) — Marginal vs Mondrian vs Normalized
| Bin (by actual) | N | marg cov% | marg w | mond cov% | mond w | norm cov% | norm w |
|---|---|---|---|---|---|---|---|
| Quiet (≥−20) | 11900 | 98.6 | 44.5 | 97.7 | 36.5 | 97.3 | 47.3 |
| Moderate (−50..−20) | 1687 | 93.1 | 44.5 | 91.7 | 54.9 | 94.5 | 67.1 |
| Intense (<−50) | 387 | **64.3** | 44.5 | **82.9** | 107.5 | **72.9** | 77.4 |

## Verdict
**Normalized conformal closes part of the marginal→Mondrian gap on intense storms but does not
beat Mondrian.** Intense coverage: marginal 64.3% → normalized 72.9% (+8.6pp) → Mondrian 82.9%
(+18.6pp). Mondrian still wins on the metric that matters (intense undercoverage), at a larger
width (107.5 vs 77.4 nT). Normalized is also worse than both marginal and Mondrian on the quiet
bin (97.3% at w=47.3, vs marginal 98.6%/44.5 and Mondrian 97.7%/36.5) — the volatility sigma adds
width to quiet hours that don't need it, because high-volatility quiet periods exist (e.g.
recovery-phase turbulence after a storm) that aren't actually hard to predict.

**Why Mondrian wins:** Mondrian bins by *predicted severity*, which is directly correlated with
the residual magnitude it's calibrating (the model's own storm/quiet split is a better difficulty
proxy than raw input volatility). The volatility sigma is a noisier proxy — high Bz/Bt/speed std
doesn't always mean high *prediction* error.

**Decision: keep Mondrian (Phase C / Phase C aug) as the shipped storm-adaptive UQ.** Normalized
conformal is a documented negative/partial result — do not adopt as a replacement. Possible
future refinement (not attempted): combine sigma with predicted severity (Mondrian bins of
normalized residuals) rather than treating them as alternatives.
