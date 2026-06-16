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
- `solar_bilstm_omni1m_model.pth` — base + 1-min OMNI augmentation (Phase D). **t+6 storm warning ONLY when live Dst unavailable** (Phase N: persistence 31.43 beats aug 36.10 at t+6 intense when Dst is available). Held-out test t+6: aggregate 10.94, **intense 36.10 nT (−19% vs base, but still loses to persistence)**. 5-seed paired Δ: aggregate +0.48±0.28, intense +8.36±3.39 (all 5 seeds positive). Crossover ~t+4; loses t+1–t+3/quiet (t+6 quiet 7.94→8.51).
- `solar_bilstm_idx_model.pth` — base + **Kp/ap/AE geomag-index INPUTS** (Phase J, next-step #4). **Primary t+1–t+3 storm warning.** 32 feat (paper-29 + lagged Kp/ap/AE). Checkpoint test (seed42): agg 9.83, intense 36.70, **t+1 intense 28.01 nT (−21% vs base 35.49)**, t+6 intense 45.49 (no t+6 gain). 5-seed paired Δ vs base (rebuilt, same windows): t+1 intense **+4.84±1.53 (all 5 +, t-p .002)**, all-horiz intense +2.80±1.17 (t-p .006), agg +0.55±0.14. Gain **front-loaded** (decays to ns by t+6) — complements OMNI-aug (owns t+6). Scaler `idx_scaler.pkl`. ⚠️ Deploy needs **near-real-time** AE/Kp/ap (quicklook): gain halves by ~3h index staleness, gone by ~6h.

**UQ:** Mondrian conformal (by predicted severity) = shipped storm-adaptive intervals. Aug-model recalibrated (Phase C aug): intense @95% t+6 coverage 90.4% (base marginal 64.3%). Use aug-model UQ storm deployment. Marginal width grows t+1→t+6 (40.0→44.5 nT @95%) = honest error-growth-with-lead, which TriQXNet's t+1-only UQ can't show.

**Storm-conditional t+6 (base, held-out test):** Quiet (≥−20, N=11900) 7.94 | Moderate (−50..−20, N=1687) 12.16 | Intense (<−50, N=387) 44.64.

**Persistence (CORRECTED Phase N 2026-06-16):** persistence beats model at every horizon for **aggregate t+1–t+5 AND for intense at ALL horizons t+1–t+6** on the fixed held-out test. Model wins only t+6 **aggregate** (base 11.26 < persist 12.71) — quiet/moderate-driven, NOT intense. Base deploy: persistence t+1–t+5; t+6 use base only for the aggregate/quiet regime, **persistence for intense** (persist 31.43 < base 44.64 < aug-deployed 36.10). Switchover by horizon only — no threshold (hybrid dead, see below). **Storm-warning refinement (Phase J):** for intense storms, `idx`-model beats base at t+1–t+3 (t+1 intense 28.0 vs base 35.5) when near-real-time indices available — but persistence (t+1 intense 12.24) still beats idx; idx-model's value is when Dst itself is unavailable/stale (model inputs exclude Dst). With live Dst, persistence is the intense baseline to beat and none of base/aug/idx beats it on this fixed test.

**✅ Phase N discrepancy RESOLVED (2026-06-16):** confirmed by recompute (`_phase_n_verify.py` → `phase_n_verify.csv`). Base reproduces documented t+6 agg 11.26 / intense 44.64 EXACTLY → preds trustworthy. **Persistence wins intense at every horizon** (t+1 12.24→t+6 31.43, vs base 35.49→44.64). "OMNI-aug t+6 for intense" deploy rule was WRONG — persistence owns t+6 intense (31.43 < aug 36.10). Reconciled vs Phase-E CV (base−persist +5.12 sig): CV = storm-rich blocks / all-horizon avg / less train-per-fold, "not directly comparable"; fixed-test is deploy-relevant. Mechanism: model inputs exclude Dst, persistence exploits storm-time Dst autocorrelation. See "Phase N discrepancy RESOLVED" section below for full per-horizon table + aug-scaler caveat.

**vs TriQXNet (arXiv:2407.06658v3):** same dataset/features/split. Our t+1 10.14 vs their 9.27 (their number is t0/t+1 avg, t0=trivial nowcast, quiet-dominated; on extremes they hit t0 20.33/t+1 20.86). Gap NOT window (Step 3) nor arch (A4) — their 3-pipeline ensemble + t0-inflated metric, not worth chasing. We add multi-horizon + honest storm-conditional eval + per-horizon UQ they lack.

## Significance (Phase E — leakage-safe blocked 10-fold paired)

- intense `aug−base` +3.01 nT (Wilcoxon p=0.049 **sig**, paired-t p=0.054). Phase D holds under stricter protocol.
- intense `base−persist` +5.12 nT (p=0.007 **sig**) — model >> persistence on storms.
- agg `aug−base` ns (p=0.105); agg `base−persist` −0.98 (p=0.016, persistence wins aggregate) — both expected, quiet-dominated.
- CV intense N/fold 231–727; CV RMSE runs higher than fixed held-out (storm-rich blocks, less train/fold) — not directly comparable by design.

## ✅ Phase N discrepancy RESOLVED (2026-06-16): persistence owns t+6 intense on fixed test.
Recomputed documented base + OMNI-aug preds on the exact 387-hour held-out test (`_phase_n_verify.py` → `phase_n_verify.csv`). Base reproduces documented numbers **EXACTLY** (t+6 agg 11.26, intense 44.64 — scaler refit on `_seq_base.npz` Xtr verified correct). **Result: persistence beats base at EVERY horizon intense** (t+1 12.24<35.49, t+2 19.66<34.88, t+3 24.61<35.78, t+4 27.42<38.27, t+5 29.51<41.39, t+6 31.43<44.64) AND beats documented OMNI-aug at t+6 (31.43<36.10).
**→ Deploy rule fixed:** "OMNI-aug t+6 for intense" was WRONG; **persistence owns t+6 intense** (and all intense horizons) on the fixed held-out test. Model still wins t+6 AGGREGATE (base 11.26 < persist 12.71) — that aggregate win is quiet/moderate-driven, NOT intense.
**Why no contradiction with Phase E:** model inputs exclude Dst; persistence exploits Dst autocorrelation (very strong 6h-out during a storm). Phase E CV (base−persist +5.12 sig) is storm-rich blocks / all-horizon avg / less train-per-fold = "not directly comparable" — the fixed-test number is the deploy-relevant one. **Caveat:** aug-model recompute with base scaler gave 29.40 (would edge persistence) but that's a scaler mismatch (aug trained on augmented train, no cached aug scaler) → discard; deployed aug uses its own scaler = documented 36.10, which loses to persistence. Even the favorable 29.40 only ties persistence.

## STATUS: Phases A–M COMPLETE. Phase J (Kp/ap/AE inputs) ADOPTED — first/only new-modality win. Phase K (IG interpretability) DONE — all 5 next-steps closed. Phase L (SYM-H target) REJECTED (#11). Phase M (CME+flare+F10.7 inputs) TESTED & REJECTED (#12) — last untried modality, regressed all horizons. **ACCURACY LEVER FULLY EXHAUSTED** (raw SDO/AIA imagery physically can't cover 1998/2004 periods → dead for this dataset). Remaining work = bucket 3 (productionize/publish) only.

Phase J (next-step #4) = first/only external-data lever to beat base on held-out test (leakage-verified, multi-seed). All other levers now ruled out: SHAP done (#5 = Phase K), SYM-H rejected (#11), CME+flare+F10.7 rejected (#12), raw imagery impossible (pre-2010 periods). See DO NOT REPEAT.

### Phase J record — date-recovery + geomag indices (2026-06-13)
MagNet base periods are **anonymized** (relative timedelta). Recovered absolute dates by Dst-fingerprint cross-correlation vs OMNI hourly Dst (`_date_recovery_full.py` → `_magnet_dates.json`): train_a 1998-02-16, train_c 2004-05-01 (RMSE ~0 exact lock), train_b 2013-06-01 (RMSE 5.6, unambiguous 2nd-best gap). Enabled merging OMNI Kp/ap/AE into base periods incl held-out test. **Leakage guard:** indices live in input window only (`create_sequences`) AND cadence-lagged to last completed interval before t (AE 1h, Kp/ap 3h). **Leak-robustness verified:** lag-decay sweep (`geomag_index_lag_decay.csv`) shows t+1 intense gain decays SMOOTHLY with extra index staleness (+0h +7.5 → +1h +5.9 → +3h +4.1 → +6h ~0), no cliff → real magnetosphere-autocorrelation signal, not boundary label-leak.

### Pending-cell record — CLOSED 2026-06-11 (option 1: embed, no rerun)
4 pending cells (`cv_significance`, `phase_c_uq`, `phase_c_uq_aug`, `phase_f_normalized`) had empty in-notebook outputs (computed standalone, env `dnn`). Result tables now embedded as markdown cells right after each (ids `*_results`); CSVs remain source of truth. Code cells still carry `EXECUTION PENDING` header + no kernel rerun (cv_significance = ~4.3h GPU, reproduces same CSV). Backup `SolarWindLSTM.ipynb.bak`.

## Options to proceed (model done; pick by goal)

**Bucket 1 — housekeeping: DONE** (results embedded above). Optional leftover: reword `EXECUTION PENDING` headers → "results embedded below"; or full kernel rerun to populate real outputs (incl 4.3h CV).

**Bucket 2 — new-data levers: CLOSED.** curated-storm #10, SYM-H #11, CME+flare+F10.7 modality #12 all rejected. Only raw solar imagery (SDO/AIA) untried, but SDO launched 2010 → can't cover recovered train_a(1998)/train_c(2004) periods → physically dead for this dataset. No remaining accuracy lever.

**Bucket 3 — productionize / publish (work is paper-shaped) — ONLY remaining path:**
- Real-time inference: persistence t+1–t+5, base+aug t+6, idx-model t+1–t+3 (when near-real-time Kp/ap/AE), Mondrian UQ bands. Plus index-staleness monitor + mag-outage fallback (Phase I).
- Writeup — novel vs TriQXNet = multi-horizon + honest storm-conditional + per-horizon UQ.

Recommendation: accuracy lever fully exhausted (storm-RMSE bound; transforms/curation/SYM-H/CME all rejected; raw imagery physically impossible here) → ship/finish via bucket 3.

## Next steps (from deep-research-report cross-check — recommended order)

Accuracy lever exhausted; these add value via operational framing + honest eval + new modality (the actual differentiation vs TriQXNet). Ordered cheapest→biggest. Items below are NEW (survived DO NOT REPEAT check); report's transformer/ensemble/loss-reweight proposals all collide with rejected list — skip them.

1. **Storm-detection eval metrics** — ✅ DONE (Phase G, cell `storm_detect_metrics` + `_results`, held-out TEST, CSV `storm_detection_metrics.csv`). POD/FAR/CSI/HSS/BIAS + per-horizon + ROC at Dst<−50/−80/−100. Result: model **under-forecasts intensity** on extremes (BIAS 0.67@−80, 0.54@−100 → low FAR 0.16/0.04 but lower POD than persistence 0.57/0.52 vs 0.72/0.69); at −50 persistence edges CSI (0.53 vs 0.52). Per-horizon honest skill decay POD 0.695→0.669 t+1→t+6. Confirms conservative-alarm regime (data-scarcity, #8) and motivates #2 (tunable prob threshold). NB: notebook Phase-1 inference cell uses val_loader (early-stop set) — Phase G re-runs checkpoint on test_loader for clean hold-out.
2. **Multi-task storm-classifier head** — ✅ DONE (Phase H, cells `mtl_model_def`/`mtl_train`/`mtl_eval` + `_results`, checkpoint `solar_bilstm_mtl_model.pth`, adopted base untouched). Shared encoder + reg head + clf head (P(Dst<−50) per horizon), joint loss = base weighted MSE + 200·BCE. Held-out test: **AUC 0.983, Brier 0.012, ECE 0.003** (well-calibrated — capability Phase G lacked). Reg cost small (agg t+6 11.26→11.58; intense 44.64→46.14 within seed noise; moderate improved 12.16→11.56). **Win:** tunable prob threshold — at τ=0.30 POD 0.713/CSI 0.573/HSS 0.721 beats persistence (0.695/0.533) AND deterministic base (0.669/0.519). Resolves Phase G's fixed-conservative-bias finding. Adoptable as storm-alert layer. (Single seed OK: headline is calibration/AUC; intense RMSE got worse not better, so no false-win risk.) Extension: add −80/−100 heads.
3. **Robustness / degradation tests** — ✅ DONE (Phase I, cell `robustness_tests` + `_results`, CSV `robustness_metrics.csv`). No retrain; base checkpoint on perturbed test inputs. **Meets <1.5× target at 10% dropout** (agg 1.02–1.10×, intense 1.05–1.11×). Noise asymmetry: σ=2 noise → agg 1.89× but intense ~1.0× (storm signal robust, quiet fragile). **Magnetometer outage critical: intense 2.05× (44.6→91.5 nT)** — Bz drives Dst; plasma outage mild (1.03×) → prioritize mag redundancy. (Single-satellite mapped to instrument-block outage since MagNet data is pre-merged, no per-satellite source.) Novel vs literature.
4. **Geomagnetic-index INPUTS (Kp/Ap/AE)** — ✅ DONE & ADOPTED (Phase J, checkpoint `solar_bilstm_idx_model.pth`, cells `phase_j_build`/`phase_j_results`, CSVs `geomag_index_ab.csv`/`geomag_index_lag_decay.csv`). New modality, NOT a transform → escapes #9. Required date-recovery (anonymized periods, see Phase J record). Leakage handled: input-window-only + cadence-lag + verified by smooth lag-decay (no cliff). **5-seed win, all positive:** t+1 intense +4.84±1.53 (−21%, t-p .002), all-horiz intense +2.80±1.17, agg +0.55±0.14. Front-loaded — owns t+1–t+3 storm warning, complements OMNI-aug (t+6). Deploy caveat: needs near-real-time indices. Extension: SYM-H/ASY indices, or +6h-lagged-only variant for latency-robust deploy.
5. **Interpretability (SHAP / integrated gradients)** — ✅ DONE (Phase K, cells `phase_k_shap`/`phase_k_results`, script `_shap_ig.py`, CSVs `shap_feature_importance.csv`/`shap_time_profile.csv`, plot `shap_ig.png`). Hand-rolled IG (shap/captum absent), base model, held-out TEST (13974 seqs, 529 storm), baseline=scaled-zero, M=32 steps. **Top features:** bz_gsm_mean (0.161), speed_mean (0.161), theta_gsm_mean (0.154), bt_mean (0.109). **On storm rows:** bt_mean (0.545) + bz_gsm_mean (0.511) + speed_mean (0.306) dominate — southward-Bz/field-magnitude/speed = exactly the Dst-driving physics. **Recency:** mean|IG| t=−1 (0.310) / t=−48 (0.025) = 12.4×, last 6h carry bulk (echoes Phase J autocorrelation). **Validates #9:** model already concentrates on Bz/Bt/speed that the rejected coupling transforms (vBs/clock_angle/epsilon/Newell) recombine → they were redundant, not under-used. Interpretability gap closed, paper-friendly.

Physics baseline (Burton/OM2000) — ✅ DONE (Phase N, `_burton_baseline.py` → `burton_baseline.csv`). Held-out test, N=13974 (intense 387 — composition verified vs ML test). **(1) ML >> Burton as forecaster** (frozen-driver): t+6 agg 11.26 vs 17.14, quiet 7.94 vs 13.83, intense 44.64 vs 56.47 — ML ~34% better aggregate than classic coupling ODE run as forecaster. **(2) ML beats Burton even as HINDCAST** (perfect concurrent driver): agg 11.26 vs 14.62 — forecasting 6h ahead from solar wind beats the empirical formula even when it's handed the actual concurrent driver; perfect-driver edge shows only on intense (hindcast 41.74 < ML 44.64). Strong paper point: ML captures coupling the analytic formula misses. **Surfaced the persistence-t+6-intense discrepancy above.**

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

11. **SYM-H target (hourly-mean OR hourly-min) — Phase L, both rejected.** Probe (16yr, leakage-safe recovered dates): hourly-**mean** SYM-H ≈ Dst (corr 0.945, RMSE 6.23) → relabel trivial. hourly-**min** SYM-H exposes +16–30% more extreme-storm-labeled hours (grows at extremes), so 5-seed paired tested it as target on identical a/b/c windows vs Dst-control. Result: RMSE uniformly **worse** (t+6 symh 12.5–13.2 vs dst 11.1–12.1, no seed overlap; intense 43.99 vs 38.54). Detection: SYM-H-min **does** raise extreme POD (+0.068 @−80/−100) and fixes Phase G under-forecast BIAS (0.57/0.80→0.78/1.06, near-unbiased) BUT FAR blows up (0.03→0.19, 0.19→0.32) → net **CSI tie-to-worse**. Killer: that POD/BIAS gain = lower effective alarm threshold baked into target — **Phase H τ-sweep already buys the same POD/FAR trade on the existing Dst model, no new data.** SYM-H-min dominated by cheaper existing solution. Closes bucket-2 SYM-H sub-option. Artifacts: `_symh_probe.py`/`_build_symh.py`/`_symh_train.py`, `symh_perhorizon.csv`/`symh_stormcond.csv`/`symh_detection.csv`, `solar_bilstm_symh_model.pth` (ablation only, don't promote).

12. **Solar-source modality: CME catalog + GOES flares + F10.7 INPUTS — Phase M, rejected.** Last untried new modality (bucket 2). Probe PASSED marginally (earth-directed CME launches predict storms ~2× lift at 24–108h lead; ballistic arrival ±12h 2.0–2.2×) → built 8 causal-lagged feats (CME recency/speed/count/arrival-proximity, M-flare recency, flare-energy-24h, F10.7+27d-trend) on 37-feat seq, 5-seed paired vs 29-feat base, identical windows. **Result: regressed every horizon×regime.** all-intense Δ −2.46±3.60 (2/5 +), agg −0.43 (2/5), moderate −1.10 (**0/5**). **Lead hypothesis specifically failed: t+6 intense −3.38±2.85 (1/5 +)** — the horizon CME transit-lead should help most got *worse*. No band all-positive → fails adoption bar. Cause: sparse/weak feats (~2× lift on 0.94% base) dilute the 29 clean solar-wind feats the model already mines for Bz/Bt/speed (Phase K SHAP); same signal:noise failure as #9/#10. Probe-greenlight≠model-win — the marginal 2× lift was too weak to survive as input. Lag-decay leak-check skipped (only validates wins; this is a loss). **Closes bucket-2 entirely → accuracy lever exhausted; only raw solar imagery (SDO/AIA) untried but physically can't cover 1998/2004 periods → dead for this dataset.** Artifacts: `_cme_probe.py`/`_fetch_solar_sources.py`/`_build_features_cme.py`/`_ab_train_cme.py`, `cme_ab.csv`, caches `_cme_catalog.pkl`/`_goes_flares.pkl`/`_f107.pkl`/`_seq_cme.npz`/`_seq_cme_base.npz`. No checkpoint (A/B only, don't promote).

**Process rule: ALWAYS multi-seed before declaring storm-metric win.** Intense N≈387 → ±1–3.5 nT seed noise, enough to fake ±4 nT improvement on one seed (nearly caused false ADOPT). **Phase M corollary: a passing cheap probe (~2× association lift) does NOT predict a model win — modality must beat base AS INPUT under 5-seed paired, signal can be too weak/sparse to survive feature dilution.**

## Bucket 3 — productionize (Phase O, 2026-06-16): real-time inference module SHIPPED
`solar_dst_inference.py` — `DstForecaster.predict(window29, last_dst, dst_age_h, window32, index_age_h, mag_ok)` → per-horizon `Dst`/`source`/`lo95`/`hi95`/`storm_prob`/`severity` + `storm_alert` + monitor notes. Routing matches corrected Phase-N policy: persistence t+1–t+5 + t+6-intense when Dst fresh; base t+6-aggregate; idx t+1–t+3 / base t+4–t+6 when Dst stale; mtl classifier P(Dst<−50) alarm at τ=0.30; Mondrian 95% bands by predicted severity (quiet 36.5 / moderate 54.9 / intense 107.5, base bins); monitors = Dst-staleness (>1.5h → model path), index-staleness (idx halves >3h, off >6h), mag-outage (intense bands ×2.05). OMNI-aug intentionally NOT routed (persistence owns t+6 intense; aug scaler not persisted). Loads base/idx/mtl checkpoints + `base_scaler.pkl` (refit on `_seq_base.npz` Xtr, reproduces documented base EXACTLY) + `idx_scaler.pkl`. Smoke test `_smoke_inference.py`: 4 routing scenarios + batch base intense t+6 RMSE=44.64 PASS. Remaining bucket-3: paper figures/draft.

## Repo layout (cleaned 2026-06-16)

Root = runtime only: `SolarWindLSTM.ipynb`, `solar_dst_inference.py`, `base_scaler.pkl`, `idx_scaler.pkl`, `.pth` checkpoints (gitignored, local), `README.md`, `CLAUDE.md`. Standalone phase scripts (`_*.py`, `phase_f_normalized_conformal.py`, `test_transformer.py`) + small artifacts (`_magnet_dates.json`, `_cme_newfeat_names.npy`) → `scripts/`. Historical reports/papers (`PAPER_*.md`, `PROGRESS_REPORT.md`, `RESEARCH_NOTES.md`, `ROADMAP.md`, `results.md`, `deep-research-report.md`, etc.) → `docs/`. Plots → `images/` (display) + `figs/` (paper, README-referenced). Run scripts from repo root (cache paths are cwd-relative). Script/CSV names below are unchanged — find scripts under `scripts/`.

## Key cells

Pipeline: `0d486255` (split/exec) · `bc9878f0` (model def) · `8a4080ce` (train) · `af6ee96c` (inference) · `446d08ce`/`3b7398e3` (per-step + storm tables) · `37d6b8b0` (A3 TriQXNet) · `bb012d7a` (writes `results.md`).
Adopted aug: `omni1m_build` (15-storm list, true hourly mean+std from `OMNI_HRO_1MIN`, label DST1800) → `omni1m_multiseed`.
UQ/sig: `phase_c_uq`, `phase_c_uq_aug`, `cv_significance`, `phase_f_normalized`.
Storm-detection (Phase G, next-step #1): `storm_detect_metrics` (+ `_results` md) → `storm_detection_metrics.csv`/`.png`. Self-contained, re-runs checkpoint on test_loader.
Multi-task classifier (Phase H, next-step #2): `mtl_model_def`/`mtl_train`/`mtl_eval` (+ `_results` md) → `solar_bilstm_mtl_model.pth`, `mtl_regression_ab.csv`, `mtl_prob_threshold_sweep.csv`, `mtl_storm_classifier.png`. Code cells carry EXECUTION PENDING (computed standalone env dnn, ~train+eval).
Robustness (Phase I, next-step #3): `robustness_tests` (+ `_results` md) → `robustness_metrics.csv`/`.png`. No retrain, base checkpoint on perturbed test inputs.
Geomag-index inputs (Phase J, next-step #4): `phase_j_build` (date-recovery + Kp/ap/AE merge, EXECUTION PENDING) + `phase_j_results` md → `solar_bilstm_idx_model.pth`/`idx_scaler.pkl`, `geomag_index_ab.csv`/`geomag_index_lag_decay.csv`. Standalone scripts: `_date_recovery_full.py`, `_fetch_indices.py`, `_build_features_indices.py`, `_ab_train_indices.py`, `_lag_sweep.py`, `_finalize_phase_j.py`. Caches `_magnet_dates.json`, `_omni_dst_1995_2020.pkl`, `_omni_indices_1998_2019.pkl`, `_seq_base.npz`/`_seq_aug.npz`.
Interpretability (Phase K, next-step #5): `phase_k_shap` (IG code, EXECUTION PENDING — computed standalone via `_shap_ig.py`) + `phase_k_results` md → `shap_feature_importance.csv`, `shap_time_profile.csv`, `shap_ig.png`. Hand-rolled IG, cuDNN disabled for eval-mode RNN backward, base checkpoint on held-out test.
SYM-H target (Phase L, bucket-2, REJECTED): `phase_l_results` md only (computed standalone). Scripts `_symh_probe.py`/`_build_symh.py`/`_symh_train.py` → `symh_perhorizon.csv`/`symh_stormcond.csv`/`symh_detection.csv`, `solar_bilstm_symh_model.pth`, caches `_omni_symh_1min.pkl`/`_seq_symh.npz`/`_symh_aligned.pkl`. Don't promote.
Physics baseline (Phase N, bucket-3 context): `_burton_baseline.py` (OM2000 coupling ODE, hindcast + frozen-driver forecast + persistence on held-out test) → `burton_baseline.csv`. Standalone, no checkpoint. Surfaced persistence-t+6-intense OPEN DISCREPANCY (see Current models section).
Solar-source inputs CME+flares+F10.7 (Phase M, bucket-2, REJECTED #12): scripts `_cme_probe.py` (probe, passed 2× — `_cme_probe.log`), `_fetch_solar_sources.py` (GOES flares 1998–2017 + LISIRD F10.7), `_build_features_cme.py` (8 causal feats → `_seq_cme.npz`/`_seq_cme_base.npz`, names `_cme_newfeat_names.npy`), `_ab_train_cme.py` (5-seed paired → `cme_ab.csv`, `_cme_train.log`). CME catalog parsed from CDAW `univ_all.txt` → `_cme_catalog.pkl`; flares `_goes_flares.pkl`; F10.7 `_f107.pkl`. No checkpoint. Don't promote.
Ablation artifacts only (rejected, don't promote): `conv1d_step2`, `conv1d_plainmse`, `seq_ablation_step3`, `omni_download`/`omni_features`/`omni_augment`/`omni_multiseed`.

Full historical tables: `results.md`, `PROGRESS_REPORT.md`.