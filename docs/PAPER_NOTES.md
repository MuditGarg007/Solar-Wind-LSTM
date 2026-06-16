# Paper Notes — Key Points & Findings

> **2026-06-16 reframe (Phase N):** persistence beats the model on **intense point-RMSE at every horizon** on the fixed held-out test. The old "model wins t+6 mainly intense, +17% skill" claim was wrong (it came from storm-rich CV blocks, not the fixed test). The honest finding is now the **central contribution** — see §"Reframed contribution" below.

## Problem
Predict Dst geomagnetic index 6h ahead (multi-horizon t+1..t+6) from solar wind data (OMNI/MagNet dataset). Early storm warning. Model inputs **exclude Dst** (forecast from upstream solar wind only).

## Reframed contribution (the honesty IS the novelty)

Most Dst-forecasting papers report a single aggregate RMSE on quiet-dominated data and claim a persistence beat — hiding that **persistence is a deceptively strong baseline during storms** because Dst is strongly autocorrelated 6h out. We show, on a fixed held-out test, that a solar-wind-only BiLSTM does **NOT** beat persistence on intense-storm point-RMSE at any horizon. The value of the model is therefore **not** lower storm point-error; it is:

1. **Dst-independent forecasting** — the model never sees Dst, so it remains usable when Dst telemetry is stale/unavailable (persistence dies the moment the last Dst is old). Operational complement, not replacement.
2. **Calibrated probabilistic storm detection** — multi-task classifier head (Phase H): AUC 0.983, Brier 0.012, ECE 0.003, tunable threshold beats persistence AND deterministic model on POD/CSI/HSS. This is the real operational win (alerting), distinct from point-RMSE.
3. **Per-horizon, severity-adaptive uncertainty** — Mondrian conformal intervals widen honestly with lead time (40.0→44.5 nT @95%, t+1→t+6); persistence gives no calibrated UQ.
4. **Honest multi-horizon, per-regime evaluation** — per-horizon degradation curves + quiet/moderate/intense split + physics baseline (Burton/OM2000) + persistence at every horizon. Most works report none of this.
5. **Robustness / degradation characterization** (Phase I) — graceful under 10% dropout (≤1.1×); magnetometer outage is the critical failure (intense 2.05×) → mag-redundancy is the operational priority.

## Key honest results (held-out test, fixed N=387 intense)

**Intense-storm RMSE by horizon (nT) — persistence wins everywhere:**

| horizon | persistence | base model | OMNI-aug (deployed) |
|---|---|---|---|
| t+1 | **12.24** | 35.49 | 36.10-class |
| t+3 | **24.61** | 35.78 | — |
| t+6 | **31.43** | 44.64 | 36.10 |

**Aggregate RMSE t+6:** base 11.26 < persistence 12.71 → model wins **aggregate** t+6 (quiet/moderate-driven, NOT intense). Persistence wins aggregate t+1–t+5.

**Storm-conditional t+6 (base):** Quiet (N=11900) 7.94 | Moderate (N=1687) 12.16 | Intense (N=387) 44.64.

**Mechanism:** model excludes Dst; persistence exploits storm-time Dst autocorrelation. The model is the better forecaster only where Dst autocorrelation is weak/aggregate-averaged.

## Storm detection (Phase H/G — the operational headline)
- Multi-task: shared encoder + regression head + per-horizon P(Dst<−50) classifier; joint loss = weighted MSE + 200·BCE.
- AUC 0.983, Brier 0.012, ECE 0.003 (well-calibrated).
- Tunable threshold τ=0.30: POD 0.713 / CSI 0.573 / HSS 0.721 — beats persistence (0.695/0.533) and deterministic base (0.669/0.519).
- Resolves Phase-G finding that the deterministic model under-forecasts storm intensity (conservative-alarm regime, data-scarcity bound).

## Geomag-index inputs (Phase J — only adopted accuracy lever)
- Adding lagged Kp/ap/AE inputs (32 feat): 5-seed paired, all positive — t+1 intense +4.84±1.53 (−21%), front-loaded (decays to ns by t+6).
- Required recovering anonymized MagNet absolute dates via Dst-fingerprint cross-correlation. Leakage-safe: input-window-only + cadence-lag, verified by smooth lag-decay (no cliff).
- Caveat: even idx-model intense does NOT beat persistence when live Dst exists; its value is the Dst-stale/unavailable regime. Needs near-real-time indices (gain halves by ~3h staleness).

## Interpretability (Phase K — hand-rolled integrated gradients)
- Top features: bz_gsm, speed, theta_gsm, bt. On storm rows: bt + bz_gsm + speed dominate = exactly the Dst-driving physics.
- Recency: |IG| at t=−1 is 12.4× t=−48; last 6h carry the bulk → echoes the autocorrelation persistence exploits.
- Explains why physics-coupling transforms (vBs/clock-angle/epsilon/Newell) were redundant: model already concentrates on Bz/Bt/speed.

## Physics baseline (Phase N — Burton/OM2000)
- ML beats Burton as a forecaster (t+6 agg 11.26 vs 17.14) AND even as a hindcast with perfect concurrent driver (11.26 vs 14.62) — ML captures coupling the analytic ODE misses.
- Surfaced the persistence-intense discrepancy that drove the reframe.

## vs TriQXNet (arXiv:2407.06658v3)
- Same dataset/features/split. Our t+1 10.14 vs their reported 9.27 (their number = t0/t+1 avg; t0 = trivial nowcast, quiet-dominated; on extremes they hit t0/t+1 = 20.33/20.86).
- Gap is NOT window length or architecture — their 3-pipeline ensemble + t0-inflated metric. We add multi-horizon + honest storm-conditional eval + per-horizon UQ + calibrated detection, which they lack.

## Architecture
2-layer BiLSTM (hidden=128/dir, dropout=0.3) + attention over 48h window → concat(context, last hidden) → linear → 6-step Dst. Two-tier asymmetric weighted MSE (Dst<−20 →5×, Dst<−50 →15×).

## Negative results (shows rigor — full trail)
- More OMNI storms (27 vs 15) regressed — curation > count.
- Longer windows (96/128h) didn't help; 48h optimal.
- Ensemble/stacking (LGBM, GRU correction) traded quiet gains for worse storms.
- Persistence-vs-model hybrid switching: dead (both grid searches picked model ~5%, lost on intense).
- Loss reweighting/oversampling: all failed — storm perf data-scarcity-bound, not loss-bound.
- Physics-coupling features (vBs/clock-angle/epsilon/Newell): worse — redundant (confirmed by Phase K).
- Curated extreme storms (18 vs 15, Dst<−200, ssn-matched): regressed vs 15-set.
- Transformer encoder: lost to BiLSTM-attention every horizon/regime (overfits at this data scale).
- SYM-H target (mean or min): rejected — relabel trivial or POD/FAR trade already buyable via Phase-H τ-sweep on existing Dst model.
- CME catalog + GOES flares + F10.7 inputs (Phase M): probe passed ~2× lift but regressed every horizon as input — signal too weak/sparse, dilutes clean solar-wind features.
- **Accuracy lever exhausted:** raw SDO/AIA imagery is the only untried modality but launched 2010 → can't cover recovered 1998/2004 periods → physically dead for this dataset.

## Statistical-significance caveat (important for honesty)
- Phase-E blocked 10-fold CV reported intense base−persist +5.12 nT (Wilcoxon p=0.007). **This does NOT transfer to the fixed held-out test** (Phase N: persistence wins intense everywhere). CV blocks are storm-rich, all-horizon-averaged, less train-per-fold → "not directly comparable." Report the fixed-test result as deploy-relevant; present CV as a separate, caveated robustness check, NOT as a persistence-beat claim.
- aug−base intense +8.36±3.39 (5-seed, all positive) is a real **aug-vs-base** effect, but aug still loses to persistence on intense — frame as relative model improvement, not absolute storm skill.

## Suggested paper framing
Title direction: "When Persistence Wins: Honest Multi-Horizon Dst Forecasting with Calibrated Storm Detection and Adaptive Uncertainty"

Narrative: most Dst papers hide that persistence is hard to beat on storms because Dst is autocorrelated. We (a) evaluate per-horizon and per-regime against persistence AND a physics baseline; (b) show a solar-wind-only model does not beat persistence on intense point-error — and argue the right contributions are Dst-independent forecasting, calibrated probabilistic storm detection (AUC 0.983), and severity-adaptive per-horizon UQ; (c) document a statistically validated ablation trail incl. negative results and a CV-vs-fixed-test discrepancy resolved in favor of the fixed test.
