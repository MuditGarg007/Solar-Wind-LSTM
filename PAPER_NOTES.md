# Paper Notes — Key Points & Findings

## Problem
Predict Dst geomagnetic index 6h ahead (multi-horizon t+1..t+6) from solar wind data (OMNI/MagNet dataset). Early storm warning.

## What differentiates this work

1. **Multi-horizon forecasting (t+1 to t+6), not single-step.** TriQXNet (arXiv:2407.06658v3) reports only t0/t+1 avg. We give honest per-horizon degradation curve.

2. **Storm-conditional evaluation, not just aggregate RMSE.** Quiet/Moderate/Intense breakdown reveals model value concentrated in storms — aggregate metrics hide this.

3. **Per-horizon uncertainty quantification (Mondrian conformal, severity-binned).** Intervals widen honestly with lead time (40.0→44.5 nT @95%, t+1→t+6) — shows error growth with horizon, which single-horizon UQ baselines can't demonstrate.

4. **1-min OMNI augmentation (Phase D)** — true hourly mean+std features from OMNI_HRO_1MIN (vs hourly-only proxy), validated multi-seed (5 seeds, all positive on intense).

5. **Rigorous statistical validation**: 5-seed paired tests, leakage-safe blocked 10-fold CV with Wilcoxon/paired-t significance (Phase E).

6. **Persistence baseline honestly reported** — persistence beats model t+1–t+5; model only wins t+6 (mainly intense storms, +17% skill). Deploy strategy = horizon-based switchover, not naive "our model always wins" claim.

## Major results (held-out test, t+6)

**Two adopted models, deployed by regime:**
- Base BiLSTM (`solar_bilstm_model.pth`): RMSE 11.26 nT, r=0.8247, R²=0.6780. Primary for t+1–t+5 and quiet conditions.
- 1-min OMNI augmented (`solar_bilstm_omni1m_model.pth`): aggregate RMSE 10.94, **intense storms 36.10 nT (−19% vs base)**. Primary for t+6 storm warning.

**5-seed paired Δ (aug vs base, t+6):**
- Aggregate: +0.48 ± 0.28 nT
- Intense: +8.36 ± 3.39 nT (all 5 seeds positive)

**Storm-conditional t+6 (base, held-out):**
| Regime | N | RMSE (nT) |
|---|---|---|
| Quiet (Dst≥−20) | 11900 | 7.94 |
| Moderate (−50<Dst<−20) | 1687 | 12.16 |
| Intense (Dst<−50) | 387 | 44.64 |

**Persistence comparison:** persistence wins t+1–t+5; model wins t+6 (especially intense, +17% skill over persistence).

## Significance (Phase E, blocked 10-fold CV)
- Intense aug−base: +3.01 nT, Wilcoxon p=0.049 (sig), paired-t p=0.054. Confirms Phase D under stricter protocol.
- Intense base−persistence: +5.12 nT, p=0.007 (sig) — model clearly beats persistence on storms.
- Aggregate aug−base: ns (p=0.105) — expected, quiet-dominated.
- Aggregate base−persistence: −0.98 nT (p=0.016) — persistence wins aggregate, expected.

## UQ results (Mondrian conformal, severity-binned)
- Aug-model recalibrated: intense @95% t+6 coverage 90.4% (vs base marginal coverage only 64.3%).
- Marginal interval width grows honestly with horizon: 40.0 → 44.5 nT (@95%, t+1→t+6).

## vs TriQXNet (arXiv:2407.06658v3)
- Same dataset/features/split.
- Our t+1: 10.14 nT vs their reported 9.27 (their number = t0/t+1 avg, t0 trivial nowcast, quiet-dominated).
- On extremes their t0/t+1 = 20.33/20.86 nT — much worse than our intense numbers in relative terms.
- Gap is NOT input window length or architecture — likely their 3-pipeline ensemble + t0-inflated metric. Not the focus; our contribution is multi-horizon + storm-conditional + per-horizon UQ, which they lack entirely.

## Architecture
2-layer BiLSTM (hidden=128/dir, dropout=0.3) + attention over 48h input window → concat(context, last hidden) → linear → 6-step Dst output. Two-tier asymmetric weighted MSE (Dst<−20 → 5×, Dst<−50 → 15×).

## Negative results worth mentioning (shows rigor)
- More OMNI storms (27 vs 15) regressed — curation > count.
- Longer input windows (96/128h) did not help; 48h optimal.
- Ensemble/stacking (LGBM, GRU correction) traded quiet gains for worse storm performance.
- Persistence-vs-model hybrid switching: dead end, both grid searches picked model only ~5% of samples and lost on intense.
- Loss reweighting/oversampling for storm bias: all failed — storm performance is data-scarcity-bound, not loss-bound.
- Physics-coupling features (vBs, clock angle, etc.) made things worse — BiLSTM already learns these nonlinear transforms.
- Curated extreme storms (18 vs 15, Dst<−200, ssn-matched) regressed vs adopted 15-storm set — confirms 15-storm set is local optimum.
- Transformer encoder (2-layer, d_model=128, 4 heads, ~276K params, same 48h window/29 features/storm-weighted loss) lost to BiLSTM-attention on every horizon and regime, held-out test, single seed=42: t+6 agg 12.56 vs 11.26, t+6 quiet 9.60 vs 7.94, t+6 intense 46.58 vs 44.64, t+1 10.62 vs 10.14. Early-stopped epoch 25 (val loss rising after ep10) — overfits on this dataset size. Confirms alt-arch pattern (Conv1D, etc.): BiLSTM-attention remains best for this data scale.

## Suggested paper framing
Title direction: "Multi-Horizon Dst Forecasting with Storm-Conditional Evaluation and Adaptive Uncertainty Quantification"

Key narrative: most Dst forecasting papers report single aggregate metric on quiet-dominated data, hiding poor storm performance — exactly when forecasts matter most. This work (a) evaluates per-horizon and per-regime, (b) shows where models add value over persistence (t+6 storms), (c) provides calibrated, severity-adaptive uncertainty bands that widen honestly with lead time, and (d) documents a thorough, statistically validated ablation trail (including negative results) for what does/doesn't help storm-time accuracy.
