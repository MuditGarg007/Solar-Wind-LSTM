# When Persistence Wins: Honest Multi-Horizon Dst Forecasting with Calibrated Storm Detection and Adaptive Uncertainty

*Working draft — generated from PAPER_NOTES.md + result CSVs. Figures in `figs/`.*

## Abstract

We forecast the Dst geomagnetic index 1–6 hours ahead from upstream solar-wind
data using a bidirectional LSTM with attention. Unlike most prior work, which
reports a single aggregate RMSE on quiet-dominated data and claims to beat a
persistence baseline, we evaluate per-horizon and per-storm-regime against both
persistence and a physics baseline (Burton/OM2000). Our central finding is a
negative one, honestly reported: a solar-wind-only model **does not** beat
persistence on intense-storm point error at any lead time (t+6 intense RMSE
31.4 nT for persistence vs 44.6 nT for the model), because Dst is strongly
autocorrelated during storms and the model is not given Dst as input. The model
wins only aggregate error at t+6 (11.3 vs 12.7 nT), driven by quiet/moderate
conditions. We argue the operationally meaningful contributions of a
solar-wind-only model are therefore (i) Dst-independent forecasting that remains
usable when Dst telemetry is stale or unavailable; (ii) a calibrated
probabilistic storm-detection head (AUC 0.983, ECE 0.003) whose tunable
threshold beats persistence on POD/CSI/HSS; and (iii) severity-adaptive,
per-horizon conformal uncertainty bands that widen honestly with lead time. We
provide a fully routed real-time inference system reflecting these findings and
a statistically validated ablation trail, including a cross-validation-versus-
fixed-test discrepancy that we resolve in favor of the fixed test.

## 1. Introduction

Geomagnetic storms, quantified by the Dst index, drive space-weather hazards.
Forecasting Dst hours ahead from L1 solar-wind monitors (ACE/DSCOVR) supports
early warning. Recent ML work (MagNet 2021; TriQXNet 2024; multi-fidelity GRU
2023) reports strong aggregate RMSE, typically at a single short horizon and
without per-regime breakdown. We make three observations: (1) aggregate RMSE on
quiet-dominated data hides storm-time performance — exactly when forecasts
matter; (2) persistence is a deceptively strong storm baseline because Dst is
autocorrelated; (3) operational value is not only point accuracy but
detection, uncertainty, and resilience to data outages. We build an honest,
multi-horizon, per-regime evaluation and reframe the contribution accordingly.

## 2. Data

MagNet/OMNI solar-wind dataset: 14 solar-wind variables (Bx/By/Bz/Bt, speed,
density, temperature, …) aggregated to hourly mean+std (28 features) plus
smoothed sunspot number = **29 paper features**. Sliding window of 48 h input →
6 h Dst target vector. Per-period chronological 70/20/10 train/val/test split;
StandardScaler fit on train only; held-out test never seen in training or early
stopping. Model inputs exclude Dst. We additionally recover the anonymized
MagNet absolute dates via Dst-fingerprint cross-correlation to merge OMNI
Kp/ap/AE indices (Section 6).

## 3. Model and training

Two-layer BiLSTM (hidden 128/direction, dropout 0.3, bidirectional) + attention
over the 48 timesteps → concat(context, last hidden) → linear → 6 Dst
predictions. Two-tier asymmetric weighted MSE (Dst<−20 → 5×, Dst<−50 → 15×) to
emphasize storm-time error. Adam (lr 1e-3, wd 1e-5), grad clip 1.0, early
stopping (patience 15) on validation loss.

## 4. Headline result: persistence wins on storms (Fig. 1, Fig. 2)

On the fixed held-out test (N=13974; intense N=387):

| Horizon | Aggregate RMSE (nT) | | | Intense RMSE (nT) | | |
|---|---|---|---|---|---|---|
| | model | persist | Burton | model | persist | Burton |
| t+1 | 9.82 | **4.13** | 14.65 | 35.49 | **12.24** | 42.12 |
| t+3 | 9.83 | **8.95** | 15.09 | 35.78 | **24.61** | 44.70 |
| t+6 | **11.26** | 12.71 | 17.14 | 44.64 | **31.43** | 56.47 |

Persistence beats the model on aggregate for t+1–t+5 and on **intense at every
horizon**. The model wins aggregate only at t+6 (quiet/moderate-driven). The
mechanism is Dst autocorrelation, which persistence exploits and the
solar-wind-only model cannot. The model does beat the Burton/OM2000 physics
baseline as both a forecaster and a perfect-driver hindcast (agg 11.26 vs
14.62), showing it captures coupling the analytic ODE misses — but that does not
overcome the autocorrelation advantage of persistence on storms.

## 5. Where the model adds value

### 5.1 Calibrated storm detection (Fig. 3)
A multi-task head (shared encoder + regression head + per-horizon P(Dst<−50)
classifier; joint loss = weighted MSE + 200·BCE) yields AUC 0.983, Brier 0.012,
ECE 0.003. At threshold τ=0.30 it achieves POD 0.713 / CSI 0.573 / HSS 0.721,
beating both persistence (0.695 / 0.533) and the deterministic model (0.669 /
0.519). Detection — not point RMSE — is the operational win, and it is
Dst-independent.

### 5.2 Severity-adaptive uncertainty (Fig. 4)
Mondrian conformal intervals binned by predicted severity give 95% coverage of
97.7% (quiet) / 91.7% (moderate) / 82.9–90.4% (intense, base/aug) versus only
64.3% marginal coverage on intense. Marginal interval width grows honestly with
lead time (40.0 → 44.5 nT @95%, t+1 → t+6), a property single-horizon UQ cannot
demonstrate.

### 5.3 Dst-independent resilience
Because the model never ingests Dst, it continues forecasting when Dst telemetry
is stale or absent — precisely when persistence fails. Robustness tests (Phase I)
show ≤1.1× degradation at 10% input dropout but a critical 2.05× intense
degradation under magnetometer outage, identifying magnetometer redundancy as
the operational priority.

## 6. Geomagnetic-index inputs (adopted accuracy lever)
Adding cadence-lagged Kp/ap/AE inputs (32 features) is the only modality that
improves the model under 5-seed paired testing (all positive): t+1 intense
−21% (+4.84 ± 1.53 nT), front-loaded and decaying to non-significant by t+6.
Leakage is controlled (input-window-only + cadence-lag) and verified by a smooth
lag-decay with no boundary cliff. Caveat: even with indices the model does not
beat persistence on intense when live Dst is available; the idx-model's value is
the Dst-stale regime, and it requires near-real-time indices (gain halves by ~3h
staleness).

## 7. Deployment system
We ship a routed real-time forecaster (`solar_dst_inference.py`) implementing the
findings: persistence for t+1–t+5 and t+6-intense when Dst is fresh; base model
for t+6-aggregate; idx-model t+1–t+3 / base t+4–t+6 when Dst is stale; multi-task
classifier alarm at τ=0.30; Mondrian severity-adaptive bands; and data-health
monitors for Dst staleness, index staleness, and magnetometer outage.

## 8. Negative results (ablation trail)
Larger storm sets, longer windows, transformer/Conv1D encoders, ensemble
stacking, loss reweighting/oversampling, physics-coupling features, curated
extreme storms, a SYM-H target, and CME+flare+F10.7 inputs were all tested and
rejected — none beat the base model on the held-out test. Interpretability
(integrated gradients) confirms the model already concentrates on Bz/Bt/speed and
on the most recent hours, explaining why hand-crafted coupling transforms are
redundant. The accuracy lever is exhausted for this dataset (raw SDO/AIA imagery,
the only untried modality, post-dates the 1998/2004 test periods).

## 9. A discrepancy, resolved
A blocked 10-fold CV (Phase E) reported intense base−persistence +5.12 nT
(Wilcoxon p=0.007), suggesting the model beats persistence on storms. This does
**not** transfer to the fixed held-out test, where persistence wins intense at
every horizon. The CV blocks are storm-rich, all-horizon-averaged, and use less
training data per fold — not directly comparable. We report the fixed-test result
as deploy-relevant and present the CV only as a caveated robustness check. We
verified base predictions reproduce the documented test numbers exactly (t+6 agg
11.26, intense 44.64), confirming the comparison.

## 10. Conclusion
The honest answer to "does our model beat persistence?" is: not on storm point
error, and not because the model is weak, but because Dst autocorrelation makes
persistence a hard baseline that a Dst-blind model cannot exceed. The right
contributions of a solar-wind-only system are Dst-independent forecasting,
calibrated storm detection, and adaptive uncertainty — measured per-horizon and
per-regime. We hope this reframing improves how space-weather ML reports
storm-time skill.

---
### Figures
- **Fig. 1** `figs/fig1_perhorizon_rmse.png` — per-horizon RMSE, aggregate vs intense (model / persistence / Burton).
- **Fig. 2** `figs/fig2_storm_conditional.png` — storm-conditional t+6 RMSE (model vs persistence).
- **Fig. 3** `figs/fig3_detection.png` — multi-task detection skill vs alarm threshold (τ=0.30 marked).
- **Fig. 4** `figs/fig4_uq.png` — conformal width vs horizon; Mondrian coverage by severity.

### Reproducibility
Figures: `_paper_figs.py` (reads `phase_n_verify.csv`, `burton_baseline.csv`,
`mtl_prob_threshold_sweep.csv`, `conformal_intervals.csv`,
`conformal_base_vs_aug_bins.csv`). Verification: `_phase_n_verify.py`.
Inference: `solar_dst_inference.py` (+ `_smoke_inference.py`).
