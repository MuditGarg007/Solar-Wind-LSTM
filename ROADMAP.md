# Research Roadmap — Dst Forecasting from Solar Wind

**Anchor paper:** Jahin et al., *TriQXNet: Forecasting Dst Index from Solar Wind Data Using an Interpretable Parallel Classical–Quantum Framework with Uncertainty Quantification* (arXiv 2407.06658v3, Oct 2025).

**Goal:** convert this repository from a single-model BiLSTM experiment into a publishable, paper-grade study that is competitive with — and differentiable from — TriQXNet on the same MagNet dataset.

---

## 0. Where we stand vs. TriQXNet

| Axis | Our project (current best) | TriQXNet | Gap |
|---|---|---|---|
| Forecast horizon | t+6 (6 hours ahead) | t0 and t+1 (now + 1 hour) | **Different task** — ours is the harder problem |
| RMSE | 8.45 nT (BiLSTM, t+6 full val) / 7.78 nT (BiLSTM+GRU, t+6) / 8.32 nT (combined ensemble, last 20% val) | 9.27 nT (t0/t+1) | Not directly comparable until we evaluate at t+1 |
| Input window | 48 h | 128 h | Theirs is 2.7× longer |
| Features | 15 (raw + rolling means + `dyn_pressure`) | 29 (mean + std for each variable, plus `smoothed_ssn`) | We lack hourly std features and the sunspot signal |
| Architectures benchmarked | 1 (BiLSTM+attention), ensembled with GRU and LGBM | 13 + proposed TriQXNet, with ablations | We have no comparative table |
| Statistical validation | None (single train/val split) | 10-fold CV paired t-tests | None |
| Uncertainty quantification | None | Conformal prediction (4 variants) | Missing |
| Interpretability | None | ShapTime + Permutation Feature Importance | Missing |
| Dataset coverage | Periods in `archive/` (train_a/b/c) | All 6 MagNet periods, 1998–2020, 11.94 M raw records | We may be using a subset |
| Storm-focused metrics | Storm-conditional RMSE (Quiet/Mod/Intense) + persistence baseline | Single-case study (Oct 2001) | We are stronger here |

**The good news.** Three things in our project that the paper does *not* do, and that are publishable in their own right:

1. **6-hour-ahead forecasting** (paper only does t0/t+1). This is the harder, more operationally useful horizon.
2. **Storm-conditional evaluation** with persistence skill scores. We have already documented that our model achieves **+29.3 % skill over persistence on intense storms at t+6**.
3. **Negative-result discipline.** We have rigorously documented three failed storm-bias fixes (linear recalibration, directional asymmetric loss, oversampling) — these are publishable as a "what does not work and why" section.

**The bad news.** We do not have a benchmark, we do not have uncertainty intervals, we do not have XAI, we do not have statistical tests, and we have used only a fraction of the MagNet dataset. Without those, the work is a hobby project, not a paper.

---

## Phase A — Parity (close the methodology gap with the paper)

These steps make our work directly comparable to the literature. They are not "improvements" in the modelling sense; they are the cost of admission to a peer-reviewed venue.

### A1. Load the full MagNet dataset
**Why:** TriQXNet uses all 6 official MagNet periods (Table 8 of the paper: 1998–2020, ~270 k hourly Dst samples after aggregation). Our `archive/` is the same source but we are only training on `train_a`/`train_b`/`train_c`. Comparable results require the same data.
- Download `solar_wind.csv`, `labels.csv`, `sunspots.csv`, and `satellite_positions.csv` directly from the MagNet challenge mirror (or via the Zenodo link in the paper's code release: `10.5281/zenodo.12694950`).
- Confirm the period boundaries match Table 8.
- Use the paper's 70/20/10 split per period (we currently use 80/20 train/val with no held-out test). This matters: our reported numbers right now are validation, not test.

### A2. Match the feature set (15 → 29)
**Why:** the paper aggregates *each* hour to **mean and standard deviation** of every solar-wind variable, plus `smoothed_ssn` (monthly smoothed sunspot number). Our pipeline only computes hourly means. PFI in Fig. 8 of the paper ranks `bt_std`, `theta_gsm_std`, `density_std`, `bz_gsm_std` all in the top 15 — we are throwing away half the signal.
- In `load_and_preprocess`, group by `(period, floor('1h'))` and compute both `.mean()` and `.std()` for: `bx_gse, by_gse, bz_gse, bx_gsm, by_gsm, bz_gsm, theta_gse, phi_gse, theta_gsm, phi_gsm, bt, density, speed, temperature` (14 vars × 2 = 28 features).
- Add `smoothed_ssn` (forward-fill monthly values to hourly).
- Total: 29 features (matches paper Table 9).
- Retain our derived features (`bz_3h`, `dyn_pressure`, etc.) in a *separate* feature-set variant so we can ablate paper-features vs. paper-features + ours.
- **Expected effect:** modest. Std features encode turbulence (a known storm precursor) so I expect 0.2–0.5 nT improvement at t+6.

### A3. Reproduce paper's evaluation protocol
**Why:** the paper reports a single number (RMSE) averaged across t0 and t+1, and trains for both heads simultaneously. To benchmark against it we must match this.
- Add a second output head trained on the t+1 target (in addition to our existing t+1…t+6 vector).
- Report both: (a) t0/t+1 RMSE for direct comparison to Table 1 of the paper, and (b) our existing t+1…t+6 per-step table as our differentiated contribution.

### A4. Build the benchmark table
**Why:** the paper compares 13 hybrid architectures. Without a benchmark our 8.45 nT number is meaningless to a reviewer.
- Re-implement at least 5 of the paper's baselines on our pipeline: `LSTM`, `Stacked BiLSTM`, `BiLSTM+BiGRU`, `CNN+BiLSTM+TimeDistributedDense`, `Conv1DTimeDistributedNet`. The paper's "Methods" section has full architectural specs (Equations 3–30) — direct port.
- Plus our `SolarAttentionLSTM` (current model).
- Train each with identical hyperparameters (BATCH_SIZE, EPOCHS, optimizer) and identical splits.
- Report mean ± std across 5 random seeds per architecture.
- Output: a Table-1 equivalent for our t+6 forecasting task.

---

## Phase B — Novel contributions (what makes it a paper)

These are the items where we can plausibly beat or differentiate from TriQXNet. They are ordered by realism, easiest first.

### B1. Storm-conditional benchmarking framework
**Why:** the paper's case study (Oct 2001) is anecdotal — a single event. Our project already has the storm-conditional RMSE bins (Quiet ≥ −20, Moderate −50 to −20, Intense < −50) and persistence skill scores. This is genuinely missing from the paper.
- Formalise the persistence skill metric:
  `SS = 1 − (RMSE_model / RMSE_persistence)²`
- Report SS per storm bin per step (Table format: rows = bins, columns = t+1…t+6).
- This is a publishable methodological contribution: *"reporting aggregate RMSE on Dst forecasting masks a 30 % skill on intense storms behind a 1 % skill on quiet times, because intense storms are ~1.2 % of the sample. We propose persistence-skill per storm-bin as the standard reporting protocol."*

### B2. Horizon-specific hybrid forecasting
**Why:** our notebook already documented the finding that *persistence beats the BiLSTM at t+1…t+5 and only loses at t+6*. The natural conclusion — "output persistence for t+1…t+5 and the model for t+6" — is genuinely novel and operationally useful.
- Formalise as a "horizon-routed hybrid": for each output step, select the predictor (persistence vs. model) that has lower validation RMSE on that step. No learned router; pure rule.
- Compare against three baselines: pure persistence, pure model, the failed-experiment threshold hybrid.
- **Caveat:** this is a *negative* result for deep learning at short horizons. Framing it as "Dst at short horizons is autocorrelation-limited, not solar-wind-limited" makes it publishable; framing it as "our model is bad at t+1" does not. Use the first frame.

### B3. OMNI storm augmentation (already in CLAUDE.md plan)
**Why:** every fix for storm bias we have tried has failed because of data scarcity (~145 intense samples). Augmenting with historic super-storms from NASA OMNI is the only untried lever.
- Execute the three-cell plan already in `CLAUDE.md` ("Next Session: OMNI Data Augmentation").
- Add at least 5 storms (Halloween 2003, Bastille 2000, St. Patrick 2015, Sep 2017, Aug 2018).
- Critical evaluation requirement: val set must remain base-only — OMNI goes into training only. This is non-negotiable for a fair number.
- **Expected effect:** unknown. If intense-storm RMSE improves by >2 nT on the held-out base val, this is the strongest result in the paper.

### B4. Conformal prediction for uncertainty quantification
**Why:** the paper uses the `crepes` library for conformal regressors. This is now a standard technique and is missing from our work. Reviewers ask for it.
- After the best model is trained, fit a `ConformalRegressor` from `crepes` on calibration residuals.
- Report PICP (Prediction Interval Coverage Probability) and MPIW (Mean Prediction Interval Width) at 95 % confidence.
- Storm-conditional intervals: do intervals *widen* during storms? This is the right thing for an operational system to do.
- Add a Mondrian conformal regressor binned by `last_dst` so intervals adapt to the storm regime.
- This is ~50 lines of code on top of the existing `aug_preds`/`aug_actuals`. High return on effort.

### B5. Explainability (ShapTime + PFI)
**Why:** the paper uses both. PFI in particular is trivial to add and is a reviewer-comfort item.
- **Permutation feature importance:** shuffle each of the 29 features in the val set, measure RMSE delta. Rank. Compare ranking against Fig. 8 of the paper. *Hypothesis: `bz_gsm_mean` will be #1 in our model too. If it is not, that is itself a publishable finding.*
- **SHAP for the LSTM:** use `shap.DeepExplainer` on a subset of 500 val sequences. Report mean |SHAP| per feature and per timestep within the 48 h window. *Look for whether the model attends most to the most recent 6 h (consistent with paper's t9 finding in their 10-supertime decomposition) or earlier — this is a real scientific result about Dst dynamics.*
- **Attention weight analysis:** we already have an attention layer. Plot mean attention across the 48 timesteps and over storm vs. quiet samples. *Storm samples should attend later in the window if our model has learned correctly.*

### B6. Statistical validation
**Why:** the paper uses 10-fold cross-validated paired t-tests. Without this our "BiLSTM beats LSTM by 0.5 nT" is statistically empty.
- Implement k-fold CV across the 6 periods (group-aware, not random). Each fold leaves one period out as test.
- For each pair (our model vs. each baseline), compute paired t-test on per-fold RMSE.
- Report t-statistic and p-value in the benchmark table.
- 5 folds is enough if 10 is too slow.

### B7. Custom storm-sensitive loss (revisit, but smaller)
**Why:** the paper's DeepSeqConvNet uses `log((y−s)²)^s + |y−s|` (Equation 11) where s controls outlier sensitivity. We tried weighted MSE with mixed results. Worth trying their formulation since they report it works for them.
- Replace our current weighted MSE with the paper's parametric loss, search over s ∈ {1.5, 2.0, 2.5, 3.0}.
- **Be honest in evaluation:** report quiet-time RMSE *and* storm RMSE. We already learned from the failed asymmetric loss experiment that storm-only metrics can mislead.
- **Caveat:** our previous storm-loss work concluded that loss reweighting is bounded by data scarcity. This step is worth doing because the paper's specific formulation is different (log-based) and may behave differently — but I would not bet on a 1 nT win.

---

## Phase C — Moonshots (high risk, big-paper potential)

### C1. Quantum branch (TriQXNet replication)
**Why:** the paper's defining innovation is a 4-qubit dressed quantum circuit using Pennylane + amplitude embedding + strongly entangling layers. Replicating this gives us a head-to-head comparison and lets us answer the question *"does the quantum branch actually help at t+6?"* — the paper only tests at t+1, where its 0.38 nT gain may or may not generalise to longer horizons.
- Install `pennylane` and `pennylane-qiskit`.
- Build the 4-qubit, 2-layer dressed circuit from the paper's Methods (Equations 24–30, Figure 13).
- Wrap in a `KerasLayer` / `TorchLayer` and slot into a parallel pipeline alongside our BiLSTM.
- Train end-to-end. **Expect 10–100× slower training than classical.**
- Run an ablation: classical-only, classical+quantum, quantum-only. Match the paper's Table 2.
- **Honest expectation:** there is real debate in the QML literature about whether dressed quantum circuits beat classical baselines at scale. The paper claims 0.38 nT improvement — small enough to be within seed noise. We should report this finding straight, including the possibility that the result does not replicate.

### C2. Architectural sweeps the paper did not do
- **Temporal Fusion Transformer (TFT)** — designed for multi-horizon forecasting with attention over both time and features. The paper has nothing like this. If a TFT beats their best classical pipeline at t+6 by even 0.3 nT, that is a strong story.
- **PatchTST / iTransformer** (2023 SOTA time-series transformers) — patch-based sequence models that outperform classical LSTM stacks on many forecasting benchmarks. Worth one architecture variant.
- **Neural ODE / DeepONet** — physics-informed sequence models. A long shot but methodologically novel.

### C3. Multi-task learning
**Why:** the paper predicts (t0, t+1) jointly. We predict (t+1…t+6) jointly. What about *also* predicting derived quantities (Dst derivative, AE index if available, Kp) as auxiliary tasks? Auxiliary task regularisation is a known way to improve under data scarcity.
- Add Kp index as an auxiliary output (it is available from the same SWPC source, hourly).
- Share the LSTM trunk; separate heads for Dst (6 steps) and Kp (1 or 6 steps).
- Loss is a weighted sum.
- **Expected effect:** small but possibly real. Worth one experiment.

### C4. Operational deployment demo
**Why:** the paper claims operational readiness but does not demonstrate it. We could.
- Build a Streamlit/Gradio app that pulls live RTSW data from NOAA, runs the model, and displays a 6-hour Dst forecast with 95 % conformal intervals.
- Daily cron job for 90 days, log live forecasts vs. realised Dst.
- Live RMSE on real operational data is a uniquely strong claim that the paper cannot make.

---

## Phase D — Paper assembly

### D1. Title (proposed)
*"Six-Hour-Ahead Dst Forecasting from Solar Wind: A Storm-Conditional Benchmark and Hybrid LSTM-Persistence Forecaster with Conformal Uncertainty"*

The keyword "storm-conditional" is the differentiator. "Six-hour-ahead" sets us apart from TriQXNet's t+1. "Conformal" signals modern uncertainty handling. "Hybrid LSTM-Persistence" is the actual operational contribution.

### D2. Section structure
1. Introduction — geomagnetic storms, operational gap at multi-hour horizons, prior work emphasising t+1
2. Related Work — MagNet ensemble (Nair 2023), TriQXNet (Jahin 2025), Gruet (2018), Xu (2020), Hu (2023)
3. Data — MagNet dataset, OMNI augmentation, storm distribution, train/val/test protocol
4. Methods — feature engineering, BiLSTM+attention, GRU residual, LGBM stack, horizon-routed hybrid, conformal calibration
5. Results — main benchmark table, per-step RMSE, storm-conditional skill scores, conformal interval calibration
6. Analysis — PFI, attention visualisation, negative results (the three failed storm fixes)
7. Discussion — why short horizons are persistence-dominated; the data-scarcity ceiling for storm prediction; deployment considerations
8. Conclusion

### D3. Target venues (realistic)
- **Space Weather** (AGU) — primary fit. Practitioner-oriented, accepts ML-on-space-weather work routinely. Impact factor 4.0.
- **Journal of Space Weather and Space Climate** — open access, European, similar scope.
- **arXiv + workshop** — NeurIPS Climate-Change-AI workshop, ICLR Tackling Climate Change workshop. Lower bar but real visibility.
- Less likely but possible if the quantum or TFT angle works: *Scientific Reports* (where TriQXNet was published).

---

## Honest realism check

What we have right now is a good engineering project but not yet a paper. The shortest path to a paper is:

- Phase A (data, features, baselines, evaluation protocol) — **2–3 focused weekends**. No research risk. Pure execution.
- Phase B1, B2, B4, B5 (storm framework, hybrid persistence, conformal, PFI) — **2–3 weeks**. Each is small, each is publishable.
- Phase B3 (OMNI augmentation) — already specced. **1 weekend.** This is the only step with real upside on the *modelling* numbers.
- That alone is a paper. Phases C are optional unless we have specific reach goals.

The phases marked C carry meaningful research risk and significant time cost (the quantum branch in particular can easily eat a month). Save them for the v2 paper.

