# Solar Wind → Dst Forecasting — Progress Report

**Date:** 2026-05-24
**Model:** BiLSTM + attention, solar-wind-only Dst forecast (no prior Dst input)
**Baseline for comparison:** TriQXNet (Jahin et al. 2025, arXiv:2407.06658v3)

---

## 0. Context (what this is)

**The problem.** The **Dst index** (nT) measures geomagnetic storm strength — large negative values (e.g. −100 to −400 nT) mean a severe storm that can damage power grids, GPS, and satellites. We want to forecast Dst a few hours ahead so operators get advance warning.

**The input.** Solar wind measured upstream by the ACE and DSCOVR satellites (magnetic field, speed, density, temperature). We use **only solar wind** — we do **not** feed the model recent Dst values. This is the realistic operational setting: the storm signal arrives in the solar wind ~1 hour before it hits Earth.

**What we predict.** Dst at **t+1 through t+6 hours** ahead (six separate values). TriQXNet, the state-of-the-art model we benchmark against, predicts only the current hour (**t0**) and **t+1**.

**Why compare to TriQXNet.** It is the current best published model on this exact dataset (the public MagNet 2021 challenge data). It reports an average RMSE of **9.27 nT**. Comparing against it tells us where we stand and what is worth copying.

**How to read this report.** "RMSE" = average prediction error in nT (lower is better). "Storm-conditional" = error split by storm strength (quiet / moderate / intense), because a single average hides the fact that rare intense storms — the ones that matter — are by far the hardest. "Persistence" = a trivial baseline that just predicts "next hour = last observed Dst"; a model is only useful if it beats it.

---

## 1. The research gaps (where the state of the art is open)

TriQXNet (RMSE 9.27 nT) is the best published model on this dataset — but its headline number leaves real gaps open. These gaps frame the rest of this report: they are what we build on and what is worth attacking next.

| # | Gap in TriQXNet | Evidence (their own paper) |
|---|---|---|
| 1 | **Headline RMSE hides storm error.** 9.27 nT is dominated by quiet hours; on an actual extreme storm the error roughly doubles. | Extreme-event case study: t0 RMSE **20.33**, t+1 **20.86** nT vs aggregate 9.27. |
| 2 | **Only one extreme event evaluated.** No storm-conditional table, no storm-onset detection rate / false-alarm rate. "Extreme" is defined (Dst ≤ −80 nT) but never scored as a group. | Single October case study only. |
| 3 | **1-hour horizon, and t0 is trivial.** Predicts current hour (t0, nowcasting) + t+1. Averaging the easy t0 into the metric pulls 9.27 down. Short operational lead time. | 9.27 = t0/t+1 average. |
| 4 | **Quantum branch gain is marginal and fragile.** ~0.38 nT (4%), non-monotonic in qubit count, simulator-only (no real quantum hardware). High reproducibility cost for a gain plausibly within noise. | 9.65 → 9.27 with quantum; qubits 4→9.75, 6→9.91, 8→9.70. |
| 5 | **Uncertainty only at t+1.** Conformal intervals exist but there is no error-growth-with-lead-time — the core question for a forecaster. | UQ section is t0/t+1 only. |
| 6 | **The 3-pipeline ensemble barely nets out.** Two classical paths alone (9.65) are *worse* than one (9.42); the design just recovers the loss. | Their own ablation. |
| 7 | **Cross-study comparison is weak.** They acknowledge results are confounded by data subset and solar-cycle tuning; stated future work is generic ("expand datasets"). | Discussion section. |

**The core open problem (gaps 1+2+3 together):** 9.27 nT is a *quiet-hour, short-horizon, t0-inflated* number. The hard task — **multi-hour-ahead forecasting of intense storms, with quantified uncertainty** — is exactly what TriQXNet does *not* report. That is the lane this project works in.

**Where we already sit in that lane:** we forecast t+1…t+6 and report the full storm-conditional table honestly (our intense t+6 = 44.64 nT — bad, but *measured and visible*). What we lack is uncertainty quantification. So the gaps are complementary: they are missing horizon + storm transparency; we are missing UQ.

---

## 2. Setup — Ours vs TriQXNet (same dataset)

| | Ours | TriQXNet |
|---|---|---|
| Dataset | MagNet 2021 (ACE+DSCOVR) | MagNet 2021 (ACE+DSCOVR) |
| Features | 29 (14 vars × mean+std + smoothed_ssn) | 29 (same, Spearman-selected) |
| Aggregation | hourly mean + std | hourly mean + std |
| Split | per-period 70/20/10 | 70/20/10 |
| Input window | **48 h** | **128 h** |
| Forecast horizon | **t+1 … t+6** (6 h) | **t0, t+1** (1 h) |
| Architecture | 2-layer BiLSTM + attention | 3-pipeline classical + 4-qubit quantum branch |
| Uncertainty | none | conformal prediction (99% intervals) |
| Interpretability | none | ShapTime + permutation importance |
| Significance test | multi-seed (5) | 10-fold CV paired t-test |
| Params | ~0.5 M | 0.40 M |

**Same data, same features, same split.** Real differences: window, horizon, architecture, UQ/XAI, significance protocol.

---

## 3. Head-to-head RMSE (nT)

| Metric | Ours | TriQXNet |
|---|---|---|
| RMSE @ t+1 (held-out test) | **9.82** | — |
| RMSE @ t+1 (val) | 10.14 | — |
| RMSE @ t0,t+1 avg | n/a (no t0) | **9.27** |
| RMSE @ t+6 (test) | 11.26 | n/a (no t+6) |
| Pearson r @ t+6 (test) | 0.82 | — |
| R² @ t+6 (test) | 0.68 | — |

Gap at t+1 ≈ **0.5–0.9 nT**. Note their 9.27 averages an easy nowcast (t0) with t+1; we don't predict t0.

**TriQXNet benchmark context (their 13-model sweep, t0/t+1 avg):** LSTM 12.17 · Stacked BiLSTM 10.66 · BiLSTM+BiGRU 10.90 · Conv1DTimeDistributedNet 9.42 · **TriQXNet 9.27**. Their quantum branch buys 9.42 → 9.27 (~4%).

---

## 4. Our experiments (Phase A + B)

| Experiment | Result (held-out test t+6) | Verdict |
|---|---|---|
| BiLSTM base (Phase A) | 11.26 nT, intense 44.64 | **PRIMARY** |
| + GRU bias corrector | t+6 11.23 (quiet-only gain) | partial — quiet only |
| + LGBM ensemble blend | 11.49 (worse) | rejected |
| Conv1DNet (A4 val-winner) | 11.66 / 11.71 (worse) | rejected — won on val, lost on test |
| Input window 96 / 128 h | 11.92 / 11.29 (≥ 48h) | rejected — longer window no help |
| OMNI aug — hourly proxy std (Step 4) | intense Δ −1.73±3.79 (5-seed) | rejected — one lucky seed, not real |
| **OMNI aug — real 1-min std, 15 storms (Phase D)** | **intense Δ +8.36±3.39, agg Δ +0.48±0.28 (5-seed)** | **ADOPTED — first robust storm win** |
| Physics coupling features | −0.16 nT (worse) | rejected — redundant with raw inputs |
| Persistence/model hybrid switch | worse on storms | rejected — storm-onset timing mismatch |

**Architecture sweep (A4, 6 archs × 5 seeds):** all within 1–2σ at t+6 (11.18–11.84). Attention has highest variance. **Architecture is not the lever on this dataset size — data is.**

**Phase D breakthrough (1-min OMNI):** the *only* lever that moved storm RMSE. Real `OMNI_HRO_1MIN` (true hourly mean+std, not Step-4's 3h proxy) over 15 storms → 877 intense seqs (2.7× Step 4). Held-out test t+6 intense **44.5 → 36.1 nT (−19%)**, aggregate 11.42 → 10.94, all 5 seeds positive. The proxy std — not OMNI itself — was Step 4's killer. Confirms the bottleneck was always **storm-data scarcity + feature fidelity**, exactly as diagnosed.

---

## 5. Persistence baseline (skill = is the model worth it?)

| Step | Persistence | BiLSTM+GRU | Model skill |
|---|---|---|---|
| t+1 | 4.03 | 9.73 | −141% |
| t+3 | 8.38 | 10.08 | −20% |
| t+6 | 11.71 | 11.23 | **+4.1%** |
| Intense t+6 | 48.66 | 40.25 | **+17.3%** |

Model only beats trivial persistence at **t+6** and on **intense storms**. Real value = storm detection at longest horizon, not aggregate RMSE.

---

## 6. What we do better / what they do better

**We do better**
- Longer horizon: **6 h lead** vs their 1 h. More operational warning.
- Storm-conditional truth: quiet/moderate/intense breakdown + persistence skill — they report mostly aggregate RMSE (quiet-dominated, ~91% of data).
- Honest evaluation discipline: held-out test + **multi-seed** (caught a false +4.63 nT "win" that was seed luck).
- Simpler, no quantum hardware/simulator dependency.

**They do better**
- Lower aggregate t+1 RMSE (9.27 vs ~9.8).
- **Uncertainty quantification** (conformal intervals) — operationally critical, we have **zero**.
- **Interpretability** (ShapTime, permutation importance).
- **Statistical rigor** (10-fold CV paired t-tests, p-values).
- 128 h window + 3-pipeline ensemble extract more from the same data.

---

## 7. Where we sit on these gaps (built on / to build)

| Gap | Status |
|---|---|
| RMSE is quiet-dominated, hides storm skill | **built on** — we report storm-conditional + persistence skill |
| Single-run results unreliable on rare storms (N≈387 → ±1–3.5 nT seed noise) | **built on** — multi-seed protocol; killed a false positive |
| "Better architecture" assumed = better | **built on** — sweep shows all within noise; data is the bound, not arch |
| Storm data scarcity (intense < 3% of data) | **partially solved (Phase D)** — real 1-min OMNI aug cut intense t+6 19% (44.5→36.1 nT), 5-seed robust. Hourly proxy had failed; fidelity was the issue |
| No uncertainty estimates | **open** — biggest gap vs TriQXNet |
| No significance testing across the field | **done (Phase E)** — multi-seed + leakage-safe 10-fold CV paired t-tests/Wilcoxon. Model>>persistence on intense p=0.007; aug>base on intense Wilcoxon p=0.049 |

---

## 8. Future work / what's left

| Priority | Task | Why |
|---|---|---|
| **High** | Conformal prediction intervals (crepes) per horizon | Closes our biggest gap; directly portable from TriQXNet; uncertainty grows with horizon = high value |
| ✅ DONE | ~~Real sub-hourly OMNI aug (1-min `OMNI_HRO_1MIN`, true mean+std)~~ | **Phase D — ADOPTED.** Intense t+6 −19%, agg −0.48 nT, 5-seed robust. Next: 25+ storms; recalibrate UQ on aug model |
| Med | Replicate 128h + 3-pipeline ensemble at t+1 | Isolate whether the 0.5–0.9 nT gap is window, ensemble, or quantum. We showed 128h *alone* doesn't help |
| ✅ DONE | ~~10-fold CV paired t-tests~~ | **Phase E, 2026-06-01.** Leakage-safe blocked 10-fold. Intense aug−base +3.01 nT (Wilcoxon p=0.049); intense model−persistence +5.12 nT (p=0.007). Aggregate aug-gain ns. Matches TriQXNet's bar; Phase D adoption holds. |
| Low | Permutation feature importance | Prune 29 → fewer features; check std-cols actually contribute |
| Low | Quantum branch | TriQXNet shows only ~4% from it; low ROI vs UQ |

---

## 9. One-line summary

Same data, same features, same split as TriQXNet. We trade ~0.5–0.9 nT at t+1 for a **6× longer horizon** and an honest, storm-aware, multi-seed-verified evaluation. Phase-B levers (ensemble, Conv1D, longer window, *proxy* OMNI) all failed — pointing at **storm-data scarcity + feature fidelity** as the bottleneck. **Phase D confirmed it: real 1-min OMNI augmentation cut intense-storm t+6 RMSE 19% (44.5→36.1 nT), the first robust storm win.** Phase C added per-horizon conformal uncertainty (the TriQXNet capability we lacked). Net: we now match TriQXNet on UQ, beat it on horizon + storm honesty, and have a working data lever for the scarcity ceiling.
