import json, shutil
NB='SolarWindLSTM.ipynb'
shutil.copy(NB, NB+'.bak_phasel')
nb=json.load(open(NB,encoding='utf-8'))
md="""### Phase L results (bucket-2 SYM-H higher-cadence target — TESTED & REJECTED)

**Hypothesis:** OMNI hourly-**min** SYM-H captures within-hour storm peaks Dst smooths away → more extreme-storm labels in the data-scarce regime (Phase G under-forecast). Standalone (env `dnn`): `_symh_probe.py` (feasibility) → `_build_symh.py` (paired seqs, train_a/b/c, identical windows, both targets) → `_symh_train.py` (5-seed paired vs Dst-control). Leakage-safe: SYM-H from OMNI on Phase-J recovered dates, target-only (no input leak). Artifacts `symh_perhorizon.csv`/`symh_stormcond.csv`/`symh_detection.csv`, `solar_bilstm_symh_model.pth` (ablation, **don't promote**).

**Probe (16yr):** hourly-**mean** SYM-H ≈ Dst (corr 0.945, RMSE 6.23) → relabel trivial. hourly-**min** exposes +16–30% more extreme-storm hours (grows at extremes) → tested as target.

**5-seed paired result (held-out test, each arm on own target):**

| metric | Dst-control | SYM-H-min |
|--|--|--|
| RMSE t+1 / t+6 | 9.89 / 11.47 | 10.65 / 12.77 |
| intense RMSE | 38.54 | 43.99 |
| POD @−80 / −100 | 0.648 / 0.552 | **0.716 / 0.620** |
| FAR @−80 / −100 | 0.187 / 0.034 | 0.319 / 0.190 |
| CSI @−80 / −100 | **0.561 / 0.540** | 0.531 / 0.536 |
| BIAS @−80 / −100 | 0.801 / 0.574 | 1.062 / 0.780 |

**Verdict — REJECT.** RMSE uniformly worse (no seed overlap). SYM-H-min *does* raise extreme POD and fix the under-forecast BIAS (→near-1.0), but FAR blows up → net CSI tie-to-worse. The POD/BIAS gain is just a lower effective alarm threshold baked into the target — **Phase H's τ-sweep already buys the same POD/FAR trade on the existing Dst model with no new data.** SYM-H dominated by a cheaper existing solution. Bucket-2 SYM-H sub-option closed (DO-NOT-REPEAT #11)."""
nb['cells'].append({'cell_type':'markdown','id':'phase_l_results','metadata':{},'source':md.splitlines(keepends=True)})
json.dump(nb,open(NB,'w',encoding='utf-8'),indent=1)
print('appended phase_l_results; cells',len(nb['cells']))
