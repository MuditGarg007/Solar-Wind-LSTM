# Phase J — Kp/ap/AE geomag-index INPUTS (next-step #4) — ✅ COMPLETE & ADOPTED (2026-06-13)

## Verdict
First new external-data modality to beat base on held-out test. Multi-seed win, leakage VERIFIED real.
- 5-seed paired Δ vs base: **t+1 intense +4.84±1.53 (−21%, t-p .002)**, all-horiz intense +2.80±1.17, agg +0.55±0.14, all 5 seeds positive.
- Gain front-loaded (fades to ns by t+6) → owns t+1–t+3 storm warning; complements OMNI-aug (t+6).
- Leak test: t+1 intense gain decays SMOOTHLY with extra index staleness (+0h+7.5→+1h+5.9→+3h+4.1→+6h~0), no cliff → real magnetosphere-autocorr, not label-leak.

## Shipped
- Checkpoint `solar_bilstm_idx_model.pth` + `idx_scaler.pkl` (32 feat, seed42; test agg 9.83 / intense 36.70 / t+1 intense 28.01).
- Notebook cells `phase_j_build` + `phase_j_results` (EXECUTION PENDING convention, results embedded).
- CLAUDE.md updated: adopted-models, deploy note, STATUS (Phase J record), next-step #4 → DONE, Key cells.
- CSVs `geomag_index_ab.csv`, `geomag_index_lag_decay.csv`. Dates `_magnet_dates.json`.

## Reproduce (env dnn, all standalone)
```
_date_recovery_full.py  -> _magnet_dates.json
_fetch_indices.py       -> _omni_indices_1998_2019.pkl
_build_features_indices.py -> _seq_base.npz / _seq_aug.npz
_ab_train_indices.py    -> geomag_index_ab.csv   (note: replace Unicode Δ print if rerun)
_lag_sweep.py / _leak_check.py -> lag-decay leak test
_finalize_phase_j.py    -> solar_bilstm_idx_model.pth / idx_scaler.pkl / geomag_index_lag_decay.csv
```

## Deploy caveat
Needs near-real-time AE/Kp/ap (quicklook); gain halves by ~3h index staleness, gone by ~6h.

## Next open levers (unchanged)
SHAP (#5, cheap), SYM-H target, solar imagery. Or extend Phase J: SYM-H/ASY indices, latency-robust +6h-lag variant.
