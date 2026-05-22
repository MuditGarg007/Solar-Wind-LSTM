# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a deep learning project that predicts the **Dst (Disturbance Storm Time) geomagnetic index** 6 hours into the future using solar wind data. It serves as an early warning system for geomagnetic storms.

## Running the Pipeline

All code lives in `SolarWindLSTM.ipynb`. Run it cell-by-cell in Jupyter:

```bash
jupyter notebook SolarWindLSTM.ipynb
```

Or execute the full pipeline non-interactively:

```bash
jupyter nbconvert --to notebook --execute SolarWindLSTM.ipynb
```

The saved model weights are written to `solar_lstm_model.pth` automatically when validation loss improves.

## Data

Raw data lives in `archive/`:
- `solar_wind.csv` — solar wind parameters (Bx, By, Bz, Bt, speed, density, temperature) at sub-hourly resolution
- `labels.csv` — Dst index values
- Both files use a `(period, timedelta)` multi-index; periods are named event windows (e.g., `train_a`, `train_b`)

## Architecture & Pipeline

**Data pipeline** (`load_and_preprocess` → `create_sequences`):
1. Load CSVs, set `(period, timedelta)` multi-index
2. Resample to hourly means per period
3. Engineer 15 features: `energy = speed * bz_gse`; rolling means for `bz_gse` and `speed` at 3h/6h/12h windows; `dyn_pressure = density * speed²`
4. Create sliding windows: 48-hour input → 6-hour target vector
5. Chronological 80/20 split; `StandardScaler` fit only on training data

**Model** (`SolarAttentionLSTM`):
- 2-layer BiLSTM (hidden=128 per direction, dropout=0.3, bidirectional=True)
- Custom attention layer that weights each of the 48 input timesteps
- Output: concatenation of attention context vector + last LSTM hidden state → linear → 6 Dst predictions

**Training**:
- Two-tier asymmetric weighted MSE: moderate storms (Dst < −20 nT) 5×, intense storms (Dst < −50 nT) 15×
- Adam optimizer, lr=0.001, weight_decay=1e-5
- Gradient clipping max_norm=1.0
- Early stopping with patience=15 on validation loss

## Key Hyperparameters

```python
SEQ_LEN = 48          # Input window (hours)
FORECAST_HORIZON = 6  # Prediction horizon (hours)
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
BIDIRECTIONAL = True
BATCH_SIZE = 64
LEARNING_RATE = 0.001
TRAIN_SPLIT = 0.8
EPOCHS = 50
```

## Performance (current best model)

**BiLSTM + GRU Correction (full val set, t+6):**
- RMSE: 7.7751 nT
- Pearson Correlation: 0.7201 (BiLSTM base)
- R²: 0.4141 (BiLSTM base)

**BiLSTM base (Phase 2, full val set, t+6):**
- RMSE: 8.4531 nT
- Pearson Correlation: 0.7201
- R²: 0.4141

---

## Improvement Roadmap

Based on research into MagNet competition winners (RMSE 10.6 nT ensemble on the same dataset) and Hu et al. 2023 (best published 6-hour model: 13.54 nT), here is the planned upgrade path. Implement phases in order.

### ✅ Phase 1 — Fair Evaluation (no architecture change) — DONE

Current metrics only report t+6. Fix the inference loop to collect all 6 steps:

```python
all_preds   = np.vstack(...)  # (N_val, 6)
all_actuals = np.vstack(...)  # (N_val, 6)
```

Add two new cells after the existing eval cell:
- **Per-step RMSE table:** print RMSE / Pearson r / R² for t+1 through t+6, save `per_step_rmse.png`
- **Storm-conditional RMSE table:** three bins on `all_actuals[:, -1]`:
  - Quiet: Dst ≥ −20 nT
  - Moderate: −50 ≤ Dst < −20 nT
  - Intense: Dst < −50 nT

### Phase 2 — Architecture Upgrade

**Config changes (Cell 2):**
```python
SEQ_LEN       = 48    # 24 → 48h (top models use 48–72h)
HIDDEN_DIM    = 128   # 64 → 128
DROPOUT       = 0.3   # 0.4 → 0.3 (larger model needs less regularisation)
BIDIRECTIONAL = True  # new flag
```

**5 new engineered features (Cell 3 — `load_and_preprocess`)**, after current `bz_3h`/`speed_3h`:
```python
df_solar_hourly['bz_6h']        = ...rolling(6,  min_periods=1).mean()
df_solar_hourly['speed_6h']     = ...rolling(6,  min_periods=1).mean()
df_solar_hourly['bz_12h']       = ...rolling(12, min_periods=1).mean()
df_solar_hourly['speed_12h']    = ...rolling(12, min_periods=1).mean()
df_solar_hourly['dyn_pressure'] = density * speed**2   # ram pressure proxy
```
Feature count: 10 → 15. `input_dim` is read from `X_train_scaled.shape[2]` dynamically.

**BiLSTM model (Cell 6 — `SolarAttentionLSTM`):**
Add `bidirectional` parameter; `D = 2 if bidirectional else 1`. Update:
- `attention_fc`: `Linear(hidden_dim * D, 1)`
- `fc`: `Linear(hidden_dim * D * 2, output_dim)`

**Training tweaks (Cell 7):**
- Pass `bidirectional=BIDIRECTIONAL` to model constructor
- Add `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` after `loss.backward()`
- Early stopping patience: 10 → 15
- Save to `solar_bilstm_model.pth`

*Expected gain: t+6 RMSE 9.54 → ~7.5–8.5 nT*
*Risk: if GPU OOM, drop BATCH_SIZE 64→32 first, then HIDDEN_DIM 128→96*

### Phase 3 — LightGBM Ensemble (MagNet winner strategy)

Stack a gradient-boosted tree on top of LSTM predictions (the key differentiator of MagNet's top finishers).

New cells after training:
1. `pip install lightgbm` inline + collect LSTM val predictions `(N_val, 6)`
2. Build meta-feature matrix: 6 LSTM outputs + 6 lag statistics from `X_val_raw[:, -24:, :]` (mean/std/min of bz_gse, mean/std/max of speed) → `(N_val, 12)`
3. Split val 80/20 for LGBM train/test. Train one `LGBMRegressor` per step with early stopping
4. Grid-search blend weight α ∈ [0,1]: `final = α·lstm + (1−α)·lgbm`. Print 3-way RMSE table per step

*Expected gain: ~0.3–1.0 nT further on t+6 RMSE*
*Note: training LGBM on the same val set used for LSTM evaluation is acceptable for research but not competition — add a comment to the cell noting this*

### Phase 4 — Multi-Fidelity GRU Residual Correction (Hu et al. 2023)

Train a small GRU to predict and correct LSTM's errors.

New cells:
1. Define `ResidualGRU`: 1-layer GRU (hidden=32) → Linear(32→6)
2. Collect LSTM training-set predictions via a **`shuffle=False` loader** (critical — standard `train_loader` shuffles, breaking residual alignment). Compute `residuals = y_train - lstm_pred`
3. Train `ResidualGRU` on `(X_train_scaled, train_residuals)` for 30 epochs, patience=8. Save to `gru_corrector.pth`
4. Evaluate: `final = model(X_v) + gru_corrector(X_v)`. Print per-step RMSE: LSTM-alone vs LSTM+GRU

*Expected gain: 0.5–2.0 nT if residuals contain learnable structure; near-zero is also informative*
*Critical: always use a separate `shuffle=False` DataLoader when collecting training residuals*

### Expected Metric Trajectory

| Stage | t+6 RMSE | Pearson r |
|---|---|---|
| Current | 9.54 nT | 0.644 |
| Phase 1 (eval reform) | 9.54 nT | 0.644 | ← same model, fuller picture |
| Phase 2 (BiLSTM + features) | ~7.5–8.5 nT | ~0.75 |
| Phase 3 (+ LGBM blend) | ~7.0–8.0 nT | ~0.78 |
| Phase 4 (+ GRU correction) | ~6.8–8.0 nT | ~0.80 |
| Combined (BiLSTM+GRU+LGBM) | 8.3234 nT | — | ← last 20% of val
| Transformer encoder (standalone) | 8.2666 nT | — | ← full val; marginal gain over BiLSTM alone

---

## Progress Log

### Phase 1 — Completed (2026-04-26)
- Added full 6-step inference collection (`all_preds`, `all_actuals` of shape `(N_val, 6)`)
- Added per-step RMSE/Pearson r/R² table for t+1 through t+6, saved `per_step_rmse.png`
- Added storm-conditional RMSE table (Quiet / Moderate / Intense bins), saved `storm_conditional_rmse.png`
- **Results:** RMSE 8.85 nT, Pearson r 0.696, R² 0.358 (t+6)
- Per-step RMSE flat (8.61→8.85 nT), indicating stable multi-step forecasting
- Critical finding: storm-conditional metrics catastrophically bad — Intense storms RMSE 26.1 nT, R² −7.56; model predicts near-mean during extreme events

### Phase 2 — Completed (2026-04-27)
- `SEQ_LEN` 24→48h, `HIDDEN_DIM` 64→128, `DROPOUT` 0.4→0.3, `BIDIRECTIONAL=True`
- 5 new features: `bz_6h`, `speed_6h`, `bz_12h`, `speed_12h`, `dyn_pressure` (10→15 features)
- BiLSTM model with updated attention/fc dimensions (`hidden*D` factor)
- **Modified loss:** two-tier penalty — moderate (Dst < −20) 5×, intense (Dst < −50) 15×
- Gradient clipping `max_norm=1.0`, early stopping patience 10→15
- **Results (full val set):** RMSE 8.4531 nT, Pearson r 0.7201, R² 0.4141 (t+6)
- Intense storm RMSE improved 26.1→20.7 nT due to 15× penalty; storm R² still negative
- Saves to `solar_bilstm_model.pth`

### Phase 3 — Completed (2026-04-27)
- LightGBM ensemble stacked on BiLSTM predictions
- Meta-features (15 total): 6 LSTM outputs + bz_gse mean/std/min + speed mean/std/max + dyn_pressure mean/std + bz_12h mean (last 24h of input window)
- Chronological 80/20 split within val for LGBM train/test (no shuffle)
- One `LGBMRegressor` per step, early stopping patience=50; blend alpha grid-searched ∈ [0,1]
- **Results (last 20% of val):** blend RMSE 8.00 nT at t+6, best alpha ~0.33 (65-75% LGBM weight)
- Key finding: LGBM alone outperforms LSTM alone at every step — tree captures nonlinear interactions LSTM misses

### Combined Ensemble — Completed (2026-04-27)
- Stacks all three improvements: BiLSTM → GRU correction → LGBM blend
- LGBM meta-features rebuilt using GRU-corrected predictions (better signal than raw BiLSTM)
- Grid-searched blend α ∈ [0,1]; best alpha at t+6 = 0.60 (60% GRU-corrected, 40% LGBM)
- Evaluated on last 20% of val (same test split as Phase 3) to avoid leakage
- **Results (last 20% of val):** BiLSTM+GRU+LGBM RMSE 8.3234 nT at t+6
- Per-step gains vs BiLSTM alone: +0.38 nT at t+6, largest gain at t+1 (+1.18 nT)
- Key finding: with GRU-corrected inputs, neural ensemble now contributes 60% vs LGBM 40% — opposite of Phase 3 (where LGBM dominated at 80-90%). GRU correction shifted the balance back toward the neural model.

### Phase 4 — Completed (2026-04-27)
- `ResidualGRU`: 1-layer GRU (hidden=32) → Linear(32→6), trained on training-set residuals
- Collected training residuals via `shuffle=False` DataLoader to preserve alignment
- Early stopping patience=8; training stopped at epoch 11 (best at epoch 1 — see below)
- **Results (full val set):** LSTM+GRU RMSE 7.775 nT at t+6, delta +0.68 nT vs LSTM alone
- Gains consistent across all steps: 0.67–0.92 nT improvement
- Key finding: residual mean = +3.15 nT — BiLSTM has a systematic positive bias (underpredicts Dst). GRU corrected this mean shift in epoch 1; subsequent epochs overfitted. Improvement is largely mean-bias correction, not complex residual structure.
- Saves to `gru_corrector.pth`

---

## Key Observations & Negative Results (DO NOT REPEAT)

### Storm Bias Investigation (2026-04-27)

**Root cause identified:** The BiLSTM has condition-dependent bias — not a global mean shift.
- Quiet times (Dst ≥ −20): model predicts −2.89 nT too negative (overshoots)
- Moderate storms: model predicts +4.07 nT too positive (undershoots magnitude)
- Intense storms: model predicts +10.55 nT too positive (undershoots magnitude)
- Per-step bias grows with horizon: +0.95 nT at t+1 → +2.23 nT at t+6 (regression toward mean)
- The overall +2.22 nT mean residual is misleading — dominated by quiet samples (91% of data)

**Three fixes attempted — all failed or backfired:**

1. **Linear recalibration (post-hoc)** — fitted `actual = a*pred + b` on training predictions per step. Slopes ~0.90 (< 1.0) indicated training predictions were too negative. Applied to val: helped quiet times (+0.88 nT) but made storms worse (intense −3.53 nT). Verdict: training/val bias patterns differ structurally; a single linear transform cannot fix both simultaneously.

2. **Directional asymmetric loss** — added `2.5×` extra penalty when `pred > actual` during storm periods (outputs.detach() > y_batch & y_batch < -20). Effective weights: moderate underpred 12.5×, intense underpred 37.5×. Result: overall RMSE 8.45→9.54 nT, quiet 8.01→9.28 nT (badly regressed), storm improvement marginal (intense 20.69→20.49 nT). Verdict: 37.5× weight on 145 intense samples is enough to destabilise quiet-time learning but not enough to fix storm predictions.

3. **Oversampling** — duplicated moderate windows 4× and intense windows 15× in training set, shuffled before DataLoader. Result: overall RMSE 8.45→9.67 nT, moderate 11.30→14.90 nT, intense 20.69→27.01 nT — everything worse. Verdict: model memorises duplicated sequences without generalising; with only 145 intense samples, 15× repetition causes severe overfitting.

**Conclusion:** Storm performance is a hard data-scarcity constraint. ~145 intense storm samples in the val set (and proportionally few in training) means loss reweighting and oversampling cannot bridge the gap. The Phase 2 model (RMSE 8.4531 nT) remains the best checkpoint. Any future attempt at storm improvement requires either (a) more training data covering extreme events, or (b) a physics-informed prior rather than pure data-driven reweighting.

**Best model checkpoint:** `solar_bilstm_model.pth` (Phase 2, RMSE 8.4531 nT, Pearson r 0.7201)

### Persistence Baseline Analysis (2026-04-28)

**Key finding: the BiLSTM is worse than persistence at t+1 through t+5. Only BiLSTM+GRU barely beats persistence at t+6 (+2.9% skill).**

| Step | Persistence RMSE | BiLSTM RMSE | BiLSTM+GRU RMSE | Skill (GRU) |
|---|---|---|---|---|
| t+1 | 2.88 nT | 8.17 nT | 7.25 nT | −151.8% |
| t+2 | 4.73 nT | 7.94 nT | 7.17 nT | −51.6% |
| t+3 | 5.99 nT | 7.97 nT | 7.28 nT | −21.5% |
| t+4 | 6.86 nT | 8.15 nT | 7.48 nT | −9.0% |
| t+5 | 7.50 nT | 8.32 nT | 7.64 nT | −1.9% |
| t+6 | 8.01 nT | 8.45 nT | 7.78 nT | +2.9% |

Storm-conditional at t+6:
- **Quiet:** persistence 6.72 nT vs GRU 6.75 nT — GRU is 0.4% *worse* than persistence
- **Moderate:** persistence 14.19 nT vs GRU 13.73 nT — GRU +3.3% skill
- **Intense:** persistence 33.64 nT vs GRU 23.80 nT — **GRU +29.3% skill**

**Interpretation:** Dst is strongly autocorrelated at short horizons — persistence dominates t+1-t+5 in quiet conditions (91% of data). The model's real value is storm detection: +29.3% skill on intense events at t+6. The model is penalised in aggregate by being worse than persistence during quiet times.

### Hybrid Switching Experiment (2026-04-28)

Tested condition-dependent switching: use persistence when `last_dst ≥ threshold`, BiLSTM+GRU when `last_dst < threshold`. Grid-searched thresholds −5 to −40 nT.

**Result:** Grid search selected −40 nT (only 332 samples = 1.2% use the model). At this threshold:
- t+6 RMSE = 7.98 nT — **worse** than pure BiLSTM+GRU (7.78 nT)
- Intense storm hybrid RMSE = 33.63 nT ≈ pure persistence (33.64 nT) — storm skill completely lost

**Root cause of failure:** Storm onset (when the model's storm skill is needed) doesn't align with current Dst. A storm peaking at Dst = −60 nT at t+6 often has last_dst = −25 nT at prediction time — it gets routed to persistence, which is terrible. Hard threshold switching can't solve a timing mismatch.

**Key finding from per-step table:** The horizon-specific breakdown reveals the real opportunity:

| Horizon | Best strategy | RMSE |
|---|---|---|
| t+1 | Persistence | 2.88 nT (vs GRU 7.25 nT) |
| t+2 | Persistence | 4.73 nT (vs GRU 7.17 nT) |
| t+3 | Persistence | 5.99 nT (vs GRU 7.28 nT) |
| t+4 | Persistence | 6.86 nT (vs GRU 7.48 nT) |
| t+5 | Persistence | 7.50 nT (vs GRU 7.64 nT) |
| t+6 | BiLSTM+GRU | 7.78 nT (vs persistence 8.01 nT) |

**Conclusion:** For a real-time forecasting system, output persistence for t+1–t+5 and BiLSTM+GRU for t+6. No threshold needed — the switchover is purely by horizon. Do NOT attempt further threshold tuning; the timing mismatch is structural.

### Physics-Informed Feature Engineering (2026-04-28)

- Added 5 solar wind coupling functions to the 15-feature set (15→20 features): `bz_south` (max(0,−Bz)), `vBs` (Burton injection), `clock_angle` (IMF clock angle θ), `epsilon` (Perreault-Akasofu), `newell` (Newell coupling Φ)
- Retrained BiLSTM with identical hyperparameters; saves to `solar_bilstm_physics_model.pth`
- **Results (full val set):** t+6 RMSE 8.6117 nT — **worse** than base BiLSTM (8.4531 nT, delta −0.16 nT)
- Worse at every step (t+1 through t+6, delta −0.16 to −0.27 nT) and every storm bin: quiet −0.16, moderate −0.07, intense −0.71 nT
- **Verdict:** Physics coupling functions are derived nonlinear transforms of raw inputs (bz, by, bt, speed) already in the feature set. The BiLSTM already learns these relationships implicitly. Adding explicit redundant features introduces collinearity and hurts generalisation — the extra signal is noise relative to what the model already extracts. Do not add these features.

---

## Next Session: OMNI Data Augmentation (implement next)

**Why:** Every algorithmic improvement has hit the data-scarcity ceiling (~145 intense storm samples in val, ~proportionally few in training). More storm data is the only remaining lever for intense storm RMSE improvement.

**What to implement:** Add 3 new cells to `SolarWindLSTM.ipynb` after cell `b3aaa3d6` (hybrid baseline cell). The notebook already has all required imports (`torch`, `nn`, `np`, `pd`, `DataLoader`, etc.) from earlier cells.

### Notebook state when these cells run
All of these variables must be in memory (they are if the notebook was run top-to-bottom):
- `df_data` — the base 15-feature DataFrame with `(period, timedelta)` multi-index
- `create_sequences`, `SEQ_LEN=48`, `FORECAST_HORIZON=6`, `TRAIN_SPLIT=0.8`, `BATCH_SIZE=64`
- `HIDDEN_DIM=128`, `NUM_LAYERS=2`, `DROPOUT=0.3`, `BIDIRECTIONAL=True`, `LEARNING_RATE=0.001`, `EPOCHS=50`
- `SolarAttentionLSTM` — the BiLSTM class (already defined, reused here with `input_dim=15`)
- `all_preds`, `all_actuals` — base BiLSTM val predictions (15 features)
- `device` — cuda or cpu

### Data source
**NASA CDAWeb REST API** — no extra packages, just `requests`:
```
https://cdaweb.gsfc.nasa.gov/WS/cdasr/1/dataviews/sp_phys/datasets/OMNI2_H0_MRG1HR/data/{START}T000000Z,{END}T235959Z/BX_GSE,BY_GSE,BZ_GSE,BY_GSM,BZ_GSM,Magnitude,Proton_Density,flow_speed,T,DST1800/?format=json
```
Returns JSON with a `CDF[0].Variables` list. Each variable has `Name` and `Values[0]` (flat array).

**OMNI variable → our column mapping:**
- `BX_GSE` → `bx_gse`, `BY_GSE` → `by_gse`, `BZ_GSE` → `bz_gse`
- `BY_GSM` → `by_gsm`, `BZ_GSM` → `bz_gsm`, `Magnitude` → `bt`
- `Proton_Density` → `density`, `flow_speed` → `speed`, `T` → `temperature`, `DST1800` → `dst`

**Fill values to replace with NaN:**
`BX/BY/BZ/Magnitude`: ≥9999.0 | `Proton_Density`: ≥999.0 | `flow_speed`: ≥99999.0 | `T`: ≥9999999.0 | `DST1800`: ≥99999

### Storm periods to download (all Dst min < −100 nT)
| `period` name | start | end | known Dst min |
|---|---|---|---|
| `omni_halloween_2003` | `20031025` | `20031105` | −422 nT |
| `omni_bastille_2000` | `20000712` | `20000718` | −301 nT |
| `omni_stpatrick_2015` | `20150315` | `20150322` | −222 nT |
| `omni_sep2017` | `20170904` | `20170912` | −142 nT |
| `omni_aug2018` | `20180822` | `20180829` | −174 nT |

Use start/end as `YYYYMMDD` strings in the URL.

### Cell 1 — Download + parse OMNI

```python
import requests, json

CDAWEB = "https://cdaweb.gsfc.nasa.gov/WS/cdasr/1/dataviews/sp_phys/datasets/OMNI2_H0_MRG1HR/data"
OMNI_VARS = "BX_GSE,BY_GSE,BZ_GSE,BY_GSM,BZ_GSM,Magnitude,Proton_Density,flow_speed,T,DST1800"

STORM_PERIODS = {
    'omni_halloween_2003': ('20031025', '20031105'),
    'omni_bastille_2000':  ('20000712', '20000718'),
    'omni_stpatrick_2015': ('20150315', '20150322'),
    'omni_sep2017':        ('20170904', '20170912'),
    'omni_aug2018':        ('20180822', '20180829'),
}

COL_MAP = {'BX_GSE':'bx_gse','BY_GSE':'by_gse','BZ_GSE':'bz_gse',
           'BY_GSM':'by_gsm','BZ_GSM':'bz_gsm','Magnitude':'bt',
           'Proton_Density':'density','flow_speed':'speed','T':'temperature','DST1800':'dst'}

def fetch_omni(start, end):
    url = f"{CDAWEB}/{start}T000000Z,{end}T235959Z/{OMNI_VARS}/?format=json"
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.json()

def parse_omni_to_df(raw, period_name):
    variables = {v['Name']: np.array(v['Values'][0], dtype=float)
                 for v in raw['CDF'][0]['Variables'] if v['Name'] in COL_MAP}
    # First variable's time dimension gives length
    n = len(list(variables.values())[0])
    df = pd.DataFrame(index=range(n))
    df['period']    = period_name
    df['timedelta'] = pd.to_timedelta(np.arange(n), unit='h')   # 0h, 1h, 2h...
    for omni_var, our_col in COL_MAP.items():
        vals = variables[omni_var].copy()
        # Mask fill values
        if our_col in ('bx_gse','by_gse','bz_gse','by_gsm','bz_gsm','bt'):
            vals[vals >= 9999.0] = np.nan
        elif our_col == 'density':
            vals[vals >= 999.0] = np.nan
        elif our_col == 'speed':
            vals[vals >= 99999.0] = np.nan
        elif our_col == 'temperature':
            vals[vals >= 9999999.0] = np.nan
        elif our_col == 'dst':
            vals[vals >= 99999] = np.nan
        df[our_col] = vals
    return df

sw_dfs = []
for name, (start, end) in STORM_PERIODS.items():
    print(f'Downloading {name} ...', end=' ', flush=True)
    raw = fetch_omni(start, end)
    df  = parse_omni_to_df(raw, name)
    n_intense = (df['dst'] < -50).sum()
    print(f'{len(df)} h | intense: {n_intense}')
    sw_dfs.append(df)

omni_df = pd.concat(sw_dfs, ignore_index=True)
print(f'\nTotal: {len(omni_df)} h | intense (Dst<-50): {(omni_df["dst"]<-50).sum()}')
```

### Cell 2 — Build augmented training set + retrain BiLSTM

```python
# Set multi-index, derive the same 15 features as load_and_preprocess, interpolate
omni_indexed = omni_df.set_index(['period', 'timedelta'])

grp = omni_indexed.groupby('period')
omni_indexed['energy']       = omni_indexed['speed'] * omni_indexed['bz_gse']
omni_indexed['bz_3h']        = grp['bz_gse'].transform(lambda x: x.rolling(3,  min_periods=1).mean())
omni_indexed['speed_3h']     = grp['speed'].transform(lambda x: x.rolling(3,  min_periods=1).mean())
omni_indexed['bz_6h']        = grp['bz_gse'].transform(lambda x: x.rolling(6,  min_periods=1).mean())
omni_indexed['speed_6h']     = grp['speed'].transform(lambda x: x.rolling(6,  min_periods=1).mean())
omni_indexed['bz_12h']       = grp['bz_gse'].transform(lambda x: x.rolling(12, min_periods=1).mean())
omni_indexed['speed_12h']    = grp['speed'].transform(lambda x: x.rolling(12, min_periods=1).mean())
omni_indexed['dyn_pressure'] = omni_indexed['density'] * omni_indexed['speed']**2
omni_indexed = omni_indexed.interpolate(method='linear', limit_direction='both').ffill().bfill()

# Sequences from OMNI data only — these become extra TRAINING samples
X_omni, y_omni = create_sequences(omni_indexed[df_data.columns], SEQ_LEN, FORECAST_HORIZON)
print(f'OMNI sequences: {X_omni.shape}  intense targets: {(y_omni[:,-1] < -50).sum()}')

# Augmented train = base train + all OMNI; val = base val unchanged (fair comparison)
split_idx     = int(len(X_all) * TRAIN_SPLIT)   # X_all is the base 15-feat array from cell 0d486255
X_aug_train   = np.concatenate([X_all[:split_idx], X_omni])
y_aug_train   = np.concatenate([y_all[:split_idx], y_omni])
X_aug_val     = X_all[split_idx:]
y_aug_val     = y_all[split_idx:]

scaler_aug    = StandardScaler()
N_tr, T, F   = X_aug_train.shape
X_aug_train_sc = scaler_aug.fit_transform(X_aug_train.reshape(-1,F)).reshape(N_tr,T,F)
X_aug_val_sc   = scaler_aug.transform(X_aug_val.reshape(-1,F)).reshape(len(X_aug_val),T,F)

aug_train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_aug_train_sc), torch.FloatTensor(y_aug_train)), batch_size=BATCH_SIZE, shuffle=True)
aug_val_loader   = DataLoader(TensorDataset(torch.FloatTensor(X_aug_val_sc),   torch.FloatTensor(y_aug_val)),   batch_size=BATCH_SIZE, shuffle=False)

aug_model     = SolarAttentionLSTM(input_dim=F, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
                                    dropout=DROPOUT, bidirectional=BIDIRECTIONAL).to(device)
aug_optimizer = torch.optim.Adam(aug_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

best_aug_val, aug_patience, aug_counter = float('inf'), 15, 0
print(f'Training augmented BiLSTM: {N_tr} train samples (+{len(X_omni)} OMNI) ...')

for epoch in range(EPOCHS):
    aug_model.train()
    bl = []
    for X_b, y_b in aug_train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        aug_optimizer.zero_grad()
        loss_el = (aug_model(X_b) - y_b)**2
        w = torch.ones_like(y_b); w[y_b < -20] = 5.0; w[y_b < -50] = 15.0
        loss = torch.mean(loss_el * w)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(aug_model.parameters(), max_norm=1.0)
        aug_optimizer.step(); bl.append(loss.item())
    aug_model.eval()
    vl = []
    with torch.no_grad():
        for X_v, y_v in aug_val_loader:
            vl.append(nn.MSELoss()(aug_model(X_v.to(device)), y_v.to(device)).item())
    avg_t, avg_v = np.mean(bl), np.mean(vl)
    if (epoch+1)%5==0 or epoch==0: print(f'Epoch {epoch+1} | Train {avg_t:.4f} | Val {avg_v:.4f}')
    if avg_v < best_aug_val:
        best_aug_val = avg_v; aug_counter = 0
        torch.save(aug_model.state_dict(), 'solar_bilstm_augmented_model.pth'); print('  -> Saved')
    else:
        aug_counter += 1
        if aug_counter >= aug_patience: print(f'Early stopping at epoch {epoch+1}'); break
print('Done.')
```

### Cell 3 — Evaluate augmented vs base

```python
aug_model.load_state_dict(torch.load('solar_bilstm_augmented_model.pth', map_location=device))
aug_model.eval()
ap, aa = [], []
with torch.no_grad():
    for X_v, y_v in aug_val_loader:
        ap.append(aug_model(X_v.to(device)).cpu().numpy()); aa.append(y_v.numpy())
aug_preds   = np.vstack(ap); aug_actuals = np.vstack(aa)

print('--- OMNI Augmentation: Base vs Augmented BiLSTM ---')
rows = []
for s in range(FORECAST_HORIZON):
    rb = np.sqrt(mean_squared_error(all_actuals[:,s], all_preds[:,s]))
    ra = np.sqrt(mean_squared_error(aug_actuals[:,s], aug_preds[:,s]))
    rows.append({'Step':f't+{s+1}','Base RMSE':round(rb,4),'Aug RMSE':round(ra,4),'Delta':round(rb-ra,4)})
print(pd.DataFrame(rows).to_string(index=False))

dst_t6 = aug_actuals[:,-1]
bins = {'Quiet  (>=−20)':dst_t6>=-20,'Moderate (−50 to −20)':(dst_t6<-20)&(dst_t6>=-50),'Intense (<−50)':dst_t6<-50}
print('\n--- Storm-Conditional t+6: Base vs Augmented ---')
for lbl, mask in bins.items():
    rb = np.sqrt(mean_squared_error(all_actuals[mask,-1], all_preds[mask,-1]))
    ra = np.sqrt(mean_squared_error(aug_actuals[mask,-1], aug_preds[mask,-1]))
    print(f'{lbl:<28} N={mask.sum():>5}  Base={rb:.4f}  Aug={ra:.4f}  Δ={rb-ra:+.4f}')
```

### Critical notes
- `X_all` and `y_all` (the full base sequence arrays before the 80/20 split) come from cell `0d486255` — they're the `X_all, y_all` variables, not `X_train_raw`/`X_val_raw`.
- Val set is strictly base data only — OMNI goes training-only. This keeps comparison fair.
- If CDAWeb returns 404 or times out: retry once; if still failing, check that the URL has `T000000Z` format for the timestamps.
- After running, update this section of CLAUDE.md with the results and delete this "Next Session" block.
