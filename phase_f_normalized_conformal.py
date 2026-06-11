"""
Phase F: Normalized conformal prediction with a difficulty estimator
(recent solar-wind volatility) vs marginal / Mondrian conformal.

Standalone script (mirrors prior standalone-script phases). No retrain --
loads the existing solar_bilstm_model.pth checkpoint, reproduces the
Phase A data pipeline + scaler, and evaluates conformal intervals on the
held-out test set.

Difficulty estimator: for each 48h input window, sigma(x) = mean over the
window of (bz_gse_std + bt_std + speed_std) -- the three std-of-hourly
features for the most geoeffective variables. This is "recent solar-wind
volatility", known at prediction time, and is exactly the lever flagged
as the optional Phase C follow-up in CLAUDE.md.

Outputs: conformal_normalized.csv, conformal_method_comparison.csv
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from crepes import ConformalRegressor

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# --- Config (matches CLAUDE.md / cell 3) ---
SOLAR_PATH    = 'archive/solar_wind.csv'
LABELS_PATH   = 'archive/labels.csv'
SUNSPOTS_PATH = 'archive/sunspots.csv'

SEQ_LEN          = 48
FORECAST_HORIZON = 6
BATCH_SIZE       = 64
HIDDEN_DIM       = 128
NUM_LAYERS       = 2
DROPOUT          = 0.4
BIDIRECTIONAL    = True
TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT = 0.7, 0.2, 0.1
FEATURE_SET = 'paper'

RAW_VARS = [
    'bx_gse', 'by_gse', 'bz_gse',
    'bx_gsm', 'by_gsm', 'bz_gsm',
    'theta_gse', 'phi_gse',
    'theta_gsm', 'phi_gsm',
    'bt', 'density', 'speed', 'temperature',
]


def load_and_preprocess(solar_path, labels_path, sunspots_path, feature_set='paper'):
    print('Loading CSV data...')
    df_solar   = pd.read_csv(solar_path)
    df_labels  = pd.read_csv(labels_path)
    df_sunspot = pd.read_csv(sunspots_path)

    for df in (df_solar, df_labels, df_sunspot):
        df['timedelta'] = pd.to_timedelta(df['timedelta'])

    df_solar.set_index(['period', 'timedelta'],   inplace=True)
    df_labels.set_index(['period', 'timedelta'],  inplace=True)
    df_sunspot.set_index(['period', 'timedelta'], inplace=True)

    print('Aggregating to hourly mean + std for 14 solar-wind variables...')
    grp_solar = df_solar.groupby(['period', pd.Grouper(level='timedelta', freq='1h')])

    mean_df = grp_solar[RAW_VARS].mean()
    mean_df.columns = [f'{c}_mean' for c in RAW_VARS]

    std_df = grp_solar[RAW_VARS].std()
    std_df.columns = [f'{c}_std' for c in RAW_VARS]

    hourly = pd.concat([mean_df, std_df], axis=1)

    print('Merging smoothed_ssn...')
    ssn_hourly = df_sunspot.groupby(
        ['period', pd.Grouper(level='timedelta', freq='1h')]
    )['smoothed_ssn'].mean()
    hourly = hourly.join(ssn_hourly, how='left')
    hourly['smoothed_ssn'] = (
        hourly.groupby('period')['smoothed_ssn']
              .transform(lambda x: x.ffill().bfill())
    )

    data = hourly.join(df_labels, how='inner')
    data = data.interpolate(method='linear', limit_direction='both')
    data = data.ffill().bfill()
    return data


def create_sequences(data, seq_len=24, forecast_horizon=6):
    feature_cols = [c for c in data.columns if c != 'dst']
    target_col = 'dst'
    X_list, y_list = [], []
    for period in data.index.get_level_values('period').unique():
        group = data.loc[period]
        if len(group) <= seq_len + forecast_horizon:
            continue
        arr_X = group[feature_cols].values
        arr_y = group[target_col].values
        for i in range(len(group) - seq_len - forecast_horizon + 1):
            X_list.append(arr_X[i:i + seq_len])
            y_list.append(arr_y[i + seq_len:i + seq_len + forecast_horizon])
    return np.array(X_list), np.array(y_list), feature_cols


class SolarAttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=6, dropout=0.3, bidirectional=False):
        super().__init__()
        D = 2 if bidirectional else 1
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                             batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.attention_fc = nn.Linear(hidden_dim * D, 1)
        self.fc = nn.Linear(hidden_dim * D * 2, output_dim)

    def forward(self, x, return_attention=False):
        lstm_out, _ = self.lstm(x)
        scores = self.attention_fc(lstm_out)
        attention_weights = F.softmax(scores, dim=1)
        context_vector = torch.sum(lstm_out * attention_weights, dim=1)
        last_hidden = lstm_out[:, -1, :]
        combined = torch.cat((context_vector, last_hidden), dim=1)
        out = self.fc(combined)
        if return_attention:
            return out, attention_weights
        return out


# --- 1. Data pipeline (reproduces cell 0d486255) ---
df_data = load_and_preprocess(SOLAR_PATH, LABELS_PATH, SUNSPOTS_PATH, feature_set=FEATURE_SET)

train_X, train_y = [], []
val_X, val_y = [], []
test_X, test_y = [], []
feature_cols = None

for period in df_data.index.get_level_values('period').unique():
    period_mask = df_data.index.get_level_values('period') == period
    period_data = df_data[period_mask]
    X_p, y_p, feature_cols = create_sequences(period_data, SEQ_LEN, FORECAST_HORIZON)
    if len(X_p) == 0:
        continue
    n = len(X_p)
    n_tr = int(n * TRAIN_SPLIT)
    n_vl = int(n * VAL_SPLIT)
    train_X.append(X_p[:n_tr]); train_y.append(y_p[:n_tr])
    val_X.append(X_p[n_tr:n_tr + n_vl]); val_y.append(y_p[n_tr:n_tr + n_vl])
    test_X.append(X_p[n_tr + n_vl:]); test_y.append(y_p[n_tr + n_vl:])

X_train_raw = np.concatenate(train_X); y_train = np.concatenate(train_y)
X_val_raw   = np.concatenate(val_X);   y_val   = np.concatenate(val_y)
X_test_raw  = np.concatenate(test_X);  y_test  = np.concatenate(test_y)

print(f'Train: {len(X_train_raw)} | Val: {len(X_val_raw)} | Test: {len(X_test_raw)}')

scaler = StandardScaler()
N_tr, T, Fd = X_train_raw.shape
X_train_scaled = scaler.fit_transform(X_train_raw.reshape(-1, Fd)).reshape(N_tr, T, Fd)
X_val_scaled   = scaler.transform(X_val_raw.reshape(-1, Fd)).reshape(len(X_val_raw), T, Fd)
X_test_scaled  = scaler.transform(X_test_raw.reshape(-1, Fd)).reshape(len(X_test_raw), T, Fd)

# --- 2. Load adopted base checkpoint, infer val + test ---
model = SolarAttentionLSTM(input_dim=Fd, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
                            dropout=DROPOUT, bidirectional=BIDIRECTIONAL).to(device)
model.load_state_dict(torch.load('solar_bilstm_model.pth', map_location=device))
model.eval()


def infer(Xs):
    P = []
    dl = DataLoader(TensorDataset(torch.FloatTensor(Xs)), batch_size=BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for (Xb,) in dl:
            P.append(model(Xb.to(device)).cpu().numpy())
    return np.vstack(P)


val_preds  = infer(X_val_scaled)
test_preds = infer(X_test_scaled)

# --- 3. Difficulty estimator: recent solar-wind volatility ---
# sigma(x) = mean over the 48h window of (bz_gse_std + bt_std + speed_std)
vol_cols = [feature_cols.index(c) for c in ('bz_gse_std', 'bt_std', 'speed_std')]
print('Volatility feature indices:', dict(zip(('bz_gse_std', 'bt_std', 'speed_std'), vol_cols)))


def difficulty(X_raw):
    win_vol = X_raw[:, :, vol_cols].sum(axis=2)   # (N, T)
    return np.nanmean(win_vol, axis=1) + 1e-6     # (N,) avoid zero sigma


sigma_val  = difficulty(X_val_raw)
sigma_test = difficulty(X_test_raw)
print(f'sigma_val  range: {sigma_val.min():.3f} - {sigma_val.max():.3f}, mean {sigma_val.mean():.3f}')
print(f'sigma_test range: {sigma_test.min():.3f} - {sigma_test.max():.3f}, mean {sigma_test.mean():.3f}')

CONFS = [0.90, 0.95, 0.99]
H = FORECAST_HORIZON
res_cal = y_val - val_preds


def sev_bin(pred):
    b = np.zeros_like(pred, dtype=int)
    b[pred < -20] = 1
    b[pred < -50] = 2
    return b


# Fit marginal, Mondrian (predicted-severity bins), and normalized (volatility sigma) regressors
cr_marg, cr_mond, cr_norm = [], [], []
for h in range(H):
    m = ConformalRegressor(); m.fit(residuals=res_cal[:, h]); cr_marg.append(m)
    mm = ConformalRegressor(); mm.fit(residuals=res_cal[:, h], bins=sev_bin(val_preds[:, h])); cr_mond.append(mm)
    nn_ = ConformalRegressor(); nn_.fit(residuals=res_cal[:, h], sigmas=sigma_val); cr_norm.append(nn_)

# --- (1) Marginal vs normalized: per-horizon coverage / mean width ---
rows = []
for h in range(H):
    y, yhat = y_test[:, h], test_preds[:, h]
    row = {'Step': f't+{h+1}'}
    for c in CONFS:
        lo, hi = cr_marg[h].predict_int(y_hat=yhat, confidence=c).T
        row[f'marg_cov@{int(c*100)}'] = round(np.mean((y >= lo) & (y <= hi)) * 100, 1)
        row[f'marg_w@{int(c*100)}']   = round(float(np.mean(hi - lo)), 1)

        lo, hi = cr_norm[h].predict_int(y_hat=yhat, sigmas=sigma_test, confidence=c).T
        row[f'norm_cov@{int(c*100)}'] = round(np.mean((y >= lo) & (y <= hi)) * 100, 1)
        row[f'norm_w@{int(c*100)}']   = round(float(np.mean(hi - lo)), 1)
    rows.append(row)
df_norm = pd.DataFrame(rows)
print('\n--- Phase F: Marginal vs Normalized (volatility-sigma) conformal, per-horizon, held-out test ---')
print(df_norm.to_string(index=False))

# --- (2) Per-storm-bin coverage @95% (t+6): marginal vs Mondrian vs normalized ---
h = H - 1
y6, yhat6 = y_test[:, h], test_preds[:, h]
bins_actual = {
    'Quiet (Dst >= -20)':          y6 >= -20,
    'Moderate (-50 <= Dst < -20)': (y6 < -20) & (y6 >= -50),
    'Intense (Dst < -50)':         y6 < -50,
}
C = 0.95
lo_m, hi_m = cr_marg[h].predict_int(y_hat=yhat6, confidence=C).T
lo_o, hi_o = cr_mond[h].predict_int(y_hat=yhat6, bins=sev_bin(yhat6), confidence=C).T
lo_n, hi_n = cr_norm[h].predict_int(y_hat=yhat6, sigmas=sigma_test, confidence=C).T

brows = []
for lbl, mask in bins_actual.items():
    n = int(mask.sum())
    brows.append({
        'Bin (by actual)': lbl, 'N': n,
        'marg cov%': round(np.mean((y6[mask] >= lo_m[mask]) & (y6[mask] <= hi_m[mask])) * 100, 1),
        'marg w':    round(float(np.mean(hi_m[mask] - lo_m[mask])), 1),
        'mond cov%': round(np.mean((y6[mask] >= lo_o[mask]) & (y6[mask] <= hi_o[mask])) * 100, 1),
        'mond w':    round(float(np.mean(hi_o[mask] - lo_o[mask])), 1),
        'norm cov%': round(np.mean((y6[mask] >= lo_n[mask]) & (y6[mask] <= hi_n[mask])) * 100, 1),
        'norm w':    round(float(np.mean(hi_n[mask] - lo_n[mask])), 1),
    })
df_bin = pd.DataFrame(brows)
print(f'\n--- Per-storm-bin coverage @ {int(C*100)}% (t+6): Marginal vs Mondrian vs Normalized (volatility) ---')
print(df_bin.to_string(index=False))

# --- aggregate sanity ---
print('\n--- Aggregate marginal vs normalized coverage across horizons (sanity: should ~= nominal) ---')
for c in CONFS:
    covs_m, covs_n = [], []
    for h in range(H):
        lo, hi = cr_marg[h].predict_int(y_hat=test_preds[:, h], confidence=c).T
        covs_m.append(np.mean((y_test[:, h] >= lo) & (y_test[:, h] <= hi)) * 100)
        lo, hi = cr_norm[h].predict_int(y_hat=test_preds[:, h], sigmas=sigma_test, confidence=c).T
        covs_n.append(np.mean((y_test[:, h] >= lo) & (y_test[:, h] <= hi)) * 100)
    print(f'  {int(c*100)}%: marginal {np.mean(covs_m):.1f}%  normalized {np.mean(covs_n):.1f}%')

df_norm.to_csv('conformal_normalized.csv', index=False)
df_bin.to_csv('conformal_method_comparison.csv', index=False)
print('\nSaved conformal_normalized.csv, conformal_method_comparison.csv')
