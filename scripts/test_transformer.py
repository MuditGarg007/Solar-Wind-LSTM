"""
Standalone test: Transformer encoder architecture vs adopted BiLSTM (11.26 nT t+6, intense 44.64).
Single seed=42 quick check on held-out test set. Same pipeline/loss/split as notebook.
Not embedded in notebook unless results justify multi-seed follow-up.
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

SOLAR_PATH    = 'archive/solar_wind.csv'
LABELS_PATH   = 'archive/labels.csv'
SUNSPOTS_PATH = 'archive/sunspots.csv'

SEQ_LEN          = 48
FORECAST_HORIZON = 6
BATCH_SIZE       = 64
HIDDEN_DIM       = 128
DROPOUT          = 0.3
LEARNING_RATE    = 0.001
EPOCHS           = 50
PATIENCE         = 15

TRAIN_SPLIT = 0.7
VAL_SPLIT   = 0.2
TEST_SPLIT  = 0.1
FEATURE_SET = 'paper'


def load_and_preprocess(solar_path, labels_path, sunspots_path, feature_set='paper'):
    print('Loading CSV data...')
    RAW_VARS = [
        'bx_gse', 'by_gse', 'bz_gse',
        'bx_gsm', 'by_gsm', 'bz_gsm',
        'theta_gse', 'phi_gse',
        'theta_gsm', 'phi_gsm',
        'bt', 'density', 'speed', 'temperature',
    ]
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
    n_feats = data.shape[1] - 1
    print(f'Feature count: {n_feats} (feature_set="{feature_set}")')
    return data


def create_sequences(data, seq_len=24, forecast_horizon=6):
    print(f'Creating sequences (Input: {seq_len}h, Target: +1h to +{forecast_horizon}h)...')
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
            X_window = arr_X[i : i + seq_len]
            y_window = arr_y[i + seq_len : i + seq_len + forecast_horizon]
            X_list.append(X_window)
            y_list.append(y_window)
    return np.array(X_list), np.array(y_list)


# --- Pipeline ---
df_data = load_and_preprocess(SOLAR_PATH, LABELS_PATH, SUNSPOTS_PATH, feature_set=FEATURE_SET)

train_X, train_y = [], []
val_X,   val_y   = [], []
test_X,  test_y  = [], []

_all_periods = df_data.index.get_level_values('period').unique()
for period in _all_periods:
    period_mask = df_data.index.get_level_values('period') == period
    period_data = df_data[period_mask]
    X_p, y_p = create_sequences(period_data, SEQ_LEN, FORECAST_HORIZON)
    n    = len(X_p)
    n_tr = int(n * TRAIN_SPLIT)
    n_vl = int(n * VAL_SPLIT)
    train_X.append(X_p[:n_tr]);           train_y.append(y_p[:n_tr])
    val_X.append(X_p[n_tr:n_tr + n_vl]); val_y.append(y_p[n_tr:n_tr + n_vl])
    test_X.append(X_p[n_tr + n_vl:]);    test_y.append(y_p[n_tr + n_vl:])

X_train_raw = np.concatenate(train_X);  y_train = np.concatenate(train_y)
X_val_raw   = np.concatenate(val_X);    y_val   = np.concatenate(val_y)
X_test_raw  = np.concatenate(test_X);   y_test  = np.concatenate(test_y)

print(f'Train: {len(X_train_raw)} | Val: {len(X_val_raw)} | Test: {len(X_test_raw)}')

scaler = StandardScaler()
N_tr, T, F_dim = X_train_raw.shape
X_train_scaled = scaler.fit_transform(X_train_raw.reshape(-1, F_dim)).reshape(N_tr, T, F_dim)
N_vl = len(X_val_raw)
X_val_scaled = scaler.transform(X_val_raw.reshape(-1, F_dim)).reshape(N_vl, T, F_dim)
N_te = len(X_test_raw)
X_test_scaled = scaler.transform(X_test_raw.reshape(-1, F_dim)).reshape(N_te, T, F_dim)

train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train))
val_dataset   = TensorDataset(torch.FloatTensor(X_val_scaled),   torch.FloatTensor(y_val))
test_dataset  = TensorDataset(torch.FloatTensor(X_test_scaled),  torch.FloatTensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f'Input dim: {F_dim} | Data pipeline complete.')


# --- Transformer model ---
class SolarTransformer(nn.Module):
    """Encoder-only transformer: input projection + learned positional embedding
    -> TransformerEncoder -> concat(mean-pool, last-token) -> Linear -> 6 preds.
    Mirrors SolarAttentionLSTM's context+last_hidden output head for parity."""
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2,
                 dim_feedforward=256, output_dim=6, dropout=0.3, seq_len=48):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed  = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model * 2, output_dim)

    def forward(self, x):
        h = self.input_proj(x) + self.pos_embed
        h = self.encoder(h)
        mean_pool = h.mean(dim=1)
        last_tok  = h[:, -1, :]
        combined = torch.cat((mean_pool, last_tok), dim=1)
        combined = self.dropout(combined)
        return self.fc(combined)


model = SolarTransformer(
    input_dim=F_dim, d_model=HIDDEN_DIM, nhead=4, num_layers=2,
    dim_feedforward=HIDDEN_DIM * 2, output_dim=FORECAST_HORIZON,
    dropout=DROPOUT, seq_len=SEQ_LEN,
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f'SolarTransformer params: {n_params:,}')

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-5)

best_val_loss = float('inf')
counter = 0
best_state = None

print(f'Starting training for {EPOCHS} epochs (storm-weighted MSE)...')
for epoch in range(EPOCHS):
    model.train()
    batch_losses = []
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss_elements = (outputs - y_batch) ** 2
        weights = torch.ones_like(y_batch)
        weights[y_batch < -20] = 5.0
        weights[y_batch < -50] = 15.0
        loss = torch.mean(loss_elements * weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        batch_losses.append(loss.item())

    model.eval()
    val_batch_losses = []
    with torch.no_grad():
        for X_v, y_v in val_loader:
            X_v, y_v = X_v.to(device), y_v.to(device)
            preds = model(X_v)
            val_batch_losses.append(criterion(preds, y_v).item())

    avg_train_loss = np.mean(batch_losses)
    avg_val_loss = np.mean(val_batch_losses)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f'Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        counter += 1
        if counter >= PATIENCE:
            print(f'Early stopping triggered at epoch {epoch+1}')
            break

model.load_state_dict(best_state)
model.eval()

# --- Held-out test evaluation ---
test_preds_list, test_actuals_list = [], []
with torch.no_grad():
    for X_b, y_b in test_loader:
        test_preds_list.append(model(X_b.to(device)).cpu().numpy())
        test_actuals_list.append(y_b.numpy())
test_preds_arr   = np.vstack(test_preds_list)
test_actuals_arr = np.vstack(test_actuals_list)

print('\n--- SolarTransformer: Held-out Test Set ---')
rows_test = []
for s in range(FORECAST_HORIZON):
    rmse_s = np.sqrt(mean_squared_error(test_actuals_arr[:, s], test_preds_arr[:, s]))
    corr_s = stats.pearsonr(test_actuals_arr[:, s], test_preds_arr[:, s])[0]
    r2_s   = r2_score(test_actuals_arr[:, s], test_preds_arr[:, s])
    rows_test.append({'Step': f't+{s+1}', 'RMSE (nT)': round(rmse_s, 4),
                      'Pearson r': round(corr_s, 4), 'R2': round(r2_s, 4)})
df_test = pd.DataFrame(rows_test)
print(df_test.to_string(index=False))

dst_t6_test = test_actuals_arr[:, -1]
preds_t6_test = test_preds_arr[:, -1]
bins_test = {
    'Quiet (Dst >= -20)':          dst_t6_test >= -20,
    'Moderate (-50 <= Dst < -20)': (dst_t6_test < -20) & (dst_t6_test >= -50),
    'Intense (Dst < -50)':         dst_t6_test < -50,
}
print('\nTest set storm-conditional (t+6):')
for lbl, mask in bins_test.items():
    if mask.sum() == 0:
        print(f'  {lbl:<38} N=0')
        continue
    rmse_s = np.sqrt(mean_squared_error(dst_t6_test[mask], preds_t6_test[mask]))
    print(f'  {lbl:<38} N={mask.sum():>5}  RMSE={rmse_s:.4f} nT')

print('\n--- Reference (adopted BiLSTM base, held-out test) ---')
print('  t+6 aggregate RMSE: 11.26 nT | Intense (N=387): 44.64 nT')
