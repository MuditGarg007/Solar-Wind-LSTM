"""Probe: recover absolute dates of MagNet periods (train_a/b/c) by cross-correlating
their hourly Dst against the historical OMNI2 hourly Dst record.
If lock is clean, #4 (Kp/Ap/AE inputs) is feasible on the held-out test."""
import requests, numpy as np, pandas as pd
from scipy.signal import fftconvolve

CDAWEB = 'https://cdaweb.gsfc.nasa.gov/WS/cdasr/1/dataviews/sp_phys/datasets/OMNI2_H0_MRG1HR/data'

def fetch_dst(start, end):
    url = f'{CDAWEB}/{start}T000000Z,{end}T235959Z/DST1800/?format=json'
    r = requests.get(url, timeout=300, headers={'Accept': 'application/json'})
    r.raise_for_status()
    j = r.json()
    var = j['CDF'][0]['cdfVariables']['variable']
    epoch = dst = None
    for v in var:
        if v.get('name') == 'Epoch':
            epoch = [rec['value'][0] for rec in v['cdfVarData']['record']]
        elif v.get('name') == 'DST1800':
            dst = [float(rec['value'][0]) for rec in v['cdfVarData']['record']]
    s = pd.Series(dst, index=pd.to_datetime(epoch))
    s[s >= 99999.0] = np.nan
    return s

print('Downloading OMNI hourly Dst 1995-2020 (chunked)...')
chunks = []
for y0 in range(1995, 2020, 5):
    y1 = min(y0 + 5, 2020)
    print(f'  {y0}-{y1}...', flush=True)
    chunks.append(fetch_dst(f'{y0}0101', f'{y1}0101'))
omni = pd.concat(chunks)
omni = omni[~omni.index.duplicated()].sort_index()
omni = omni.asfreq('1h')
print(f'OMNI Dst: {omni.index.min()} -> {omni.index.max()}, n={len(omni)}, nan={omni.isna().sum()}')

# MagNet labels
lab = pd.read_csv('archive/labels.csv')
lab['timedelta'] = pd.to_timedelta(lab['timedelta'])

omni_vals = omni.values.astype(float)
omni_filled = np.nan_to_num(omni_vals, nan=0.0)
omni_mask = (~np.isnan(omni_vals)).astype(float)

for period, g in lab.groupby('period'):
    g = g.sort_values('timedelta')
    seq = g['dst'].values.astype(float)
    # use a distinctive 60-day window centered on the period's strongest storm (min Dst)
    c = int(np.argmin(seq)); W = 24 * 60
    a0, a1 = max(0, c - W // 2), min(len(seq), c + W // 2)
    win = seq[a0:a1]
    wmask = (~np.isnan(win)).astype(float)
    win_f = np.nan_to_num(win, nan=0.0)
    n = len(win)
    # sliding SSE = sum(b^2 over window) - 2*sum(a*b)   (a = win, b = omni slice; mask on both)
    cross = fftconvolve(omni_filled, win_f[::-1], mode='valid')          # sum a*b at each lag
    b2 = fftconvolve(omni_filled ** 2, wmask[::-1], mode='valid')        # sum b^2 (only where a valid)
    cnt = fftconvolve(omni_mask, wmask[::-1], mode='valid')              # overlap count
    a2 = np.sum(win_f ** 2)
    sse = a2 + b2 - 2 * cross
    valid = cnt > (0.5 * wmask.sum())
    sse[~valid] = np.inf
    rmse = np.sqrt(sse / np.maximum(cnt, 1))
    best = int(np.argmin(rmse))
    # best lag = start index in omni for win[0]; win[0] is at period offset a0
    period_start_idx = best - a0
    period_start_time = omni.index[max(0, period_start_idx)]
    second = np.partition(rmse[np.isfinite(rmse)], 1)[1]
    print(f'{period}: match RMSE={rmse[best]:.2f} nT (2nd best {second:.2f}) '
          f'-> start ~ {period_start_time}  span {g["timedelta"].max()}')
