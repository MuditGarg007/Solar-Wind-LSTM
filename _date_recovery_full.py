"""Full-series alignment for all 3 MagNet periods -> recover absolute start dates.
Caches OMNI Dst locally."""
import requests, numpy as np, pandas as pd, os
from scipy.signal import fftconvolve

CDAWEB = 'https://cdaweb.gsfc.nasa.gov/WS/cdasr/1/dataviews/sp_phys/datasets/OMNI2_H0_MRG1HR/data'
CACHE = '_omni_dst_1995_2020.pkl'

def fetch_dst(start, end):
    url = f'{CDAWEB}/{start}T000000Z,{end}T235959Z/DST1800/?format=json'
    r = requests.get(url, timeout=300, headers={'Accept': 'application/json'}); r.raise_for_status()
    var = r.json()['CDF'][0]['cdfVariables']['variable']
    epoch = dst = None
    for v in var:
        if v.get('name') == 'Epoch': epoch = [rec['value'][0] for rec in v['cdfVarData']['record']]
        elif v.get('name') == 'DST1800': dst = [float(rec['value'][0]) for rec in v['cdfVarData']['record']]
    s = pd.Series(dst, index=pd.to_datetime(epoch)); s[s >= 99999.0] = np.nan; return s

if os.path.exists(CACHE):
    omni = pd.read_pickle(CACHE)
else:
    omni = pd.concat([fetch_dst(f'{y}0101', f'{min(y+5,2020)}0101') for y in range(1995, 2020, 5)])
    omni = omni[~omni.index.duplicated()].sort_index().asfreq('1h')
    omni.to_pickle(CACHE)
print(f'OMNI Dst {omni.index.min()}..{omni.index.max()} n={len(omni)}')

lab = pd.read_csv('archive/labels.csv'); lab['timedelta'] = pd.to_timedelta(lab['timedelta'])
of = np.nan_to_num(omni.values.astype(float)); om = (~np.isnan(omni.values.astype(float))).astype(float)

results = {}
for period, g in lab.groupby('period'):
    g = g.sort_values('timedelta'); seq = g['dst'].values.astype(float)
    sm = (~np.isnan(seq)).astype(float); sf = np.nan_to_num(seq)
    cross = fftconvolve(of, sf[::-1], mode='valid')
    b2 = fftconvolve(of ** 2, sm[::-1], mode='valid')
    cnt = fftconvolve(om, sm[::-1], mode='valid')
    sse = np.sum(sf ** 2) + b2 - 2 * cross
    valid = cnt > 0.8 * sm.sum(); sse[~valid] = np.inf
    rmse = np.sqrt(sse / np.maximum(cnt, 1))
    best = int(np.argmin(rmse)); start = omni.index[best]
    second = np.partition(rmse[np.isfinite(rmse)], 1)[1]
    results[period] = start
    print(f'{period}: full-series RMSE={rmse[best]:.3f} nT (2nd {second:.2f}) start={start} '
          f'end={start + g["timedelta"].max()}')
import json; json.dump({k: str(v) for k, v in results.items()}, open('_magnet_dates.json','w'), indent=2)
print('saved _magnet_dates.json')
