"""Download OMNI2 hourly Kp/ap/AE for 1998-2019, cache, report coverage."""
import requests, numpy as np, pandas as pd, os
CDAWEB = 'https://cdaweb.gsfc.nasa.gov/WS/cdasr/1/dataviews/sp_phys/datasets/OMNI2_H0_MRG1HR/data'
WANT = ['KP1800', 'AP_INDEX1800', 'AE1800']
CACHE = '_omni_indices_1998_2019.pkl'

def fetch(start, end):
    url = f'{CDAWEB}/{start}T000000Z,{end}T235959Z/{",".join(WANT)}/?format=json'
    r = requests.get(url, timeout=300, headers={'Accept': 'application/json'}); r.raise_for_status()
    var = r.json()['CDF'][0]['cdfVariables']['variable']
    d = {}
    for v in var:
        nm = v.get('name')
        if nm == 'Epoch':
            d['Epoch'] = [rec['value'][0] for rec in v['cdfVarData']['record']]
        elif nm in WANT:
            d[nm] = [float(rec['value'][0]) for rec in v['cdfVarData']['record']]
    df = pd.DataFrame({k: v for k, v in d.items() if k != 'Epoch'}, index=pd.to_datetime(d['Epoch']))
    return df

if os.path.exists(CACHE):
    idx = pd.read_pickle(CACHE)
else:
    parts = []
    for y in range(1998, 2020, 4):
        y1 = min(y + 4, 2020)
        print(f'  {y}-{y1}...', flush=True)
        parts.append(fetch(f'{y}0101', f'{y1}0101'))
    idx = pd.concat(parts)
    idx = idx[~idx.index.duplicated()].sort_index().asfreq('1h')
    # OMNI fill values: Kp 99, ap 999, AE 9999  (stored *10 for Kp)
    idx.loc[idx['KP1800'] >= 990, 'KP1800'] = np.nan      # Kp*10 fill 990/99
    idx.loc[idx['AP_INDEX1800'] >= 999, 'AP_INDEX1800'] = np.nan
    idx.loc[idx['AE1800'] >= 9999, 'AE1800'] = np.nan
    idx.to_pickle(CACHE)

print(idx.describe())
print('\nNaN fraction by year:')
for nm in WANT:
    fr = idx[nm].isna().groupby(idx.index.year).mean().round(2)
    print(nm, dict(fr[fr > 0.01]))
