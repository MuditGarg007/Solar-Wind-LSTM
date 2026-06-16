"""Phase L probe (SYM-H target feasibility, next-step bucket-2).
Fetch OMNI 1-min SYM-H for recovered MagNet date ranges (train_a/b/c), then measure:
  (1) redundancy: hourly-mean SYM-H vs MagNet Dst (corr/RMSE) - validates date lock + tells if relabel is trivial.
  (2) resolution payoff: how much deeper 1-min SYM-H minima go than hourly Dst on storms,
      and how many extra storm-threshold crossings SYM-H exposes that hourly Dst smooths away.
Decision: payoff large -> SYM-H higher-cadence target worth full retrain (and which cadence). Small -> reject.
"""
import requests, os, json, time, numpy as np, pandas as pd
CDAS="https://cdaweb.gsfc.nasa.gov/WS/cdasr/1/dataviews/sp_phys/datasets/OMNI_HRO_1MIN/data"
CACHE="_omni_symh_1min.pkl"

def fetch_window(t0,t1,tries=4):
    """t0,t1 pandas Timestamps. monthly-ish window. retry on transient 503/timeout."""
    url=f"{CDAS}/{t0.strftime('%Y%m%dT%H%M%SZ')},{t1.strftime('%Y%m%dT%H%M%SZ')}/SYM_H/?format=json"
    for k in range(tries):
        try:
            r=requests.get(url,headers={'Accept':'application/json'},timeout=300); r.raise_for_status()
            j=r.json(); ep=None; val=None
            cdf=j.get('CDF')
            if not cdf: return pd.Series(dtype=float,name='symh')
            for v in cdf[0]['cdfVariables']['variable']:
                nm=v.get('name')
                if nm=='Epoch': ep=[rec['value'][0] for rec in v['cdfVarData']['record']]
                elif nm=='SYM_H': val=[float(rec['value'][0]) for rec in v['cdfVarData']['record']]
            if ep is None: return pd.Series(dtype=float,name='symh')
            return pd.Series(val,index=pd.to_datetime(ep),name='symh')
        except Exception as e:
            if k==tries-1: raise
            print('   retry',k+1,type(e).__name__,flush=True); time.sleep(3*(k+1))

if os.path.exists(CACHE):
    symh=pd.read_pickle(CACHE)
else:
    parts=[]
    months=pd.date_range('1998-01-01','2020-01-01',freq='MS')
    for i in range(len(months)-1):
        t0,t1=months[i],months[i+1]
        print(' fetch',t0.strftime('%Y-%m'),flush=True)
        s=fetch_window(t0,t1)
        if len(s): parts.append(s)
    symh=pd.concat(parts)
    symh=symh[~symh.index.duplicated()].sort_index()
    symh[symh>=99999]=np.nan   # OMNI fill
    symh.to_pickle(CACHE)
print('symh 1-min rows',len(symh),'NaN frac',round(symh.isna().mean(),4))

dl=pd.read_csv('archive/labels.csv'); dl['timedelta']=pd.to_timedelta(dl['timedelta'])
dates={k:pd.Timestamp(v).tz_localize(None) for k,v in json.load(open('_magnet_dates.json')).items()}
symh.index=symh.index.tz_localize(None) if symh.index.tz is not None else symh.index

allrows=[]
for p,g in dl.groupby('period'):
    if p not in dates: continue
    start=dates[p]
    g=g.copy(); g['abs']=start+g['timedelta']
    lo,hi=g['abs'].min(),g['abs'].max()+pd.Timedelta('1h')
    sm=symh[(symh.index>=lo)&(symh.index<hi)]
    if sm.empty: print(p,'NO symh'); continue
    # hourly aggregates aligned to label hours
    h_mean=sm.resample('1h').mean(); h_min=sm.resample('1h').min()
    df=g.set_index('abs')[['dst']].join(h_mean.rename('symh_mean')).join(h_min.rename('symh_min'))
    df['period']=p
    allrows.append(df)
M=pd.concat(allrows).dropna(subset=['dst','symh_mean'])
print('\nmerged hours',len(M))

# (1) redundancy
c=np.corrcoef(M['dst'],M['symh_mean'])[0,1]
rmse=np.sqrt(((M['dst']-M['symh_mean'])**2).mean())
bias=(M['symh_mean']-M['dst']).mean()
print(f"(1) hourly-mean SYM-H vs Dst: corr {c:.4f}  RMSE {rmse:.2f} nT  bias {bias:+.2f}")

# (2) resolution payoff on storms
for thr in (-50,-80,-100):
    st=M[M['dst']<thr]
    if len(st)==0: print(f'  no Dst<{thr}'); continue
    extra=(st['dst']-st['symh_min'])   # how much deeper 1-min min vs hourly Dst (positive=deeper)
    print(f"(2) storm hours Dst<{thr}: N={len(st)}  mean extra depth(symh_min vs Dst) {extra.mean():+.1f} nT  max {extra.max():+.1f}")
# extra threshold crossings exposed by 1-min resolution
print("\n(2b) threshold crossings (count of units below thr):")
for thr in (-50,-80,-100,-150,-200):
    n_dst_h=(M['dst']<thr).sum()
    n_symh_h=(M['symh_min']<thr).sum()              # hours whose 1-min min dips below thr
    n_symh_1m=int((symh<thr).sum())                  # raw 1-min samples below thr (whole record)
    print(f"  <{thr}: Dst hours {n_dst_h} | SYM-H-min hours {n_symh_h} (+{n_symh_h-n_dst_h}) | 1-min samples {n_symh_1m}")
M.to_pickle('_symh_aligned.pkl')
print('\nsaved _symh_aligned.pkl')
