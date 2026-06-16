"""Phase M probe (CME-catalog modality feasibility, bucket-2 new modality).
Key insight: by the time CME plasma reaches L1, the in-situ solar wind ALREADY shows it,
so the catalog's ONLY added value is LEAD TIME -- a halo CME launched 1-4 days before a
storm is visible at the Sun before the plasma (and thus the storm) hits.
Probe tests whether earth-directed CME launches predict storms the 48h solar-wind window
cannot yet see. Cheap kill-switch before any GPU spend (mirrors Phase L SYM-H probe).

Sources:
  CME = CDAW SOHO/LASCO catalog univ_all.txt (1996+, covers all recovered MagNet dates).
  Dst = MagNet labels mapped to absolute time via _magnet_dates.json (Phase J date recovery).
"""
import os, re, json, time, numpy as np, pandas as pd, requests

CME_URL="https://cdaw.gsfc.nasa.gov/CME_list/UNIVERSAL_ver2/text_ver/univ_all.txt"
CACHE="_cme_catalog.pkl"
AU_KM=1.496e8
DATE_RE=re.compile(r'^\d{4}/\d{2}/\d{2}$')

# ---- fetch + parse CME catalog ----
if os.path.exists(CACHE):
    cme=pd.read_pickle(CACHE)
else:
    print('downloading univ_all.txt ...',flush=True)
    for k in range(4):
        try:
            r=requests.get(CME_URL,timeout=300); r.raise_for_status(); break
        except Exception as e:
            if k==3: raise
            print(' retry',k+1,type(e).__name__,flush=True); time.sleep(3*(k+1))
    rows=[]
    for ln in r.text.splitlines():
        t=ln.split()
        if len(t)<5 or not DATE_RE.match(t[0]): continue
        try: dt=pd.Timestamp(f"{t[0].replace('/','-')} {t[1]}")
        except Exception: continue
        cpa=t[2]; halo=(cpa.lower()=='halo')
        def num(x):
            try: return float(x)
            except Exception: return np.nan
        width=360.0 if halo else num(t[3])
        speed=num(t[4])
        rows.append((dt,halo,width,speed))
    cme=pd.DataFrame(rows,columns=['t','halo','width','speed']).dropna(subset=['speed'])
    cme=cme[cme['speed']>0].sort_values('t').reset_index(drop=True)
    cme.to_pickle(CACHE)
print(f'CME events parsed: {len(cme)}  halo {cme.halo.sum()}  partial(w>=120) {(cme.width>=120).sum()}')

# earth-directed = full halo OR partial halo width>=120
ed=cme[(cme.halo)|(cme.width>=120)].copy()
ed['arr']=ed['t']+pd.to_timedelta(AU_KM/ed['speed'],unit='s')  # ballistic L1 arrival est
print(f'earth-directed CMEs: {len(ed)}  speed med {ed.speed.median():.0f} km/s')

# ---- Dst absolute-time series from MagNet ----
dl=pd.read_csv('archive/labels.csv'); dl['timedelta']=pd.to_timedelta(dl['timedelta'])
dates={k:pd.Timestamp(v).tz_localize(None) for k,v in json.load(open('_magnet_dates.json')).items()}
parts=[]
for p,g in dl.groupby('period'):
    if p not in dates: continue
    g=g.copy(); g['abs']=dates[p]+g['timedelta']; g['period']=p
    parts.append(g[['abs','dst','period']])
D=pd.concat(parts).dropna(subset=['dst']).set_index('abs').sort_index()
D=D[~D.index.duplicated()]
print(f'\nDst hours: {len(D)}  span {D.index.min()} .. {D.index.max()}')

cme_t=ed['t'].values.astype('datetime64[ns]')
arr_t=ed['arr'].values.astype('datetime64[ns]')
hours=D.index.values.astype('datetime64[ns]')

def flag_recent_launch(lo_h,hi_h):
    """1 if any earth-directed CME LAUNCHED between lo_h..hi_h hours before this Dst hour."""
    f=np.zeros(len(hours),bool)
    for i,h in enumerate(hours):
        lo=h-np.timedelta64(hi_h,'h'); hi=h-np.timedelta64(lo_h,'h')
        f[i]=np.any((cme_t>=lo)&(cme_t<=hi))
    return f

def flag_arrival_window(win_h):
    """1 if predicted ballistic L1 arrival of any earth-directed CME falls within +/-win_h."""
    f=np.zeros(len(hours),bool)
    for i,h in enumerate(hours):
        f[i]=np.any(np.abs((arr_t-h)/np.timedelta64(1,'h'))<=win_h)
    return f

dst=D['dst'].values
def report(name,flag,thr=-50):
    base=(dst<thr).mean()
    if flag.sum()==0: print(f'  {name}: no flagged hours'); return
    p_storm=(dst[flag]<thr).mean()
    lift=p_storm/base if base>0 else float('nan')
    print(f'  {name}: flagged {flag.mean()*100:4.1f}% hours | P(Dst<{thr}|flag) {p_storm*100:5.2f}% '
          f'vs base {base*100:4.2f}% | lift {lift:4.2f}x | meanDst flag {dst[flag].mean():6.1f} vs {dst[~flag].mean():6.1f}')

print('\n=== (A) recent earth-directed CME launch -> storm lift (the LEAD signal) ===')
for lo,hi in [(12,72),(24,96),(36,108),(12,120)]:
    for thr in (-50,-80):
        report(f'launch {lo}-{hi}h before, Dst<{thr}',flag_recent_launch(lo,hi),thr)

print('\n=== (B) ballistic arrival-window alignment ===')
for w in (12,24):
    for thr in (-50,-80):
        report(f'arrival +/-{w}h, Dst<{thr}',flag_arrival_window(w),thr)

print('\nDecision: strong lift (>~2x) at usable lead (24-96h before) = CME adds info the '
      '48h solar-wind window lacks -> proceed to full build. Lift ~1x = no marginal lead signal.')
