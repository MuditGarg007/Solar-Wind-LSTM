"""Phase M Step 2-4: build paired seq tensors base(29) vs cme(37 = +8 solar-source feats).
Identical rows/windows/splits to base. Leakage-safe: per-hour features use ONLY events with
timestamp < t (small ground-processing lag); arrival-expectation bumps are centered at predicted
arrival (always AFTER launch, evaluated only at hours >= launch -> causal by construction).
Features live in INPUT WINDOW only (create_sequences) -> never touch target hours.
Saves _seq_cme_base.npz and _seq_cme.npz.

8 new feats: cme_hrs_since, cme_last_speed, cme_n72h, cme_arr_prox,
             flare_hrs_since_M, flare_logE24h, f107, f107_trend.
"""
import json, numpy as np, pandas as pd

SOLAR='archive/solar_wind.csv'; LABELS='archive/labels.csv'; SUN='archive/sunspots.csv'
SEQ_LEN=48; HORIZON=6; TRAIN=0.7; VAL=0.2
AU_KM=1.496e8; LAG_H=2          # ground-processing lag (CME/flare known ~2h after observation)
CAP=240.0                        # hours-since cap (10 days)
DATES={k: pd.Timestamp(v).tz_localize(None) for k,v in json.load(open('_magnet_dates.json')).items()}
print('recovered starts:', {k:str(v) for k,v in DATES.items()})

# ---- base 29-feature build (replicate notebook load_and_preprocess) ----
RAW=['bx_gse','by_gse','bz_gse','bx_gsm','by_gsm','bz_gsm','theta_gse','phi_gse',
     'theta_gsm','phi_gsm','bt','density','speed','temperature']
ds=pd.read_csv(SOLAR); dl=pd.read_csv(LABELS); dsun=pd.read_csv(SUN)
for d in (ds,dl,dsun): d['timedelta']=pd.to_timedelta(d['timedelta'])
ds.set_index(['period','timedelta'],inplace=True)
dl.set_index(['period','timedelta'],inplace=True)
dsun.set_index(['period','timedelta'],inplace=True)
g=ds.groupby(['period',pd.Grouper(level='timedelta',freq='1h')])
mean=g[RAW].mean(); mean.columns=[f'{c}_mean' for c in RAW]
std=g[RAW].std(); std.columns=[f'{c}_std' for c in RAW]
hourly=pd.concat([mean,std],axis=1)
ssn=dsun.groupby(['period',pd.Grouper(level='timedelta',freq='1h')])['smoothed_ssn'].mean()
hourly=hourly.join(ssn,how='left')
hourly['smoothed_ssn']=hourly.groupby('period')['smoothed_ssn'].transform(lambda x:x.ffill().bfill())
hourly=hourly.reset_index()
hourly['abs_time']=hourly.apply(lambda r: DATES[r['period']]+r['timedelta'],axis=1)

# ---- load solar-source caches ----
cme=pd.read_pickle('_cme_catalog.pkl')
ed=cme[(cme.halo)|(cme.width>=120)].copy().sort_values('t').reset_index(drop=True)
ed_t=ed['t'].values.astype('datetime64[ns]')
ed_spd=ed['speed'].values.astype(float)
ed_arr=(ed['t']+pd.to_timedelta(AU_KM/ed['speed'],unit='s')).values.astype('datetime64[ns]')
fl=pd.read_pickle('_goes_flares.pkl')
flM=fl[fl.flux>=1e-5].copy().sort_values('t').reset_index(drop=True)   # M+ flares
flM_t=flM['t'].values.astype('datetime64[ns]')
fl_t=fl['t'].values.astype('datetime64[ns]')
fl_flux=fl['flux'].values.astype(float)
f107=pd.read_pickle('_f107.pkl')                                      # daily series
f107_tr=f107 - f107.rolling(27,min_periods=1,center=False).mean()     # rotation-phase trend

H1=np.timedelta64(1,'h')
def hrs_since(events_t, hours_lagged):
    """hours since most recent event strictly before each (already-lagged) hour; CAP if none."""
    idx=np.searchsorted(events_t, hours_lagged, side='right')-1
    out=np.full(len(hours_lagged),CAP)
    ok=idx>=0
    out[ok]=np.minimum((hours_lagged[ok]-events_t[idx[ok]])/H1, CAP)
    return out
def count_window(events_t, hours_lagged, win_h):
    hi=np.searchsorted(events_t, hours_lagged, side='right')
    lo=np.searchsorted(events_t, hours_lagged-np.timedelta64(win_h,'h'), side='left')
    return (hi-lo).astype(float)
def last_speed(hours_lagged):
    idx=np.searchsorted(ed_t, hours_lagged, side='right')-1
    out=np.zeros(len(hours_lagged)); ok=idx>=0
    out[ok]=ed_spd[idx[ok]]
    return out
def flare_logE(hours_lagged, win_h=24):
    hi=np.searchsorted(fl_t, hours_lagged, side='right')
    lo=np.searchsorted(fl_t, hours_lagged-np.timedelta64(win_h,'h'), side='left')
    cum=np.concatenate([[0.0],np.cumsum(fl_flux)])
    s=cum[hi]-cum[lo]
    return np.log10(s+1e-9)
def arr_prox(hours):
    """max over ED-CMEs of exp(-((h-t_arr)/sig)^2/2); causal: t_arr>t_launch>=eval hour region."""
    sig=12.0; out=np.zeros(len(hours))
    order=np.argsort(ed_arr); A=ed_arr[order]
    for j in range(len(A)):
        a=A[j]
        lo=np.searchsorted(hours, a-np.timedelta64(36,'h'),'left')
        hi=np.searchsorted(hours, a+np.timedelta64(36,'h'),'right')
        if hi<=lo: continue
        d=(hours[lo:hi]-a)/H1
        out[lo:hi]=np.maximum(out[lo:hi], np.exp(-(d/sig)**2/2))
    return out

# ---- compute features per period (abs_time-aligned, leakage-lagged) ----
hours_all=hourly['abs_time'].values.astype('datetime64[ns]')
lag=np.timedelta64(LAG_H,'h')
hl=hours_all-lag
hourly['cme_hrs_since']=hrs_since(ed_t, hl)
hourly['cme_last_speed']=last_speed(hl)
hourly['cme_n72h']=count_window(ed_t, hl, 72)
hourly['cme_arr_prox']=arr_prox(hours_all)         # bump is causal (centered post-launch)
hourly['flare_hrs_since_M']=hrs_since(flM_t, hl)
hourly['flare_logE24h']=flare_logE(hl)
# f107: daily, use value from PRIOR day (finalized) -> lag 1 day
fday=(pd.DatetimeIndex(hours_all).normalize()-pd.Timedelta('1D'))
hourly['f107']=f107.reindex(fday).values
hourly['f107_trend']=f107_tr.reindex(fday).values

NEW=['cme_hrs_since','cme_last_speed','cme_n72h','cme_arr_prox',
     'flare_hrs_since_M','flare_logE24h','f107','f107_trend']
print('new-feat NaN frac:',hourly[NEW].isna().mean().round(4).to_dict())

merged=hourly.set_index(['period','timedelta']).drop(columns='abs_time')
data=merged.join(dl,how='inner')
# fill: f107 ffill within period; flares/cme already 0/CAP-filled; final safety interp
data[NEW]=data.groupby('period')[NEW].transform(lambda x:x.ffill().bfill())
data=data.interpolate(method='linear',limit_direction='both').ffill().bfill()

base_cols=[c for c in data.columns if c not in ['dst']+NEW]
cme_cols=base_cols+NEW
print(f'base feats {len(base_cols)}, cme feats {len(cme_cols)}')

def make_seq(cols):
    Xtr,Ytr,Xv,Yv,Xte,Yte=[],[],[],[],[],[]
    for period in data.index.get_level_values('period').unique():
        grp=data.loc[period]
        if len(grp)<=SEQ_LEN+HORIZON: continue
        ax=grp[cols].values; ay=grp['dst'].values
        Xl,Yl=[],[]
        for i in range(len(grp)-SEQ_LEN-HORIZON+1):
            Xl.append(ax[i:i+SEQ_LEN]); Yl.append(ay[i+SEQ_LEN:i+SEQ_LEN+HORIZON])
        Xp=np.array(Xl); Yp=np.array(Yl); n=len(Xp); ntr=int(n*TRAIN); nvl=int(n*VAL)
        Xtr.append(Xp[:ntr]); Ytr.append(Yp[:ntr])
        Xv.append(Xp[ntr:ntr+nvl]); Yv.append(Yp[ntr:ntr+nvl])
        Xte.append(Xp[ntr+nvl:]); Yte.append(Yp[ntr+nvl:])
    return (np.concatenate(Xtr),np.concatenate(Ytr),np.concatenate(Xv),
            np.concatenate(Yv),np.concatenate(Xte),np.concatenate(Yte))

for tag,cols in [('cme_base',base_cols),('cme',cme_cols)]:
    Xtr,Ytr,Xv,Yv,Xte,Yte=make_seq(cols)
    np.savez(f'_seq_{tag}.npz',Xtr=Xtr,Ytr=Ytr,Xv=Xv,Yv=Yv,Xte=Xte,Yte=Yte)
    print(f'{tag}: train{Xtr.shape} val{Xv.shape} test{Xte.shape}')
np.save('_cme_newfeat_names.npy',np.array(NEW))
print('done')
