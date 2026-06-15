"""Build paired sequence tensors: base (29 feat) and aug (32 feat = +Kp/ap/AE),
identical rows/windows/splits. Leakage-safe cadence lag on indices.
Saves _seq_base.npz and _seq_aug.npz."""
import json, numpy as np, pandas as pd

SOLAR='archive/solar_wind.csv'; LABELS='archive/labels.csv'; SUN='archive/sunspots.csv'
SEQ_LEN=48; HORIZON=6; TRAIN=0.7; VAL=0.2
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

# ---- merge Kp/ap/AE with cadence lag ----
idx=pd.read_pickle('_omni_indices_1998_2019.pkl')   # hourly, tz-aware UTC
idx.index=idx.index.tz_localize(None)
# leakage-safe lag: value at hour t uses only completed intervals strictly before t
idx_lag=pd.DataFrame(index=idx.index)
idx_lag['ae']=idx['AE1800'].shift(1)        # 1h cadence -> lag 1h
idx_lag['kp']=idx['KP1800'].shift(3)/10.0   # 3h block -> lag 3h; undo *10
idx_lag['ap']=idx['AP_INDEX1800'].shift(3)  # 3h block -> lag 3h

# attach absolute time per row, map indices
hourly=hourly.reset_index()
hourly['abs_time']=hourly.apply(lambda r: DATES[r['period']]+r['timedelta'],axis=1)
merged=hourly.merge(idx_lag,left_on='abs_time',right_index=True,how='left')
print('index NaN after merge:',merged[['ae','kp','ap']].isna().mean().round(4).to_dict())
merged=merged.set_index(['period','timedelta']).drop(columns='abs_time')

# join labels, interp
data=merged.join(dl,how='inner')
data[['ae','kp','ap']]=data.groupby('period')[['ae','kp','ap']].transform(lambda x:x.interpolate(limit_direction='both').ffill().bfill())
data=data.interpolate(method='linear',limit_direction='both').ffill().bfill()

base_cols=[c for c in data.columns if c not in ('dst','ae','kp','ap')]
aug_cols=base_cols+['ae','kp','ap']
print(f'base feats {len(base_cols)}, aug feats {len(aug_cols)}')

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

for tag,cols in [('base',base_cols),('aug',aug_cols)]:
    Xtr,Ytr,Xv,Yv,Xte,Yte=make_seq(cols)
    np.savez(f'_seq_{tag}.npz',Xtr=Xtr,Ytr=Ytr,Xv=Xv,Yv=Yv,Xte=Xte,Yte=Yte)
    print(f'{tag}: train{Xtr.shape} val{Xv.shape} test{Xte.shape}')
print('done')
