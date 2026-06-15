"""Phase L build: paired sequences on train_a/b/c with TWO targets on identical windows:
  Ydst  = MagNet hourly Dst (control)
  Ysymh = OMNI hourly-MIN SYM-H (within-hour storm peak; +16-30% more extreme labels)
29 base features. Per-period 70/20/10 chronological split. Saves _seq_symh.npz.
"""
import json, numpy as np, pandas as pd
SOLAR='archive/solar_wind.csv'; LABELS='archive/labels.csv'; SUN='archive/sunspots.csv'
SEQ_LEN=48; HORIZON=6; TRAIN=0.7; VAL=0.2
DATES={k:pd.Timestamp(v).tz_localize(None) for k,v in json.load(open('_magnet_dates.json')).items()}
PERIODS=list(DATES.keys())  # train_a/b/c only
print('periods',PERIODS)

RAW=['bx_gse','by_gse','bz_gse','bx_gsm','by_gsm','bz_gsm','theta_gse','phi_gse',
     'theta_gsm','phi_gsm','bt','density','speed','temperature']
ds=pd.read_csv(SOLAR); dl=pd.read_csv(LABELS); dsun=pd.read_csv(SUN)
for d in (ds,dl,dsun): d['timedelta']=pd.to_timedelta(d['timedelta'])
# restrict to dated periods
ds=ds[ds['period'].isin(PERIODS)]; dl=dl[dl['period'].isin(PERIODS)]; dsun=dsun[dsun['period'].isin(PERIODS)]
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

# hourly-min SYM-H aligned to (period,timedelta)
symh=pd.read_pickle('_omni_symh_1min.pkl')
if symh.index.tz is not None: symh.index=symh.index.tz_localize(None)
symh_min=symh.resample('1h').min()
hourly=hourly.reset_index()
hourly['abs']=hourly.apply(lambda r:DATES[r['period']]+r['timedelta'],axis=1)
hourly=hourly.merge(symh_min.rename('symh_min'),left_on='abs',right_index=True,how='left')
print('symh_min NaN after merge',round(hourly['symh_min'].isna().mean(),4))
hourly=hourly.set_index(['period','timedelta']).drop(columns='abs')

data=hourly.join(dl,how='inner')   # adds dst
data['symh_min']=data.groupby('period')['symh_min'].transform(lambda x:x.interpolate(limit_direction='both').ffill().bfill())
data=data.interpolate(method='linear',limit_direction='both').ffill().bfill()

feat_cols=[c for c in data.columns if c not in ('dst','symh_min')]
print('feats',len(feat_cols))
assert len(feat_cols)==29, feat_cols

Xtr,Xv,Xte=[],[],[]
Ydtr,Ydv,Ydte=[],[],[]
Ystr,Ysv,Yste=[],[],[]
for p in data.index.get_level_values('period').unique():
    grp=data.loc[p]
    if len(grp)<=SEQ_LEN+HORIZON: continue
    ax=grp[feat_cols].values; ad=grp['dst'].values; asy=grp['symh_min'].values
    Xl,Ydl,Ysl=[],[],[]
    for i in range(len(grp)-SEQ_LEN-HORIZON+1):
        Xl.append(ax[i:i+SEQ_LEN])
        Ydl.append(ad[i+SEQ_LEN:i+SEQ_LEN+HORIZON])
        Ysl.append(asy[i+SEQ_LEN:i+SEQ_LEN+HORIZON])
    Xp=np.array(Xl); Ydp=np.array(Ydl); Ysp=np.array(Ysl)
    n=len(Xp); ntr=int(n*TRAIN); nvl=int(n*VAL)
    Xtr.append(Xp[:ntr]); Xv.append(Xp[ntr:ntr+nvl]); Xte.append(Xp[ntr+nvl:])
    Ydtr.append(Ydp[:ntr]); Ydv.append(Ydp[ntr:ntr+nvl]); Ydte.append(Ydp[ntr+nvl:])
    Ystr.append(Ysp[:ntr]); Ysv.append(Ysp[ntr:ntr+nvl]); Yste.append(Ysp[ntr+nvl:])

cat=np.concatenate
np.savez('_seq_symh.npz',
    Xtr=cat(Xtr),Xv=cat(Xv),Xte=cat(Xte),
    Ydst_tr=cat(Ydtr),Ydst_v=cat(Ydv),Ydst_te=cat(Ydte),
    Ysymh_tr=cat(Ystr),Ysymh_v=cat(Ysv),Ysymh_te=cat(Yste),
    feat_cols=np.array(feat_cols))
print('train',cat(Xtr).shape,'val',cat(Xv).shape,'test',cat(Xte).shape)
print('saved _seq_symh.npz')
# sanity: extreme-label counts (any-horizon) dst vs symh on TEST
for tag,Y in [('dst',cat(Ydte)),('symh_min',cat(Yste))]:
    for thr in (-50,-80,-100):
        print(f'  test rows any-h {tag}<{thr}: {int((Y<thr).any(1).sum())}')
