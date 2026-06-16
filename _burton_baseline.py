"""Bucket-3 context: physics baseline (Burton coupling model, O'Brien-McPherron 2000 refinement).
Contextualizes ML gain vs the classic empirical ring-current ODE.

Model (OM2000):
  Dst* = Dst - b*sqrt(Pdyn) + c              (pressure-corrected ring current)   b=7.26 c=11
  dDst*/dt = Q(t) - Dst*/tau(t)
  Q   = -4.4*(VBs-0.49) nT/h  if VBs>=0.49 else 0     (VBs = V*Bs/1000 in mV/m, Bs=max(0,-Bz_gsm))
  tau = 2.40*exp(9.74/(4.69+VBs)) h
  Pdyn[nPa] = 1.6726e-6 * n[cm^-3] * V[km/s]^2

Two modes, both on held-out TEST segment (last 10% hours per period, mirrors create_sequences split):
  (A) HINDCAST: integrate with OBSERVED driver -> coupling accuracy ceiling (perfect inputs, not a forecast).
  (B) FORECAST t+1..t+6: at each origin t take Dst*(t) from observed-driver integration up to t (data<=t only),
      then integrate forward holding VBs,Pdyn frozen at their t-values. Real physics forecaster.
Also Dst-persistence on same test hours. Writes burton_baseline.csv.
"""
import numpy as np, pandas as pd
SOLAR='archive/solar_wind.csv'; LABELS='archive/labels.csv'
SEQ_LEN=48; H=6; TRAIN=0.7; VAL=0.2
b_p=7.26; c_p=11.0          # pressure correction
Ec=0.49                      # mV/m threshold

ds=pd.read_csv(SOLAR); dl=pd.read_csv(LABELS)
for d in (ds,dl): d['timedelta']=pd.to_timedelta(d['timedelta'])
ds.set_index(['period','timedelta'],inplace=True); dl.set_index(['period','timedelta'],inplace=True)
g=ds.groupby(['period',pd.Grouper(level='timedelta',freq='1h')])
hourly=g[['bz_gsm','speed','density']].mean()
data=hourly.join(dl,how='inner')
data=data.groupby('period').apply(lambda x:x.interpolate(limit_direction='both').ffill().bfill()).reset_index(level=0,drop=True)

def burton_arrays(bz,V,n):
    Bs=np.maximum(0.0,-bz)
    VBs=V*Bs/1000.0
    Pdyn=1.6726e-6*n*V**2
    Q=np.where(VBs>=Ec,-4.4*(VBs-Ec),0.0)
    tau=2.40*np.exp(9.74/(4.69+VBs))
    return VBs,Pdyn,Q,tau

def integrate_hindcast(dst,Q,tau,Pdyn):
    """observed-driver integration; init Dst* from observed Dst at t0. returns Dst_pred per hour."""
    N=len(dst); ds_star=np.empty(N)
    ds_star[0]=dst[0]-b_p*np.sqrt(max(Pdyn[0],0))+c_p
    for k in range(N-1):
        ds_star[k+1]=ds_star[k]+(Q[k]-ds_star[k]/tau[k])   # dt=1h
    return ds_star+b_p*np.sqrt(np.clip(Pdyn,0,None))-c_p, ds_star

rows=[]
# accumulate forecast errors per horizon, and hindcast errors, with obs Dst for regime split
fc_err={h:[] for h in range(H)}; fc_obs={h:[] for h in range(H)}
hc_err=[]; hc_obs=[]
pe_err={h:[] for h in range(H)}; pe_obs={h:[] for h in range(H)}   # persistence

for period,grp in data.groupby('period'):
    dst=grp['dst'].values.astype(float)
    bz=grp['bz_gsm'].values; V=grp['speed'].values; n=grp['density'].values
    N=len(dst)
    if N<SEQ_LEN+H+10: continue
    VBs,Pdyn,Q,tau=burton_arrays(bz,V,n)
    dst_hc,ds_star=integrate_hindcast(dst,Q,tau,Pdyn)
    # test target hours: mirror make_seq -> origins i in [ntr+nvl .. n_win), target hours i+SEQ_LEN+h
    n_win=N-SEQ_LEN-H+1
    ntr=int(n_win*TRAIN); nvl=int(n_win*VAL)
    test_origins=range(ntr+nvl, n_win)
    for i in test_origins:
        t=i+SEQ_LEN-1            # last input hour = forecast origin
        # hindcast accuracy at the t+6 target hour (perfect-driver nowcast at that hour)
        th6=t+H
        hc_err.append(dst_hc[th6]-dst[th6]); hc_obs.append(dst[th6])
        # forecast: freeze driver at origin t, integrate forward from ds_star[t]
        s=ds_star[t]; Qf=Q[t]; tauf=tau[t]; Pf=Pdyn[t]
        for h in range(H):
            s=s+(Qf-s/tauf)
            pred=s+b_p*np.sqrt(max(Pf,0))-c_p
            fc_err[h].append(pred-dst[t+1+h]); fc_obs[h].append(dst[t+1+h])
            pe_err[h].append(dst[t]-dst[t+1+h]); pe_obs[h].append(dst[t])  # persistence = hold Dst(t)

def rmse(e,mask=None):
    e=np.asarray(e);
    if mask is not None: e=e[mask]
    return float(np.sqrt(np.mean(e**2))) if len(e) else np.nan

def regime_rmse(err,obs,tag,horizon):
    err=np.asarray(err); obs=np.asarray(obs)
    for reg,m in [('agg',np.ones(len(obs),bool)),('intense',obs<-50),
                  ('moderate',(obs>=-50)&(obs<-20)),('quiet',obs>=-20)]:
        rows.append(dict(model=tag,horizon=horizon,regime=reg,n=int(m.sum()),rmse=rmse(err,m)))

# hindcast (perfect driver) at t+6 target hours
regime_rmse(hc_err,hc_obs,'burton_hindcast(perfect driver)','t+6')
# forecast + persistence per horizon
for h in range(H):
    regime_rmse(fc_err[h],fc_obs[h],'burton_forecast(frozen driver)',f't+{h+1}')
    regime_rmse(pe_err[h],pe_obs[h],'persistence',f't+{h+1}')

df=pd.DataFrame(rows); df.to_csv('burton_baseline.csv',index=False)
pd.set_option('display.width',200)
print('=== Burton OM2000 baseline on held-out TEST (compare ML base: t+6 agg 11.26, intense 44.64) ===\n')
for tag in df.model.unique():
    print(f'--- {tag} ---')
    print(df[df.model==tag].pivot(index='horizon',columns='regime',values='rmse')[['agg','moderate','intense','quiet']].round(2).to_string())
    print()
print('N test origins:',len(hc_err))
print('saved burton_baseline.csv')
