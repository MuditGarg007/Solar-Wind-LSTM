"""Leakage-robustness: rebuild aug indices with EXTRA +6h lag (AE 7h, Kp/ap 9h).
If t+1 intense gain persists -> real signal, not boundary leakage. 3 seeds."""
import json, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

SEQ_LEN=48; H=6; TRAIN=0.7; VAL=0.2
DATES={k: pd.Timestamp(v).tz_localize(None) for k,v in json.load(open('_magnet_dates.json')).items()}
RAW=['bx_gse','by_gse','bz_gse','bx_gsm','by_gsm','bz_gsm','theta_gse','phi_gse',
     'theta_gsm','phi_gsm','bt','density','speed','temperature']
ds=pd.read_csv('archive/solar_wind.csv'); dl=pd.read_csv('archive/labels.csv'); dsun=pd.read_csv('archive/sunspots.csv')
for d in (ds,dl,dsun): d['timedelta']=pd.to_timedelta(d['timedelta'])
ds.set_index(['period','timedelta'],inplace=True); dl.set_index(['period','timedelta'],inplace=True); dsun.set_index(['period','timedelta'],inplace=True)
g=ds.groupby(['period',pd.Grouper(level='timedelta',freq='1h')])
mean=g[RAW].mean(); mean.columns=[f'{c}_mean' for c in RAW]
std=g[RAW].std(); std.columns=[f'{c}_std' for c in RAW]
hourly=pd.concat([mean,std],axis=1)
ssn=dsun.groupby(['period',pd.Grouper(level='timedelta',freq='1h')])['smoothed_ssn'].mean()
hourly=hourly.join(ssn,how='left')
hourly['smoothed_ssn']=hourly.groupby('period')['smoothed_ssn'].transform(lambda x:x.ffill().bfill())
idx=pd.read_pickle('_omni_indices_1998_2019.pkl'); idx.index=idx.index.tz_localize(None)
il=pd.DataFrame(index=idx.index)
il['ae']=idx['AE1800'].shift(1+6)         # extra 6h
il['kp']=idx['KP1800'].shift(3+6)/10.0    # extra 6h
il['ap']=idx['AP_INDEX1800'].shift(3+6)
hourly=hourly.reset_index(); hourly['abs_time']=hourly.apply(lambda r:DATES[r['period']]+r['timedelta'],axis=1)
m=hourly.merge(il,left_on='abs_time',right_index=True,how='left').set_index(['period','timedelta']).drop(columns='abs_time')
data=m.join(dl,how='inner')
data[['ae','kp','ap']]=data.groupby('period')[['ae','kp','ap']].transform(lambda x:x.interpolate(limit_direction='both').ffill().bfill())
data=data.interpolate(method='linear',limit_direction='both').ffill().bfill()
cols=[c for c in data.columns if c!='dst']
Xtr,Ytr,Xv,Yv,Xte,Yte=[],[],[],[],[],[]
for p in data.index.get_level_values('period').unique():
    grp=data.loc[p]; ax=grp[cols].values; ay=grp['dst'].values
    Xl,Yl=[],[]
    for i in range(len(grp)-SEQ_LEN-H+1): Xl.append(ax[i:i+SEQ_LEN]); Yl.append(ay[i+SEQ_LEN:i+SEQ_LEN+H])
    Xp=np.array(Xl);Yp=np.array(Yl);n=len(Xp);ntr=int(n*TRAIN);nvl=int(n*VAL)
    Xtr.append(Xp[:ntr]);Ytr.append(Yp[:ntr]);Xv.append(Xp[ntr:ntr+nvl]);Yv.append(Yp[ntr:ntr+nvl]);Xte.append(Xp[ntr+nvl:]);Yte.append(Yp[ntr+nvl:])
Xtr=np.concatenate(Xtr);Ytr=np.concatenate(Ytr);Xv=np.concatenate(Xv);Yv=np.concatenate(Yv);Xte=np.concatenate(Xte);Yte=np.concatenate(Yte)
print('aug_lag feats',Xtr.shape[2])

dev='cuda'; HID=128;NL=2;DO=0.4;BS=64;LR=1e-3;WD=5e-5;EP=50;PAT=15
class Net(nn.Module):
    def __init__(s,d):
        super().__init__(); s.lstm=nn.LSTM(d,HID,NL,batch_first=True,dropout=DO,bidirectional=True)
        s.att=nn.Linear(HID*2,1); s.fc=nn.Linear(HID*2*2,H)
    def forward(s,x):
        o,_=s.lstm(x);w=F.softmax(s.att(o),dim=1);return s.fc(torch.cat((torch.sum(o*w,1),o[:,-1,:]),1))
sc=StandardScaler();f=Xtr.shape[2]
Xtr_s=sc.fit_transform(Xtr.reshape(-1,f)).reshape(Xtr.shape)
Xv_s=sc.transform(Xv.reshape(-1,f)).reshape(Xv.shape); Xte_s=sc.transform(Xte.reshape(-1,f)).reshape(Xte.shape)
base=pd.read_csv('geomag_index_ab.csv')
print('seed | aug_lag t+1 intense | base t+1 intense (CSV) | aug(orig lag)')
for seed in [42,1,7]:
    torch.manual_seed(seed);np.random.seed(seed);torch.cuda.manual_seed_all(seed)
    mdl=Net(f).to(dev);opt=torch.optim.Adam(mdl.parameters(),lr=LR,weight_decay=WD);mse=nn.MSELoss()
    tl=DataLoader(TensorDataset(torch.FloatTensor(Xtr_s),torch.FloatTensor(Ytr)),batch_size=BS,shuffle=True)
    vl=DataLoader(TensorDataset(torch.FloatTensor(Xv_s),torch.FloatTensor(Yv)),batch_size=BS)
    bestv=1e9;cnt=0;bsd=None
    for ep in range(EP):
        mdl.train()
        for xb,yb in tl:
            xb,yb=xb.to(dev),yb.to(dev);opt.zero_grad();out=mdl(xb);le=(out-yb)**2
            w=torch.ones_like(yb);w[yb<-20]=5.0;w[yb<-50]=15.0;(le*w).mean().backward()
            nn.utils.clip_grad_norm_(mdl.parameters(),1.0);opt.step()
        mdl.eval();vb=[]
        with torch.no_grad():
            for xv,yv in vl: xv,yv=xv.to(dev),yv.to(dev);vb.append(mse(mdl(xv),yv).item())
        v=np.mean(vb)
        if v<bestv:bestv=v;cnt=0;bsd={k:t.clone() for k,t in mdl.state_dict().items()}
        else:
            cnt+=1
            if cnt>=PAT:break
    mdl.load_state_dict(bsd);mdl.eval();out=[]
    with torch.no_grad():
        for (xb,) in DataLoader(TensorDataset(torch.FloatTensor(Xte_s)),batch_size=256): out.append(mdl(xb.to(dev)).cpu().numpy())
    pred=np.concatenate(out)
    yh=Yte[:,0];eh=pred[:,0]-Yte[:,0];lag_t1=float(np.sqrt(np.mean(eh[yh<-50]**2)))
    bsel=base[(base.seed==seed)&(base.horizon=='t+1')&(base.regime=='intense')]
    aug=base[(base.tag=='aug')&(base.seed==seed)&(base.horizon=='t+1')&(base.regime=='intense')].rmse.iloc[0]
    bb=bsel[bsel.tag=='base'].rmse.iloc[0]
    print(f'{seed} | {lag_t1:.2f} | {bb:.2f} | {aug:.2f}   (aug_lag-base delta {bb-lag_t1:+.2f})',flush=True)
