"""Phase J (next-step #4): Kp/ap/AE geomag-index INPUTS.
5-seed paired A/B: base(29) vs aug(32=+Kp/ap/AE), identical windows/splits.
Leakage-safe: indices cadence-lagged, input-window only. Held-out test eval.
Writes geomag_index_ab.csv."""
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

dev='cuda' if torch.cuda.is_available() else 'cpu'
HID=128; NL=2; DO=0.4; BS=64; LR=1e-3; WD=5e-5; EP=50; PAT=15; H=6
SEEDS=[42,1,7,123,2024]

b=np.load('_seq_base.npz'); a=np.load('_seq_aug.npz')
Ytr=b['Ytr']; Yv=b['Yv']; Yte=b['Yte']
assert np.array_equal(b['Yte'],a['Yte'])  # paired

class Net(nn.Module):
    def __init__(s,d):
        super().__init__()
        s.lstm=nn.LSTM(d,HID,NL,batch_first=True,dropout=DO,bidirectional=True)
        s.att=nn.Linear(HID*2,1); s.fc=nn.Linear(HID*2*2,H)
    def forward(s,x):
        o,_=s.lstm(x); w=F.softmax(s.att(o),dim=1)
        ctx=torch.sum(o*w,dim=1); last=o[:,-1,:]
        return s.fc(torch.cat((ctx,last),dim=1))

def scale(Xtr,Xv,Xte):
    sc=StandardScaler(); n,t,f=Xtr.shape
    a=sc.fit_transform(Xtr.reshape(-1,f)).reshape(n,t,f)
    v=sc.transform(Xv.reshape(-1,f)).reshape(Xv.shape)
    e=sc.transform(Xte.reshape(-1,f)).reshape(Xte.shape)
    return a,v,e

def train(Xtr,Xv,seed,f):
    torch.manual_seed(seed); np.random.seed(seed)
    if dev=='cuda': torch.cuda.manual_seed_all(seed)
    m=Net(f).to(dev); opt=torch.optim.Adam(m.parameters(),lr=LR,weight_decay=WD)
    mse=nn.MSELoss()
    tl=DataLoader(TensorDataset(torch.FloatTensor(Xtr),torch.FloatTensor(Ytr)),batch_size=BS,shuffle=True)
    vl=DataLoader(TensorDataset(torch.FloatTensor(Xv),torch.FloatTensor(Yv)),batch_size=BS)
    best=1e9; cnt=0; bestsd=None
    for ep in range(EP):
        m.train()
        for xb,yb in tl:
            xb,yb=xb.to(dev),yb.to(dev); opt.zero_grad()
            out=m(xb); le=(out-yb)**2
            w=torch.ones_like(yb); w[yb<-20]=5.0; w[yb<-50]=15.0
            (le*w).mean().backward()
            nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        m.eval(); vb=[]
        with torch.no_grad():
            for xv,yv in vl:
                xv,yv=xv.to(dev),yv.to(dev); vb.append(mse(m(xv),yv).item())
        v=np.mean(vb)
        if v<best: best=v; cnt=0; bestsd={k:t.clone() for k,t in m.state_dict().items()}
        else:
            cnt+=1
            if cnt>=PAT: break
    m.load_state_dict(bestsd); return m

def predict(m,Xte):
    m.eval(); out=[]
    el=DataLoader(TensorDataset(torch.FloatTensor(Xte)),batch_size=256)
    with torch.no_grad():
        for (xb,) in el: out.append(m(xb.to(dev)).cpu().numpy())
    return np.concatenate(out)

def metrics(pred,seed,tag):
    err=pred-Yte; rows=[]
    # pooled over all (sample,horizon)
    yt=Yte.ravel(); e=err.ravel()
    def rmse(mask): return float(np.sqrt(np.mean(e[mask]**2))) if mask.sum() else np.nan
    rows.append(dict(seed=seed,tag=tag,horizon='all',regime='agg',n=len(e),rmse=rmse(np.ones_like(e,bool))))
    rows.append(dict(seed=seed,tag=tag,horizon='all',regime='intense',n=int((yt<-50).sum()),rmse=rmse(yt<-50)))
    rows.append(dict(seed=seed,tag=tag,horizon='all',regime='moderate',n=int(((yt>=-50)&(yt<-20)).sum()),rmse=rmse((yt>=-50)&(yt<-20))))
    rows.append(dict(seed=seed,tag=tag,horizon='all',regime='quiet',n=int((yt>=-20).sum()),rmse=rmse(yt>=-20)))
    for h in range(H):
        yh=Yte[:,h]; eh=err[:,h]
        def rh(mask): return float(np.sqrt(np.mean(eh[mask]**2))) if mask.sum() else np.nan
        rows.append(dict(seed=seed,tag=tag,horizon=f't+{h+1}',regime='agg',n=len(eh),rmse=rh(np.ones_like(eh,bool))))
        rows.append(dict(seed=seed,tag=tag,horizon=f't+{h+1}',regime='intense',n=int((yh<-50).sum()),rmse=rh(yh<-50)))
    return rows

all_rows=[]
for seed in SEEDS:
    print(f'=== seed {seed} ===',flush=True)
    for tag,d in [('base',b),('aug',a)]:
        Xtr,Xv,Xte=scale(d['Xtr'],d['Xv'],d['Xte'])
        m=train(Xtr,Xv,seed,Xtr.shape[2])
        pred=predict(m,Xte)
        r=metrics(pred,seed,tag); all_rows+=r
        agg=[x for x in r if x['horizon']=='all' and x['regime']=='agg'][0]['rmse']
        intn=[x for x in r if x['horizon']=='all' and x['regime']=='intense'][0]['rmse']
        t6=[x for x in r if x['horizon']=='t+6' and x['regime']=='intense'][0]['rmse']
        print(f'  {tag}: agg {agg:.2f} | intense {intn:.2f} | t+6 intense {t6:.2f}',flush=True)
    pd.DataFrame(all_rows).to_csv('geomag_index_ab.csv',index=False)

# paired summary
df=pd.DataFrame(all_rows)
print('\n=== PAIRED Δ (base - aug); +ve = aug better ===')
for hor in ['all','t+6']:
    for reg in (['agg','intense','moderate','quiet'] if hor=='all' else ['agg','intense']):
        sub=df[(df.horizon==hor)&(df.regime==reg)]
        if sub.empty: continue
        pv=sub.pivot(index='seed',columns='tag',values='rmse')
        d=pv['base']-pv['aug']
        print(f'{hor:4s} {reg:9s}: base {pv["base"].mean():.2f}  aug {pv["aug"].mean():.2f}  Δ {d.mean():+.2f}±{d.std():.2f}  (seeds +: {int((d>0).sum())}/{len(d)})')
print('\nsaved geomag_index_ab.csv')
