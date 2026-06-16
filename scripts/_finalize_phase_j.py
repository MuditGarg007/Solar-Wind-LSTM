"""Adopt Phase J: train final aug-index checkpoint (32 feat, seed 42) + save scaler.
Also write lag-decay CSV from logged results."""
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, pickle
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

dev='cuda' if torch.cuda.is_available() else 'cpu'
HID=128;NL=2;DO=0.4;BS=64;LR=1e-3;WD=5e-5;EP=50;PAT=15;H=6;SEED=42
a=np.load('_seq_aug.npz'); Xtr,Ytr,Xv,Yv,Xte,Yte=a['Xtr'],a['Ytr'],a['Xv'],a['Yv'],a['Xte'],a['Yte']

class Net(nn.Module):
    def __init__(s,d):
        super().__init__(); s.lstm=nn.LSTM(d,HID,NL,batch_first=True,dropout=DO,bidirectional=True)
        s.att=nn.Linear(HID*2,1); s.fc=nn.Linear(HID*2*2,H)
    def forward(s,x):
        o,_=s.lstm(x);w=F.softmax(s.att(o),dim=1);return s.fc(torch.cat((torch.sum(o*w,1),o[:,-1,:]),1))

sc=StandardScaler();f=Xtr.shape[2]
Xtr_s=sc.fit_transform(Xtr.reshape(-1,f)).reshape(Xtr.shape)
Xv_s=sc.transform(Xv.reshape(-1,f)).reshape(Xv.shape); Xte_s=sc.transform(Xte.reshape(-1,f)).reshape(Xte.shape)
torch.manual_seed(SEED);np.random.seed(SEED);torch.cuda.manual_seed_all(SEED)
m=Net(f).to(dev);opt=torch.optim.Adam(m.parameters(),lr=LR,weight_decay=WD);mse=nn.MSELoss()
tl=DataLoader(TensorDataset(torch.FloatTensor(Xtr_s),torch.FloatTensor(Ytr)),batch_size=BS,shuffle=True)
vl=DataLoader(TensorDataset(torch.FloatTensor(Xv_s),torch.FloatTensor(Yv)),batch_size=BS)
best=1e9;cnt=0
for ep in range(EP):
    m.train()
    for xb,yb in tl:
        xb,yb=xb.to(dev),yb.to(dev);opt.zero_grad();out=m(xb);le=(out-yb)**2
        w=torch.ones_like(yb);w[yb<-20]=5.0;w[yb<-50]=15.0;(le*w).mean().backward()
        nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
    m.eval();vb=[]
    with torch.no_grad():
        for xv,yv in vl: xv,yv=xv.to(dev),yv.to(dev);vb.append(mse(m(xv),yv).item())
    v=np.mean(vb)
    if v<best: best=v;cnt=0; torch.save(m.state_dict(),'solar_bilstm_idx_model.pth')
    else:
        cnt+=1
        if cnt>=PAT: break
print(f'saved solar_bilstm_idx_model.pth (best val {best:.4f}, epoch~{ep+1})')
pickle.dump(sc,open('idx_scaler.pkl','wb'))

# eval saved checkpoint on held-out test
m.load_state_dict(torch.load('solar_bilstm_idx_model.pth')); m.eval(); out=[]
with torch.no_grad():
    for (xb,) in DataLoader(TensorDataset(torch.FloatTensor(Xte_s)),batch_size=256): out.append(m(xb.to(dev)).cpu().numpy())
pred=np.concatenate(out); err=pred-Yte; yt=Yte.ravel(); e=err.ravel()
def rm(mask): return float(np.sqrt(np.mean(e[mask]**2)))
print(f'checkpoint test: agg {rm(np.ones_like(e,bool)):.2f} | intense {rm(yt<-50):.2f} | t+1 intense {np.sqrt(np.mean((err[:,0][Yte[:,0]<-50])**2)):.2f} | t+6 intense {np.sqrt(np.mean((err[:,5][Yte[:,5]<-50])**2)):.2f}')

# lag-decay CSV (from sweep logs, hardcoded measured values)
decay=pd.DataFrame([
    dict(extra_lag_h=0,seed=42,t1_intense_aug=28.01,t1_intense_base=35.49),
    dict(extra_lag_h=0,seed=7, t1_intense_aug=33.21,t1_intense_base=37.85),
    dict(extra_lag_h=1,seed=42,t1_intense_aug=29.57,t1_intense_base=35.49),
    dict(extra_lag_h=1,seed=7, t1_intense_aug=32.80,t1_intense_base=37.85),
    dict(extra_lag_h=3,seed=42,t1_intense_aug=31.41,t1_intense_base=35.49),
    dict(extra_lag_h=3,seed=7, t1_intense_aug=36.02,t1_intense_base=37.85),
    dict(extra_lag_h=6,seed=42,t1_intense_aug=37.07,t1_intense_base=35.49),
    dict(extra_lag_h=6,seed=1, t1_intense_aug=34.27,t1_intense_base=34.96),
    dict(extra_lag_h=6,seed=7, t1_intense_aug=35.83,t1_intense_base=37.85),
])
decay['gain']=decay['t1_intense_base']-decay['t1_intense_aug']
decay.to_csv('geomag_index_lag_decay.csv',index=False)
print('saved geomag_index_lag_decay.csv')
