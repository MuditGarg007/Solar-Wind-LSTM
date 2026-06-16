"""Phase L train/eval: 5-seed paired BiLSTM on identical windows.
  arm 'dst'  -> target MagNet hourly Dst (control, rebuilt on a/b/c windows)
  arm 'symh' -> target OMNI hourly-MIN SYM-H
Same arch/loss/optim as adopted base. Eval on held-out TEST: per-horizon RMSE,
storm-conditional RMSE, storm-detection POD/FAR/CSI/BIAS (each on own target).
Outputs: symh_perhorizon.csv, symh_stormcond.csv, symh_detection.csv, solar_bilstm_symh_model.pth (seed42).
"""
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('device',device)

class SolarAttentionLSTM(nn.Module):
    def __init__(s,input_dim,hidden_dim,num_layers,output_dim=6,dropout=0.3,bidirectional=False):
        super().__init__(); D=2 if bidirectional else 1
        s.lstm=nn.LSTM(input_dim,hidden_dim,num_layers,batch_first=True,dropout=dropout,bidirectional=bidirectional)
        s.attention_fc=nn.Linear(hidden_dim*D,1); s.fc=nn.Linear(hidden_dim*D*2,output_dim)
    def forward(s,x):
        o,_=s.lstm(x); w=F.softmax(s.attention_fc(o),dim=1)
        ctx=torch.sum(o*w,dim=1); return s.fc(torch.cat((ctx,o[:,-1,:]),dim=1))

z=np.load('_seq_symh.npz',allow_pickle=True)
Xtr,Xv,Xte=z['Xtr'],z['Xv'],z['Xte']
F_dim=Xtr.shape[2]
sc=StandardScaler(); sc.fit(Xtr.reshape(-1,F_dim))
def scale(X): return sc.transform(X.reshape(-1,F_dim)).reshape(X.shape)
Xtr_s,Xv_s,Xte_s=scale(Xtr),scale(Xv),scale(Xte)
TARGETS={'dst':('Ydst_tr','Ydst_v','Ydst_te'),'symh':('Ysymh_tr','Ysymh_v','Ysymh_te')}

def wmse(out,y):
    e=(out-y)**2; w=torch.ones_like(y); w[y<-20]=5.0; w[y<-50]=15.0
    return torch.mean(e*w)

def train_one(Ytr,Yv,seed,save=None):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    tl=DataLoader(TensorDataset(torch.FloatTensor(Xtr_s),torch.FloatTensor(Ytr)),batch_size=64,shuffle=True)
    vl=DataLoader(TensorDataset(torch.FloatTensor(Xv_s),torch.FloatTensor(Yv)),batch_size=64)
    m=SolarAttentionLSTM(F_dim,128,2,dropout=0.4,bidirectional=True).to(device)
    opt=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=5e-5); crit=nn.MSELoss()
    best=1e9; cnt=0; best_state=None
    for ep in range(50):
        m.train()
        for xb,yb in tl:
            xb,yb=xb.to(device),yb.to(device); opt.zero_grad()
            loss=wmse(m(xb),yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        m.eval(); vlosses=[]
        with torch.no_grad():
            for xv,yv in vl: vlosses.append(crit(m(xv.to(device)),yv.to(device)).item())
        v=np.mean(vlosses)
        if v<best: best=v; cnt=0; best_state={k:t.cpu().clone() for k,t in m.state_dict().items()}
        else:
            cnt+=1
            if cnt>=15: break
    m.load_state_dict(best_state)
    if save: torch.save(best_state,save)
    return m

def predict(m,X):
    m.eval(); out=[]
    with torch.no_grad():
        for i in range(0,len(X),256):
            out.append(m(torch.FloatTensor(X[i:i+256]).to(device)).cpu().numpy())
    return np.concatenate(out)

def detect(truth,pred,thr):
    t=truth<thr; p=pred<thr
    TP=int((t&p).sum()); FP=int((~t&p).sum()); FN=int((t&~p).sum())
    POD=TP/(TP+FN) if TP+FN else np.nan
    FAR=FP/(TP+FP) if TP+FP else np.nan
    CSI=TP/(TP+FP+FN) if (TP+FP+FN) else np.nan
    BIAS=(TP+FP)/(TP+FN) if (TP+FN) else np.nan
    return POD,FAR,CSI,BIAS

SEEDS=[42,1,2,3,4]
ph_rows=[]; sc_rows=[]; det_rows=[]
for arm,(ktr,kv,kte) in TARGETS.items():
    Ytr,Yv,Yte=z[ktr],z[kv],z[kte]
    for si,seed in enumerate(SEEDS):
        save='solar_bilstm_symh_model.pth' if (arm=='symh' and seed==42) else None
        m=train_one(Ytr,Yv,seed,save=save)
        P=predict(m,Xte_s)               # (N,6)
        # per-horizon RMSE
        for h in range(6):
            r=np.sqrt(np.mean((P[:,h]-Yte[:,h])**2))
            ph_rows.append({'arm':arm,'seed':seed,'horizon':h+1,'rmse':r})
        # storm-conditional (flatten)
        pf=P.flatten(); yf=Yte.flatten()
        for name,mask in [('quiet',yf>=-20),('moderate',(yf<-20)&(yf>=-50)),('intense',yf<-50)]:
            if mask.sum()==0: continue
            r=np.sqrt(np.mean((pf[mask]-yf[mask])**2))
            sc_rows.append({'arm':arm,'seed':seed,'bin':name,'n':int(mask.sum()),'rmse':r})
        # detection (flatten over horizons)
        for thr in (-50,-80,-100):
            POD,FAR,CSI,BIAS=detect(yf,pf,thr)
            det_rows.append({'arm':arm,'seed':seed,'thr':thr,'POD':POD,'FAR':FAR,'CSI':CSI,'BIAS':BIAS})
        print(f'{arm} seed{seed} done  t+6 rmse {np.sqrt(np.mean((P[:,5]-Yte[:,5])**2)):.2f}',flush=True)

ph=pd.DataFrame(ph_rows); sc=pd.DataFrame(sc_rows); det=pd.DataFrame(det_rows)
ph.to_csv('symh_perhorizon.csv',index=False); sc.to_csv('symh_stormcond.csv',index=False); det.to_csv('symh_detection.csv',index=False)

def agg(df,by,val):
    return df.groupby(by)[val].agg(['mean','std']).round(3)
print('\n=== PER-HORIZON RMSE (mean over seeds, own target) ===')
print(ph.groupby(['arm','horizon'])['rmse'].mean().round(2).unstack(0))
print('\n=== STORM-CONDITIONAL RMSE (mean over seeds, own target) ===')
print(sc.groupby(['arm','bin'])['rmse'].mean().round(2).unstack(0))
print('\n=== STORM DETECTION (mean over seeds, own target) ===')
print(det.groupby(['arm','thr'])[['POD','FAR','CSI','BIAS']].mean().round(3))
print('\nsaved csvs + solar_bilstm_symh_model.pth')
