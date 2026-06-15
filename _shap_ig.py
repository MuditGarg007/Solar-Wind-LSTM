"""
Phase K (next-step #5): Interpretability via Integrated Gradients (IG).

Hand-rolled IG (shap/captum not in env). Attributes base BiLSTM test-set
predictions back to the 29 input features x 48 timesteps. Baseline = scaled
zero == train mean (StandardScaler), the natural "average input" reference.

Outputs:
  shap_feature_importance.csv  per-feature mean|IG|, overall + per-horizon + storm/quiet
  shap_time_profile.csv        per-timestep mean|IG| (recency profile)
  shap_ig.png                  bar chart (top feats) + time profile
Goal: confirm Bz/speed dominance -> validates DO-NOT-REPEAT #9 (model already
learned the physics-coupling transforms, so adding them was redundant).
"""
import json, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.enabled = False  # cuDNN RNN has no eval-mode backward; native LSTM does
print('device', device)

SOLAR_PATH='archive/solar_wind.csv'; LABELS_PATH='archive/labels.csv'; SUNSPOTS_PATH='archive/sunspots.csv'
SEQ_LEN=48; FORECAST_HORIZON=6; TRAIN_SPLIT=0.7; VAL_SPLIT=0.2

# ---- model def (verbatim from notebook cell 7) ----
class SolarAttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=6, dropout=0.3, bidirectional=False):
        super().__init__()
        D=2 if bidirectional else 1
        self.lstm=nn.LSTM(input_dim,hidden_dim,num_layers,batch_first=True,dropout=dropout,bidirectional=bidirectional)
        self.attention_fc=nn.Linear(hidden_dim*D,1)
        self.fc=nn.Linear(hidden_dim*D*2,output_dim)
    def forward(self,x):
        o,_=self.lstm(x); s=self.attention_fc(o); w=F.softmax(s,dim=1)
        ctx=torch.sum(o*w,dim=1); last=o[:,-1,:]
        return self.fc(torch.cat((ctx,last),dim=1))

# ---- pipeline (mirrors cells 4/5/6) ----
RAW_VARS=['bx_gse','by_gse','bz_gse','bx_gsm','by_gsm','bz_gsm','theta_gse','phi_gse','theta_gsm','phi_gsm','bt','density','speed','temperature']
def load_and_preprocess():
    ds=pd.read_csv(SOLAR_PATH); dl=pd.read_csv(LABELS_PATH); dn=pd.read_csv(SUNSPOTS_PATH)
    for d in (ds,dl,dn): d['timedelta']=pd.to_timedelta(d['timedelta'])
    ds.set_index(['period','timedelta'],inplace=True); dl.set_index(['period','timedelta'],inplace=True); dn.set_index(['period','timedelta'],inplace=True)
    g=ds.groupby(['period',pd.Grouper(level='timedelta',freq='1h')])
    m=g[RAW_VARS].mean(); m.columns=[f'{c}_mean' for c in RAW_VARS]
    s=g[RAW_VARS].std(); s.columns=[f'{c}_std' for c in RAW_VARS]
    h=pd.concat([m,s],axis=1)
    ss=dn.groupby(['period',pd.Grouper(level='timedelta',freq='1h')])['smoothed_ssn'].mean()
    h=h.join(ss,how='left'); h['smoothed_ssn']=h.groupby('period')['smoothed_ssn'].transform(lambda x:x.ffill().bfill())
    data=h.join(dl,how='inner'); data=data.interpolate(method='linear',limit_direction='both').ffill().bfill()
    return data
def create_sequences(data,seq_len,fh):
    fcols=[c for c in data.columns if c!='dst']
    X,y=[],[]
    for p in data.index.get_level_values('period').unique():
        grp=data.loc[p]
        if len(grp)<=seq_len+fh: continue
        ax=grp[fcols].values; ay=grp['dst'].values
        for i in range(len(grp)-seq_len-fh+1):
            X.append(ax[i:i+seq_len]); y.append(ay[i+seq_len:i+seq_len+fh])
    return np.array(X),np.array(y),fcols

print('loading...')
data=load_and_preprocess()
trX,trY,vaX,vaY,teX,teY=[],[],[],[],[],[]
feat_cols=None
for p in data.index.get_level_values('period').unique():
    pd_=data[data.index.get_level_values('period')==p]
    Xp,yp,fcols=create_sequences(pd_,SEQ_LEN,FORECAST_HORIZON); feat_cols=fcols
    n=len(Xp); ntr=int(n*TRAIN_SPLIT); nvl=int(n*VAL_SPLIT)
    trX.append(Xp[:ntr]); trY.append(yp[:ntr])
    vaX.append(Xp[ntr:ntr+nvl]); vaY.append(yp[ntr:ntr+nvl])
    teX.append(Xp[ntr+nvl:]); teY.append(yp[ntr+nvl:])
Xtr=np.concatenate(trX); Xte=np.concatenate(teX); yte=np.concatenate(teY)
F_dim=Xtr.shape[2]
sc=StandardScaler(); sc.fit(Xtr.reshape(-1,F_dim))
Xte_s=sc.transform(Xte.reshape(-1,F_dim)).reshape(Xte.shape)
print('test seqs',Xte_s.shape,'feats',F_dim)

model=SolarAttentionLSTM(F_dim,128,2,output_dim=6,dropout=0.3,bidirectional=True).to(device)
sd=torch.load('solar_bilstm_model.pth',map_location=device)
sd=sd.get('model_state_dict',sd) if isinstance(sd,dict) and 'model_state_dict' in sd else sd
model.load_state_dict(sd); model.eval()
print('checkpoint loaded')

# ---- Integrated Gradients ----
# baseline = zeros in scaled space (== train mean). attr = (x-base)*mean_grad over m steps.
M_STEPS=32
def integrated_gradients(X, target):
    """X: (N,T,F) np scaled. target: output head idx. returns (N,T,F) attributions."""
    base=np.zeros_like(X[0])  # scaled-zero baseline
    out=np.zeros_like(X)
    bs=128
    for i in range(0,len(X),bs):
        xb=X[i:i+bs]; n=len(xb)
        x_t=torch.tensor(xb,dtype=torch.float32,device=device)
        b_t=torch.zeros_like(x_t)
        grad_sum=torch.zeros_like(x_t)
        for k in range(1,M_STEPS+1):
            a=k/M_STEPS
            xin=(b_t+a*(x_t-b_t)).clone().requires_grad_(True)
            yp=model(xin)[:,target].sum()
            g,=torch.autograd.grad(yp,xin)
            grad_sum+=g
        avg=grad_sum/M_STEPS
        attr=(x_t-b_t)*avg
        out[i:i+bs]=attr.detach().cpu().numpy()
    return out

# storm mask: any horizon Dst < -50 in target
storm_mask=(yte<-50).any(axis=1)
print('storm test rows',int(storm_mask.sum()),'/',len(yte))

rows=[]; time_rows=[]
per_h_feat={}
for h in range(6):
    print('IG horizon t+',h+1)
    A=integrated_gradients(Xte_s,h)          # (N,T,F)
    absA=np.abs(A)
    feat_imp=absA.mean(axis=(0,1))           # (F,) mean over samples,time
    per_h_feat[h]=feat_imp
    time_prof=absA.mean(axis=(0,2))          # (T,) recency profile
    for t in range(SEQ_LEN):
        time_rows.append({'horizon':h+1,'timestep':t-SEQ_LEN,'mean_abs_ig':time_prof[t]})
    # storm vs quiet feature importance
    fs=absA[storm_mask].mean(axis=(0,1)); fq=absA[~storm_mask].mean(axis=(0,1))
    for fi,fc in enumerate(feat_cols):
        rows.append({'horizon':h+1,'feature':fc,'mean_abs_ig':feat_imp[fi],
                     'storm_mean_abs_ig':fs[fi],'quiet_mean_abs_ig':fq[fi]})

df=pd.DataFrame(rows)
df.to_csv('shap_feature_importance.csv',index=False)
pd.DataFrame(time_rows).to_csv('shap_time_profile.csv',index=False)

# overall ranking (avg over horizons)
overall=df.groupby('feature')['mean_abs_ig'].mean().sort_values(ascending=False)
print('\n=== TOP 15 FEATURES (mean|IG|, avg over horizons) ===')
print(overall.head(15).to_string())
storm_overall=df.groupby('feature')['storm_mean_abs_ig'].mean().sort_values(ascending=False)
print('\n=== TOP 10 on STORM rows ===')
print(storm_overall.head(10).to_string())

# ---- plot ----
fig,ax=plt.subplots(1,2,figsize=(15,6))
top=overall.head(15)[::-1]
ax[0].barh(top.index,top.values,color='steelblue')
ax[0].set_title('Integrated Gradients: top-15 feature importance (base, held-out test)')
ax[0].set_xlabel('mean |IG|')
tp=pd.DataFrame(time_rows).groupby('timestep')['mean_abs_ig'].mean()
ax[1].plot(tp.index,tp.values,color='firebrick')
ax[1].set_title('Recency profile (mean |IG| by input timestep)')
ax[1].set_xlabel('timestep relative to forecast (h)'); ax[1].set_ylabel('mean |IG|')
plt.tight_layout(); plt.savefig('shap_ig.png',dpi=110)
print('\nsaved shap_feature_importance.csv, shap_time_profile.csv, shap_ig.png')
