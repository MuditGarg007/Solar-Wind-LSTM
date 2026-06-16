"""Phase N verify: recompute documented base + OMNI-aug preds on EXACT 387-hour held-out test.
Re-derive base-persist & aug-persist per-horizon. Check if persistence wins t+6 intense.
Loads documented checkpoints; scaler refit on base train windows (_seq_base.npz Xtr)."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, pandas as pd
from sklearn.preprocessing import StandardScaler
dev='cuda' if torch.cuda.is_available() else 'cpu'
H=6; HID=128; NL=2

class Net(nn.Module):
    def __init__(s,d):
        super().__init__()
        s.lstm=nn.LSTM(d,HID,NL,batch_first=True,dropout=0.3,bidirectional=True)
        s.attention_fc=nn.Linear(HID*2,1); s.fc=nn.Linear(HID*2*2,H)
    def forward(s,x):
        o,_=s.lstm(x); w=F.softmax(s.attention_fc(o),dim=1)
        ctx=torch.sum(o*w,dim=1); last=o[:,-1,:]
        return s.fc(torch.cat((ctx,last),dim=1))

b=np.load('_seq_base.npz')
Xtr,Xte,Yte=b['Xtr'],b['Xte'],b['Yte']
sc=StandardScaler(); n,t,f=Xtr.shape
sc.fit(Xtr.reshape(-1,f))
Xte_s=sc.transform(Xte.reshape(-1,f)).reshape(Xte.shape)

def predict(ckpt):
    m=Net(f).to(dev); m.load_state_dict(torch.load(ckpt,map_location=dev)); m.eval()
    with torch.no_grad():
        out=[]
        for i in range(0,len(Xte_s),256):
            xb=torch.FloatTensor(Xte_s[i:i+256]).to(dev)
            out.append(m(xb).cpu().numpy())
    return np.concatenate(out)

def rmse(e,m): return float(np.sqrt(np.mean(e[m]**2))) if m.sum() else np.nan

pred_base=predict('solar_bilstm_model.pth')
pred_aug =predict('solar_bilstm_omni1m_model.pth')

# persistence from burton csv (already on aligned test hours)
bd=pd.read_csv('burton_baseline.csv')
pers=bd[bd.model=='persistence'].set_index(['horizon','regime']).rmse

print('horizon | base agg | aug agg | persist agg || base int | aug int | persist int')
rows=[]
for h in range(H):
    yh=Yte[:,h]; mi=yh<-50; ma=np.ones_like(yh,bool)
    eb=pred_base[:,h]-yh; eaug=pred_aug[:,h]-yh
    ba,bi=rmse(eb,ma),rmse(eb,mi)
    aa,ai=rmse(eaug,ma),rmse(eaug,mi)
    pa=pers[(f't+{h+1}','agg')]; pi=pers[(f't+{h+1}','intense')]
    print(f't+{h+1}    | {ba:7.2f} | {aa:6.2f} | {pa:9.2f} || {bi:7.2f} | {ai:6.2f} | {pi:9.2f}')
    rows.append(dict(h=h+1,base_agg=ba,aug_agg=aa,pers_agg=pa,base_int=bi,aug_int=ai,pers_int=pi))

df=pd.DataFrame(rows)
print('\n=== documented sanity (expect base t+6 agg~11.26 int~44.64 ; aug t+6 agg~10.94 int~36.10) ===')
r6=df[df.h==6].iloc[0]
print(f"base t+6 agg {r6.base_agg:.2f} int {r6.base_int:.2f} | aug t+6 agg {r6.aug_agg:.2f} int {r6.aug_int:.2f}")
print('\n=== t+6 intense verdict ===')
print(f"persist {r6.pers_int:.2f} vs base {r6.base_int:.2f} vs aug {r6.aug_int:.2f}")
df.to_csv('phase_n_verify.csv',index=False)
print('saved phase_n_verify.csv')
