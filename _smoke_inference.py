"""Smoke test for solar_dst_inference.DstForecaster.
Exercises routing scenarios on real held-out windows + verifies base preds
reproduce the documented intense t+6 RMSE (44.64)."""
import numpy as np
from solar_dst_inference import DstForecaster, pretty

f = DstForecaster(model_dir=".")
print("device:", f.dev)

b = np.load("_seq_base.npz"); a = np.load("_seq_aug.npz")
Xte29, Yte = b["Xte"], b["Yte"]
Xte32 = a["Xte"]
assert np.array_equal(b["Yte"], a["Yte"]), "base/aug Yte must align"
assert Xte32.shape[2] == 32 and Xte29.shape[2] == 29

# pick a quiet and an intense window (by t+6 actual)
qi = int(np.argmax(Yte[:, 5] >= -5))          # a quiet origin
si = int(np.argmin(Yte[:, 5]))                 # deepest storm origin
print(f"\nquiet window idx {qi} (actual t+6={Yte[qi,5]:.0f})  "
      f"intense window idx {si} (actual t+6={Yte[si,5]:.0f})")

print("\n=== Scenario A: quiet, fresh Dst ===")
print(pretty(f.predict(Xte29[qi], last_dst=-4.0, dst_age_h=1.0,
                       window32=Xte32[qi], index_age_h=1.0)))

print("\n=== Scenario B: intense storm, fresh Dst ===")
print(pretty(f.predict(Xte29[si], last_dst=-150.0, dst_age_h=1.0,
                       window32=Xte32[si], index_age_h=1.0)))

print("\n=== Scenario C: intense, Dst STALE (model-driven, idx fresh) ===")
print(pretty(f.predict(Xte29[si], last_dst=-150.0, dst_age_h=8.0,
                       window32=Xte32[si], index_age_h=1.0)))

print("\n=== Scenario D: intense, Dst stale + MAG OUTAGE + no indices ===")
print(pretty(f.predict(Xte29[si], last_dst=None, dst_age_h=99.0, mag_ok=False)))

# batch sanity: base preds reproduce documented intense t+6 RMSE
print("\n=== Batch sanity: base intense t+6 RMSE (expect ~44.64) ===")
import torch
preds = []
with torch.no_grad():
    Xs = f.base_scaler.transform(Xte29.reshape(-1, 29)).reshape(Xte29.shape)
    for i in range(0, len(Xs), 512):
        xb = torch.FloatTensor(Xs[i:i+512]).to(f.dev)
        preds.append(f.base(xb).cpu().numpy())
preds = np.concatenate(preds)
m = Yte[:, 5] < -50
rmse = float(np.sqrt(np.mean((preds[m, 5] - Yte[m, 5])**2)))
print(f"intense N={int(m.sum())}  t+6 RMSE={rmse:.2f}  -> {'PASS' if abs(rmse-44.64)<0.5 else 'FAIL'}")
print("\nsmoke test done.")
