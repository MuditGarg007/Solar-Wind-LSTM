"""Generate paper figures from existing result CSVs. No modeling. -> figs/*.png"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figs", exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3})

pv = pd.read_csv("phase_n_verify.csv")          # h, base/aug/pers _agg/_int
bb = pd.read_csv("burton_baseline.csv")
H = pv.h.values

def burton_fc(reg):
    d = bb[(bb.model == "burton_forecast(frozen driver)") & (bb.regime == reg)]
    return d.sort_values("horizon").rmse.values

# ---- Fig 1: per-horizon RMSE, aggregate vs intense (model/persistence/Burton) ----
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(H, pv.base_agg, "o-", label="BiLSTM (model)")
ax[0].plot(H, pv.pers_agg, "s--", label="persistence")
ax[0].plot(H, burton_fc("agg"), "^:", label="Burton OM2000")
ax[0].set_title("Aggregate (N=13974)"); ax[0].set_xlabel("forecast horizon (h)")
ax[0].set_ylabel("RMSE (nT)"); ax[0].legend()
ax[1].plot(H, pv.base_int, "o-", label="BiLSTM (model)")
ax[1].plot(H, pv.pers_int, "s--", label="persistence")
ax[1].plot(H, burton_fc("intense"), "^:", label="Burton OM2000")
ax[1].set_title("Intense storms (N=387)"); ax[1].set_xlabel("forecast horizon (h)")
ax[1].set_ylabel("RMSE (nT)"); ax[1].legend()
fig.suptitle("Per-horizon RMSE: persistence beats the model on intense at every lead")
fig.tight_layout(); fig.savefig("figs/fig1_perhorizon_rmse.png"); plt.close(fig)

# ---- Fig 2: storm-conditional t+6 (model vs persistence) ----
reg = ["quiet", "moderate", "intense"]
base_t6 = [7.94, 12.16, 44.64]
pers_t6 = [bb[(bb.model == "persistence") & (bb.horizon == "t+6") & (bb.regime == r)].rmse.iloc[0] for r in reg]
x = np.arange(3); w = 0.38
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(x - w/2, base_t6, w, label="BiLSTM (model)")
ax.bar(x + w/2, pers_t6, w, label="persistence")
ax.set_xticks(x); ax.set_xticklabels([r.capitalize() for r in reg])
ax.set_ylabel("t+6 RMSE (nT)"); ax.set_title("Storm-conditional t+6 RMSE")
for i, (b, p) in enumerate(zip(base_t6, pers_t6)):
    ax.text(i - w/2, b + 0.5, f"{b:.1f}", ha="center", fontsize=8)
    ax.text(i + w/2, p + 0.5, f"{p:.1f}", ha="center", fontsize=8)
ax.legend(); fig.tight_layout(); fig.savefig("figs/fig2_storm_conditional.png"); plt.close(fig)

# ---- Fig 3: calibrated storm detection (MTL threshold sweep) ----
sw = pd.read_csv("mtl_prob_threshold_sweep.csv").sort_values("prob_thr")
fig, ax = plt.subplots(figsize=(6, 4))
for col, mk in [("POD", "o-"), ("FAR", "s--"), ("CSI", "^-"), ("HSS", "d:")]:
    ax.plot(sw.prob_thr, sw[col], mk, label=col)
ax.axvline(0.30, color="k", lw=1, ls=":", alpha=0.6)
ax.text(0.305, 0.05, r"$\tau$=0.30", fontsize=9)
ax.set_xlabel("alarm probability threshold"); ax.set_ylabel("skill score")
ax.set_title("Calibrated storm detection (multi-task clf, AUC 0.983)")
ax.legend(ncol=2); fig.tight_layout(); fig.savefig("figs/fig3_detection.png"); plt.close(fig)

# ---- Fig 4: UQ — interval width vs horizon + Mondrian coverage by regime ----
ci = pd.read_csv("conformal_intervals.csv")
cb = pd.read_csv("conformal_base_vs_aug_bins.csv")
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
hh = [int(s[2:]) for s in ci.Step]
ax[0].plot(hh, ci["w@90"], "o-", label="90%")
ax[0].plot(hh, ci["w@95"], "s-", label="95%")
ax[0].plot(hh, ci["w@99"], "^-", label="99%")
ax[0].set_xlabel("forecast horizon (h)"); ax[0].set_ylabel("interval width (nT)")
ax[0].set_title("Marginal conformal width grows with lead"); ax[0].legend()
bins = cb["Bin (by actual)"].values
xb = np.arange(len(bins))
ax[1].bar(xb - 0.2, cb["base mond cov%"], 0.4, label="base Mondrian")
ax[1].bar(xb + 0.2, cb["aug mond cov%"], 0.4, label="aug Mondrian")
ax[1].axhline(95, color="k", lw=1, ls=":", alpha=0.6)
ax[1].set_xticks(xb); ax[1].set_xticklabels(["Quiet", "Moderate", "Intense"])
ax[1].set_ylabel("95% coverage (%)"); ax[1].set_ylim(50, 100)
ax[1].set_title("Mondrian coverage by severity"); ax[1].legend()
fig.tight_layout(); fig.savefig("figs/fig4_uq.png"); plt.close(fig)

print("wrote figs/fig1_perhorizon_rmse.png fig2_storm_conditional.png "
      "fig3_detection.png fig4_uq.png")
