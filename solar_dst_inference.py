"""Real-time Dst forecasting — production inference module (Bucket 3).

Forecasts the Dst geomagnetic index t+1..t+6 hours ahead from a 48h solar-wind
input window, with horizon/regime-aware deploy routing, severity-adaptive
uncertainty bands, a calibrated storm-detection alarm, and data-health monitors.

Deploy policy (corrected after Phase N, 2026-06-16)
---------------------------------------------------
Persistence (hold last observed Dst) is a very strong baseline during storms
because Dst is autocorrelated; on the fixed held-out test it BEATS the model on
intense-storm point RMSE at EVERY horizon, and on aggregate at t+1..t+5. The
model wins only aggregate t+6 (quiet/moderate-driven). So:

  * Live Dst fresh  -> persistence for t+1..t+5 (all regimes) and for t+6 intense;
                       base model only for t+6 when NOT expected-intense (aggregate win).
  * Live Dst stale/absent -> fully model-driven (Dst-independent):
                       idx-model t+1..t+3 if near-real-time Kp/ap/AE present,
                       else base; base for t+4..t+6.
  * Storm alarm: multi-task classifier P(Dst<-50) per horizon, threshold tau=0.30
                 (POD 0.713 / CSI 0.573 / HSS 0.721 on held-out test). Dst-independent.
  * Uncertainty: Mondrian conformal, binned by predicted severity (95% level).
  * Monitors: magnetometer-outage (critical: intense error ~2.05x, Phase I),
              index-staleness (idx gain halves ~3h, gone ~6h, Phase J),
              Dst-staleness (gates the persistence path).

OMNI-aug model (`solar_bilstm_omni1m_model.pth`) is intentionally NOT routed:
its only edge was t+6 intense, which persistence now owns, and its training
scaler was not persisted. Base covers the Dst-stale t+6 path.

Artifacts required (same dir):
  solar_bilstm_model.pth      base BiLSTM+attention, 29 feat
  solar_bilstm_idx_model.pth  +Kp/ap/AE inputs, 32 feat
  solar_bilstm_mtl_model.pth  multi-task (reg+clf), 29 feat
  base_scaler.pkl             StandardScaler(29) refit on train (reproduces documented base)
  idx_scaler.pkl              StandardScaler(32)

Feature order must match training (`create_sequences`): 29 paper features
(14 solar-wind vars x hourly mean+std = 28, + smoothed_ssn). idx window = those
29 + [Kp, ap, AE] cadence-lagged, scaled by idx_scaler.
"""
from __future__ import annotations
import pickle
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

H = 6
HID = 128
NL = 2
ALERT_TAU = 0.30                  # mtl_prob_threshold_sweep.csv optimum (CSI/HSS)
INTENSE = -50.0
MODERATE = -20.0

# Mondrian conformal 95% FULL widths by predicted-severity bin (base model,
# conformal_base_vs_aug_bins.csv "base mond w"). Half-width = w/2.
MONDRIAN_W95 = {"quiet": 36.5, "moderate": 54.9, "intense": 107.5}
# Marginal per-horizon 95% widths (conformal_intervals.csv) — fallback only.
MARGINAL_W95 = {1: 40.0, 2: 39.5, 3: 40.3, 4: 41.3, 5: 42.7, 6: 44.5}

# Monitor thresholds.
DST_FRESH_MAX_H = 1.5             # persistence usable only if last Dst <= this old
IDX_FULL_MAX_H = 3.0             # idx full gain; 3-6h degraded; >6h disable
IDX_DEGRADE_MAX_H = 6.0
MAG_OUTAGE_INTENSE_FACTOR = 2.05  # Phase I robustness


class _AttnLSTM(nn.Module):
    """Base/idx forecaster. att_name lets us match checkpoint key naming."""
    def __init__(self, n_feat: int, att_name: str = "attention_fc"):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, HID, NL, batch_first=True,
                            dropout=0.3, bidirectional=True)
        setattr(self, att_name, nn.Linear(HID * 2, 1))
        self._att_name = att_name
        self.fc = nn.Linear(HID * 2 * 2, H)

    def forward(self, x):
        o, _ = self.lstm(x)
        w = F.softmax(getattr(self, self._att_name)(o), dim=1)
        ctx = torch.sum(o * w, dim=1)
        return self.fc(torch.cat((ctx, o[:, -1, :]), dim=1))


class _MTL(nn.Module):
    """Multi-task: shared encoder -> regression head + storm-prob (clf) head."""
    def __init__(self, n_feat: int = 29):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, HID, NL, batch_first=True,
                            dropout=0.3, bidirectional=True)
        self.attention_fc = nn.Linear(HID * 2, 1)
        self.reg_head = nn.Linear(HID * 2 * 2, H)
        self.clf_head = nn.Linear(HID * 2 * 2, H)

    def forward(self, x):
        o, _ = self.lstm(x)
        w = F.softmax(self.attention_fc(o), dim=1)
        feat = torch.cat((torch.sum(o * w, dim=1), o[:, -1, :]), dim=1)
        return self.reg_head(feat), self.clf_head(feat)


def _severity(dst: float) -> str:
    if dst < INTENSE:
        return "intense"
    if dst < MODERATE:
        return "moderate"
    return "quiet"


@dataclass
class HorizonForecast:
    horizon: int            # 1..6 (hours ahead)
    dst: float              # point forecast (nT)
    source: str             # 'persistence' | 'base' | 'idx'
    lo95: float
    hi95: float
    storm_prob: float       # P(Dst < -50), calibrated (mtl)
    severity: str


@dataclass
class Forecast:
    horizons: list[HorizonForecast]
    storm_alert: bool                       # any horizon storm_prob >= tau
    alert_horizons: list[int]
    monitors: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


class DstForecaster:
    def __init__(self, model_dir: str = ".", device: str | None = None):
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        d = model_dir.rstrip("/\\")
        self.base = self._load(_AttnLSTM(29, "attention_fc"), f"{d}/solar_bilstm_model.pth")
        self.idx = self._load(_AttnLSTM(32, "att"), f"{d}/solar_bilstm_idx_model.pth")
        self.mtl = self._load(_MTL(29), f"{d}/solar_bilstm_mtl_model.pth")
        with open(f"{d}/base_scaler.pkl", "rb") as fh:
            self.base_scaler = pickle.load(fh)
        with open(f"{d}/idx_scaler.pkl", "rb") as fh:
            self.idx_scaler = pickle.load(fh)

    def _load(self, model, path):
        sd = torch.load(path, map_location=self.dev)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        model.load_state_dict(sd)
        return model.to(self.dev).eval()

    def _scale(self, window, scaler):
        t, f = window.shape
        return scaler.transform(window.reshape(-1, f)).reshape(1, t, f)

    @torch.no_grad()
    def _base_pred(self, window29):
        x = torch.FloatTensor(self._scale(window29, self.base_scaler)).to(self.dev)
        return self.base(x).cpu().numpy()[0]

    @torch.no_grad()
    def _idx_pred(self, window32):
        x = torch.FloatTensor(self._scale(window32, self.idx_scaler)).to(self.dev)
        return self.idx(x).cpu().numpy()[0]

    @torch.no_grad()
    def _storm_prob(self, window29):
        x = torch.FloatTensor(self._scale(window29, self.base_scaler)).to(self.dev)
        _, logits = self.mtl(x)
        return torch.sigmoid(logits).cpu().numpy()[0]

    def predict(
        self,
        window29: np.ndarray,                 # (48, 29) raw paper features
        last_dst: float | None = None,        # most recent observed Dst (nT)
        dst_age_h: float = 0.0,               # staleness of last_dst (hours)
        window32: np.ndarray | None = None,   # (48, 32) for idx-model
        index_age_h: float = 0.0,             # staleness of Kp/ap/AE (hours)
        mag_ok: bool = True,                   # magnetometer healthy (Phase I)
    ) -> Forecast:
        window29 = np.asarray(window29, dtype=float)
        assert window29.shape == (48, 29), f"window29 must be (48,29), got {window29.shape}"

        storm_prob = self._storm_prob(window29)            # Dst-independent
        base = self._base_pred(window29)                   # Dst-independent

        dst_fresh = last_dst is not None and dst_age_h <= DST_FRESH_MAX_H
        idx_usable = (window32 is not None) and (index_age_h <= IDX_DEGRADE_MAX_H)
        idx = None
        if idx_usable:
            window32 = np.asarray(window32, dtype=float)
            assert window32.shape == (48, 32), f"window32 must be (48,32), got {window32.shape}"
            idx = self._idx_pred(window32)

        notes, hs = [], []
        for h in range(1, H + 1):
            i = h - 1
            p = float(storm_prob[i])
            expect_intense = p >= ALERT_TAU

            if dst_fresh:
                # persistence wins t+1..t+5 (all regimes) and t+6 intense;
                # base wins only t+6 aggregate (non-intense).
                if h <= 5 or expect_intense:
                    pred, src = float(last_dst), "persistence"
                else:
                    pred, src = float(base[i]), "base"
            else:
                # Dst-independent path
                if h <= 3 and idx is not None and index_age_h <= IDX_FULL_MAX_H:
                    pred, src = float(idx[i]), "idx"
                elif h <= 3 and idx is not None:
                    pred, src = float(idx[i]), "idx"   # degraded (3-6h), still > base front-loaded
                else:
                    pred, src = float(base[i]), "base"

            sev = _severity(pred)
            half = MONDRIAN_W95[sev] / 2.0
            if not mag_ok and sev == "intense":
                half *= MAG_OUTAGE_INTENSE_FACTOR      # widen under mag outage
            hs.append(HorizonForecast(h, round(pred, 2), src,
                                      round(pred - half, 2), round(pred + half, 2),
                                      round(p, 4), sev))

        alert_h = [hf.horizon for hf in hs if hf.storm_prob >= ALERT_TAU]
        # monitors / notes
        if not dst_fresh:
            notes.append(f"Dst stale/absent (age={dst_age_h}h > {DST_FRESH_MAX_H}h) -> model-driven path.")
        if window32 is not None and index_age_h > IDX_FULL_MAX_H:
            notes.append(f"Kp/ap/AE stale (age={index_age_h}h): idx gain halved >3h, gone >6h.")
        if window32 is None:
            notes.append("No Kp/ap/AE supplied -> idx-model path disabled.")
        if not mag_ok:
            notes.append("MAGNETOMETER OUTAGE: intense error ~2.05x (Phase I) -> bands widened; mag redundancy critical.")
        if alert_h:
            notes.append(f"STORM ALERT P(Dst<-50)>={ALERT_TAU} at t+{alert_h}.")

        return Forecast(
            horizons=hs,
            storm_alert=bool(alert_h),
            alert_horizons=alert_h,
            monitors=dict(dst_fresh=dst_fresh, idx_usable=idx_usable,
                          index_age_h=index_age_h, dst_age_h=dst_age_h, mag_ok=mag_ok),
            notes=notes,
        )


def pretty(fc: Forecast) -> str:
    lines = ["horizon  source       Dst     [ lo95 , hi95 ]   P(storm)  severity"]
    for h in fc.horizons:
        lines.append(f"  t+{h.horizon}   {h.source:<11} {h.dst:7.2f}  "
                     f"[{h.lo95:7.1f},{h.hi95:7.1f}]   {h.storm_prob:6.3f}   {h.severity}")
    lines.append(f"\nstorm_alert={fc.storm_alert}  alert_horizons={fc.alert_horizons}")
    for n in fc.notes:
        lines.append(f"  - {n}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    f = DstForecaster(model_dir=sys.argv[1] if len(sys.argv) > 1 else ".")
    print("loaded:", f.dev)
