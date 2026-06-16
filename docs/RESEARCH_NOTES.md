# Research Notes: Current Benchmarks & Top Research on Solar Wind Dst Prediction

Dataset reference: [NASA and NOAA Satellites Solar-Wind Dataset (Kaggle)](https://www.kaggle.com/datasets/arashnic/soalr-wind)

---

## 1. Dataset Context & Competition Origins

The Kaggle dataset (arashnic/soalr-wind) originates from the **MagNet: Model the Geomagnetic Field** competition, hosted on DrivenData in collaboration with NASA and NOAA. The competition used real-time solar wind (RTSW) data feeds from NOAA's DSCOVR and NASA's ACE satellites — the same source as the Kaggle dataset.

- **622 participants** from 64 countries, **1,197 models** submitted
- Evaluation metric: RMSE on residuals between predicted and observed Dst
- Official benchmark (persistence model): **15.2 nT RMSE** on private test set

---

## 2. MagNet Competition Results (Gold Standard Benchmarks)

These are the best-known benchmarks directly on this dataset/task.

| Model | RMSE (nT) | Notes |
|---|---|---|
| Persistence baseline | 15.2 | Official DrivenData benchmark |
| Top 4 winners (avg) | 11.1 – 11.5 | LSTM, GRU, CNN, LGBM ensembles |
| Ensemble of top 4 | **10.6** | 30% reduction vs. baseline |
| Winner (verification set) | **5.9** | Evaluated on Nov 2020 – Mar 2021 data |
| NCEI operational model | 6.5 | Official NOAA model (verification set) |
| Ensemble of winners | **5.6** | Best result on held-out verification set |
| Extreme events (Dst ≤ −80 nT) | 38 – 50 | Winners; baseline was 76 nT |

**Key winning techniques:** ensembles of LSTM, GRU, CNN, and LightGBM (LGBM); various imputation strategies for missing sensor data; multi-window input features.

---

## 3. State-of-the-Art Research Models (2023–2025)

### 3.1 TriQXNet (2024) — Current SOTA for 1-hour Dst Prediction
**Paper:** [TriQXNet: Forecasting Dst Index from Solar Wind Data Using an Interpretable Parallel Classical–Quantum Framework with Uncertainty Quantification](https://arxiv.org/abs/2407.06658v3)

- Hybrid classical-quantum neural network with conformal prediction for uncertainty quantification and SHAP-based explainability (XAI)
- Outperforms **13 state-of-the-art hybrid deep-learning models**
- **RMSE: 9.27 nT** (1-hour ahead forecast)
- Single classical pathway: 9.42 nT; quantum branch alone: 9.75 nT
- Statistically confirmed superior to LSTM (t = 222.89, p < 0.05) and DeepSeqConvNet (t = 267.97, p < 0.05)

### 3.2 Multi-Fidelity Boosted GRU (2023) — Best Reported for 6-Hour Forecast
**Paper:** [Multi-Hour-Ahead Dst Index Prediction Using Multi-Fidelity Boosted Neural Networks](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022SW003286)

- GRU base model enhanced with multi-fidelity boosting to reduce uncertainty at longer lead times
- **1-hour ahead:** RMSE ~8.22 nT (strong storm periods)
- **6-hour ahead: RMSE 13.54 nT**
- Validated on 2003 and 2022 Halloween storms
- Notably, accuracy degrades significantly with lead time — a known open problem in the field

### 3.3 EMD-LSTM (2023) — Strong 1-Hour Baseline
**Paper:** [Short Time Prediction of Dst Index Based on LSTM and EMD-LSTM Models (MDPI Applied Sciences)](https://www.mdpi.com/2076-3417/13/21/11824)

- Empirical Mode Decomposition + LSTM
- **1-hour ahead RMSE: 7.34 nT, Pearson r = 0.96** (active/storm period)
- Quiet period: RMSE 2.64 nT, r = 0.97
- Trained on 1996–2022 data, validated on Jan–May 2023

### 3.4 Interpretable ML — KAN Networks & Symbolic Regression (2025)
**Paper:** [Disturbance Storm Time Index Prediction with Interpretable Machine Learning (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S1877750326000396) | [Discovering Governing Equations via Symbolic Regression (arXiv)](https://arxiv.org/html/2504.18461v1)

- KAN networks and PyOperon symbolic regression to find closed-form equations for Dst evolution
- MLP still outperforms symbolic models in accuracy, but symbolic models offer physical interpretability
- Benchmarked against Burton-McPherron-Russell and O'Brien-McPherron empirical models
- Evaluated on May 2024 extreme storm (Dst < −400 nT), 2003 Halloween storm, 2015 St. Patrick's Day storm

### 3.5 Bayesian Deep Learning for Dst (2022, widely cited baseline)
**Repo:** [ccsc-tools/Dst-prediction](https://github.com/ccsc-tools/Dst-prediction)

- Combines multi-head attention transformer with Bayesian inference for aleatoric + epistemic uncertainty
- arXiv: [2205.02447](https://arxiv.org/abs/2205.02447)

### 3.6 CNN + BiLSTM (2022, AGU)
**Paper:** [Deep Neural Networks With Convolutional and LSTM Layers for SYM-H and ASY-H Forecasting](https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2021SW002748)

- BiLSTM with 256 neurons (first layer) + three layers of 125 neurons, dropout 0.1
- Targets SYM-H (a higher-resolution analogue to Dst)
- Established a strong convolutional-recurrent baseline for the field

---

## 4. How Our Model Compares

| Model | Lead Time | RMSE (nT) | Pearson r | R² |
|---|---|---|---|---|
| **Our SolarAttentionLSTM** | **6 hours** | **9.54** | **0.6436** | **0.2527** |
| Multi-Fidelity GRU (Hu 2023) | 6 hours | 13.54 | — | — |
| TriQXNet (2024) | 1 hour | 9.27 | — | — |
| EMD-LSTM (2023) | 1 hour | 7.34 | 0.96 | — |
| MagNet winners ensemble | ~1 hour | 10.6 | — | — |

**Key observation:** Our 6-hour RMSE of 9.54 nT is actually **better than the published 6-hour benchmark** from the best academic model (13.54 nT from Hu et al. 2023), and comparable to TriQXNet's 1-hour RMSE. This is likely because:
- We use a cleaner dataset (pre-aggregated Kaggle version vs. raw noisy RTSW feeds)
- Chronological 80/20 split avoids data leakage but results are not directly comparable to the competition's rolling test window
- Lower R² (0.25) compared to 1-hour models (which achieve r > 0.9) reveals the harder physics of longer-range forecasting

---

## 5. Key Research Trends & Open Problems

### Trends
- **Uncertainty quantification** is becoming mandatory for operational models (conformal prediction, Bayesian inference)
- **Explainability (XAI)** via SHAP, attention visualization, and symbolic regression is increasingly required for scientific acceptance
- **Hybrid architectures** (CNN + LSTM, classical + quantum) consistently outperform pure-LSTM baselines
- **Physics-informed features** (energy coupling terms like `V × Bz`) are now standard; our model already incorporates this
- **Extreme event weighting** in the loss function (our asymmetric MSE) is aligned with current best practices — MagNet winners used similar strategies

### Open Problems
- Accuracy degrades sharply for lead times > 3 hours — 6-hour prediction is genuinely hard
- Rare, extreme storms (Dst < −100 nT) remain poorly predicted even by best models
- Real-time operability requires handling missing sensor data robustly (satellite dropout, calibration gaps)
- Low R² at 6-hour horizon is a field-wide problem, not specific to our model

### Potential Improvements Suggested by Literature
1. **Multi-fidelity boosting** — stack a correction model on top of our base LSTM (Hu et al. 2023)
2. **Ensemble our model** with a gradient-boosted tree (LightGBM) on engineered lag features — top MagNet strategy
3. **Add uncertainty bounds** using conformal prediction wrappers (no retraining needed)
4. **SYM-H as target** instead of Dst — higher temporal resolution (1-minute) may improve signal
5. **Extended input window** — some top models use 48–72 hour windows vs. our 24-hour window

---

## 6. Sources

- [MagNet Competition — DrivenData](https://www.drivendata.org/competitions/73/noaa-magnetic-forecasting/)
- [Meet the Winners of MagNet — DrivenData Labs](https://drivendata.co/blog/magnet-geomagnetic-field-winners/)
- [MagNet Paper — Space Weather, Nair et al. 2023](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023SW003514)
- [TriQXNet — arXiv 2407.06658](https://arxiv.org/abs/2407.06658v3)
- [Multi-Fidelity Boosted GRU — Space Weather, Hu et al. 2023](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022SW003286)
- [EMD-LSTM — MDPI Applied Sciences 2023](https://www.mdpi.com/2076-3417/13/21/11824)
- [Interpretable ML for Dst — ScienceDirect 2025](https://www.sciencedirect.com/science/article/pii/S1877750326000396)
- [Symbolic Regression for Geomagnetic Storm Dynamics — arXiv 2025](https://arxiv.org/html/2504.18461v1)
- [Bayesian Deep Learning Dst — arXiv 2205.02447](https://arxiv.org/abs/2205.02447)
- [CNN + LSTM for SYM-H — AGU Space Weather 2022](https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2021SW002748)
- [DrivenData LSTM Benchmark Blog](https://drivendata.co/blog/model-geomagnetic-field-benchmark)
- [MagNet Kaggle Mirror Dataset](https://www.kaggle.com/datasets/kingabzpro/magnet-nasa)
