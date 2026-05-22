# Evaluation Results
**Run:** 2026-05-22 17:50:21
**Config:** SEQ_LEN=48, HIDDEN_DIM=128, BIDIRECTIONAL=True, DROPOUT=0.4

## Overall Performance (t+6)
| Metric | Value |
|---|---|
| RMSE | 11.5731 nT |
| Pearson r | 0.7755 |
| R2 | 0.5822 |

## Per-Step Metrics
| Step   |   RMSE (nT) |   Pearson r |     R² |
|:-------|------------:|------------:|-------:|
| t+1    |     10.144  |      0.8272 | 0.6788 |
| t+2    |     10.1132 |      0.8286 | 0.6808 |
| t+3    |     10.3338 |      0.8202 | 0.6668 |
| t+4    |     10.7069 |      0.806  | 0.6423 |
| t+5    |     11.1093 |      0.7908 | 0.615  |
| t+6    |     11.5731 |      0.7755 | 0.5822 |

## Storm-Conditional Metrics (t+6)
| Condition                  |     N |   RMSE (nT) |   Pearson r |      R² |
|:---------------------------|------:|------------:|------------:|--------:|
| Quiet (Dst ≥ −20)          | 23840 |      9.3141 |      0.5325 | -0.1652 |
| Moderate (−50 ≤ Dst < −20) |  3440 |     14.2021 |      0.4012 | -2.4207 |
| Intense (Dst < −50)        |   662 |     38.4836 |      0.5853 |  0.0249 |
