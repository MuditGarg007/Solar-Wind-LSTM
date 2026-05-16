# Evaluation Results
**Run:** 2026-04-27 23:23:29
**Config:** SEQ_LEN=48, HIDDEN_DIM=128, BIDIRECTIONAL=True, DROPOUT=0.3

## Overall Performance (t+6)
| Metric | Value |
|---|---|
| RMSE | 8.4531 nT |
| Pearson r | 0.7201 |
| R2 | 0.4141 |

## Per-Step Metrics
| Step   |   RMSE (nT) |   Pearson r |     R² |
|:-------|------------:|------------:|-------:|
| t+1    |      8.1655 |      0.7384 | 0.4533 |
| t+2    |      7.9436 |      0.7505 | 0.4826 |
| t+3    |      7.969  |      0.748  | 0.4793 |
| t+4    |      8.1479 |      0.7387 | 0.4557 |
| t+5    |      8.3153 |      0.729  | 0.4331 |
| t+6    |      8.4531 |      0.7201 | 0.4141 |

## Storm-Conditional Metrics (t+6)
| Condition                  |     N |   RMSE (nT) |   Pearson r |      R² |
|:---------------------------|------:|------------:|------------:|--------:|
| Quiet (Dst ≥ −20)          | 25414 |      8.0083 |      0.6084 | -0.0162 |
| Moderate (−50 ≤ Dst < −20) |  2384 |     11.3049 |      0.4129 | -1.793  |
| Intense (Dst < −50)        |   145 |     20.6947 |      0.2737 | -4.3855 |
