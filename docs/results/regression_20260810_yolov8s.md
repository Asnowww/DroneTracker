# AirSim 自动追踪回归报告

- weights: `runs\detect\mixed_ft_v8s\weights\best.pt`
- episode: 90.0 s @ 20 Hz
- baseline variant: `no_prediction`

## target pattern: `front_sweep`

| variant | visible_rate | lost_count | center_err_mean_px | Δmean | center_err_p95_px | Δp95 | pred_used | status |
|---|---|---|---|---|---|---|---|---|
| `no_prediction` | 1.000 | 0 | 23.22 | +0.0% · | 47.17 | +0.0% · | 0.00 | PASS |
| `kalman` | 1.000 | 0 | 22.65 | -2.5% ✓ | 45.88 | -2.7% ✓ | 1.00 | PASS |
| `imm` | 1.000 | 0 | 22.34 | -3.8% ✓ | 43.26 | -8.3% ✓ | 1.00 | PASS |

## target pattern: `figure8`

| variant | visible_rate | lost_count | center_err_mean_px | Δmean | center_err_p95_px | Δp95 | pred_used | status |
|---|---|---|---|---|---|---|---|---|
| `no_prediction` | 1.000 | 0 | 43.82 | +0.0% · | 99.00 | +0.0% · | 0.00 | PASS |
| `kalman` | 1.000 | 0 | 45.15 | +3.0% ✗ | 103.87 | +4.9% ✗ | 1.00 | PASS |
| `imm` | 1.000 | 0 | 44.26 | +1.0% ✗ | 97.89 | -1.1% ✓ | 1.00 | PASS |

## target pattern: `lateral_dash`

| variant | visible_rate | lost_count | center_err_mean_px | Δmean | center_err_p95_px | Δp95 | pred_used | status |
|---|---|---|---|---|---|---|---|---|
| `no_prediction` | 1.000 | 0 | 10.85 | +0.0% · | 23.35 | +0.0% · | 0.00 | PASS |
| `kalman` | 1.000 | 0 | 10.92 | +0.6% · | 23.64 | +1.2% ✗ | 1.00 | PASS |
| `imm` | 1.000 | 0 | 10.97 | +1.1% ✗ | 22.99 | -1.6% ✓ | 1.00 | PASS |

## 验收阈值

- `visible_rate_min` = 0.98
- `lost_count_max` = 0
- `center_error_p95_px_max` = 67.71

**9/9 组通过验收。**
