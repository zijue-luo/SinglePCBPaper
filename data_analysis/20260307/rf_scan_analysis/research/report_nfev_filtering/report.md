# nfev Filtering Impact Analysis

Filter out fit attempts with `nfev > threshold`, then apply selection logic.

- **R2层**: 同实验同n下最终被选中的fitting（fit_n_peaks 内按 max R2 选）
- **AICC层**: 同实验（包括改变n）下最终被选中的fitting（segment 内按 min AICC 跨 n 选）

## Key Results

| nfev threshold | R2层 变化率 | R2 变化时平均损失 (×0.01) | AICC层 变化率 | AICC 变化时平均损失 |
|----------------|----------------|----------------------|------------------|----------------|
| 500 | 5.5% | 6.3 | 12% | 10.7 |
| 1000 | 2.8% | 6.0 | 6.7% | 17.3 |
| 1500 | 1.4% | 0.03 | 5.3% | 3.6 |
| 2000 | 1.2% | 0.04 | 5.3% | 3.5 |
| 2500 | 0.6% | 0.01 | 2.7% | 3.9 |
| 3500 | 0.4% | 0.01 | 1.3% | 7.7 |
| 4000+ | 0% | - | 0% | - |

- **R2层**: 阈值 3500 时 0.4% 的 (实验,n) 选择发生变化；变化时 R2 平均下降约 0.0001（可忽略）。
- **AICC层**: 阈值 3500 时 1.3% 的 segment (1/75) 选择发生变化；变化时 AICC 平均增加约 7.7。
- Above 4000, no impact (all selected fits have nfev ≤ 4000).

## Plots

- `nfev_filtering_impact.png`: 2×2 panel (R2/AICC change rate and loss).
- `nfev_filtering_change_rates.png`: Combined change rates.
