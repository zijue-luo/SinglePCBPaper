# Fit History nfev Analysis

## Setup
- groups analyzed: 20
- scan_count: 120
- max_n_peaks: 9
- r2_threshold: 0.995
- max_nfev: 20000
- elapsed_s: 5114.7

## Attempt Counts
- total fit attempts: 25559
- successful attempts: 25545
- success rate: 0.999
- segment-selected attempts: 75

## nfev Quantiles (successful)
- all attempts: {0.5: 31.0, 0.9: 210.0, 0.95: 561.7999999999993, 0.99: 2151.1200000000026}
- stage-best attempts: {0.5: 29.0, 0.9: 193.89999999999992, 0.95: 588.6499999999994, 0.99: 2093.9399999999996}
- segment-selected attempts: {0.5: 41.0, 0.9: 646.6, 0.95: 1390.299999999997, 0.99: 3490.020000000001}

## Hit-max-nfev Rate
- successful attempts: 0.0
- segment-selected attempts: 0.0

## Scan Density What-if (coarse grid)
- metric: ΔR2 = full-grid best R2 - coarse-grid best R2
- stride=2: median=0.0, p95=3.166773685593364e-05, share(ΔR2<=1e-4)=0.9615384615384616
- stride=3: median=2.915612196119355e-12, p95=0.0005302044787021807, share(ΔR2<=1e-4)=0.9149797570850202

## Recommendation
- recommended max_nfev (from selected-fit q99): 3500
- selected-fit quantiles used: {0.9: 646.6, 0.95: 1390.299999999997, 0.99: 3490.020000000001}
- suggested action: try this lower max_nfev in A/B test and verify final selected model consistency.
