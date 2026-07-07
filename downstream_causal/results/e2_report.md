# E2: switch-injection dose-response and repair report

#### Manipulation check: ape_vs_base by arm and level (must be ~equal within a level)

|   p_injected |    drift |   switch |
|-------------:|---------:|---------:|
|          0   | nan      |   0      |
|          0.1 |  11.0631 |  11.0631 |
|          0.2 |  23.6572 |  23.6572 |
|          0.3 |  36.7612 |  36.7612 |
|          0.5 |  57.5504 |  57.5504 |

#### Dose-response on seg_iou (switch arm; mean over sequences x seeds)

| p_injected | mean seg_iou |
|---|---|
| 0.0 | 0.985 |
| 0.1 | 0.653 |
| 0.2 | 0.574 |
| 0.3 | 0.569 |
| 0.5 | 0.485 |

Paired Wilcoxon (adjacent levels, paired by sequence+seed):

| level_a vs level_b | W | p |
|---|---|---|
| 0.0 vs 0.1 | 0.0 | 4.66e-10 |
| 0.1 vs 0.2 | 170.0 | 0.0803 |
| 0.2 vs 0.3 | 258.0 | 0.919 |
| 0.3 vs 0.5 | 128.0 | 0.00991 |

#### Switch vs drift (APE-matched) on seg_iou, at each injected level

| p_injected | switch mean | drift mean | Wilcoxon p |
|---|---|---|---|
| 0.1 | 0.653 | 0.613 | 0.405 |
| 0.2 | 0.574 | 0.570 | 0.963 |
| 0.3 | 0.569 | 0.570 | 0.747 |
| 0.5 | 0.485 | 0.433 | 0.077 |

#### Dose-response on ari (switch arm; mean over sequences x seeds)

| p_injected | mean ari |
|---|---|
| 0.0 | 0.972 |
| 0.1 | 0.472 |
| 0.2 | 0.247 |
| 0.3 | 0.166 |
| 0.5 | 0.096 |

Paired Wilcoxon (adjacent levels, paired by sequence+seed):

| level_a vs level_b | W | p |
|---|---|---|
| 0.0 vs 0.1 | 0.0 | 7.77e-07 |
| 0.1 vs 0.2 | 22.0 | 1.47e-05 |
| 0.2 vs 0.3 | 91.0 | 0.00207 |
| 0.3 vs 0.5 | 129.0 | 0.0115 |

#### Switch vs drift (APE-matched) on ari, at each injected level

| p_injected | switch mean | drift mean | Wilcoxon p |
|---|---|---|---|
| 0.1 | 0.472 | 0.534 | 0.0258 |
| 0.2 | 0.247 | 0.419 | 7.28e-05 |
| 0.3 | 0.166 | 0.370 | 0.000307 |
| 0.5 | 0.096 | 0.126 | 0.667 |

#### Dose-response on boundary_f1 (switch arm; mean over sequences x seeds)

| p_injected | mean boundary_f1 |
|---|---|
| 0.0 | 0.990 |
| 0.1 | 0.738 |
| 0.2 | 0.710 |
| 0.3 | 0.726 |
| 0.5 | 0.674 |

Paired Wilcoxon (adjacent levels, paired by sequence+seed):

| level_a vs level_b | W | p |
|---|---|---|
| 0.0 vs 0.1 | 0.0 | 4.66e-10 |
| 0.1 vs 0.2 | 261.0 | 0.963 |
| 0.2 vs 0.3 | 232.0 | 0.561 |
| 0.3 vs 0.5 | 155.0 | 0.0413 |

#### Switch vs drift (APE-matched) on boundary_f1, at each injected level

| p_injected | switch mean | drift mean | Wilcoxon p |
|---|---|---|---|
| 0.1 | 0.738 | 0.665 | 0.0225 |
| 0.2 | 0.710 | 0.634 | 0.0156 |
| 0.3 | 0.726 | 0.643 | 0.0148 |
| 0.5 | 0.674 | 0.546 | 0.00275 |

#### Dose-response on articulated_consistency (switch arm; mean over sequences x seeds)

| p_injected | mean articulated_consistency |
|---|---|
| 0.0 | 1.000 |
| 0.1 | 0.928 |
| 0.2 | 0.856 |
| 0.3 | 0.767 |
| 0.5 | 0.626 |

Paired Wilcoxon (adjacent levels, paired by sequence+seed):

| level_a vs level_b | W | p |
|---|---|---|
| 0.0 vs 0.1 | 0.0 | 7.1e-07 |
| 0.1 vs 0.2 | 0.0 | 7.43e-07 |
| 0.2 vs 0.3 | 0.0 | 7.43e-07 |
| 0.3 vs 0.5 | 0.0 | 7.39e-07 |

#### Switch vs drift (APE-matched) on articulated_consistency, at each injected level

| p_injected | switch mean | drift mean | Wilcoxon p |
|---|---|---|---|
| 0.1 | 0.928 | 0.985 | 7.86e-07 |
| 0.2 | 0.856 | 0.974 | 7.95e-07 |
| 0.3 | 0.767 | 0.944 | 4.66e-10 |
| 0.5 | 0.626 | 0.915 | 4.66e-10 |

#### Natural-tracker repair effect on seg_iou (paired raw vs repaired)

| tracker | raw mean | repaired mean | Wilcoxon p |
|---|---|---|---|
| cotracker3 | 0.926 | 0.934 | 0.893 |
| cotracker2 | 0.817 | 0.817 | nan |

#### Natural-tracker repair effect on ari (paired raw vs repaired)

| tracker | raw mean | repaired mean | Wilcoxon p |
|---|---|---|---|
| cotracker3 | 0.934 | 0.891 | 0.109 |
| cotracker2 | 1.000 | 1.000 | nan |

#### Natural-tracker repair effect on boundary_f1 (paired raw vs repaired)

| tracker | raw mean | repaired mean | Wilcoxon p |
|---|---|---|---|
| cotracker3 | 0.947 | 0.962 | 0.893 |
| cotracker2 | 0.833 | 0.833 | nan |
