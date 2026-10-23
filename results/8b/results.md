| Metric | base | sft | Δ sft vs base |
|---|---|---|---|
| Grounded accuracy ↑ | 37.0 [30.5, 43.5] | 65.0 [58.5, 71.5] | +28.0 [+20.5, +36.0] * |
| Abstains when unanswerable ↑ | 33.3 [21.9, 46.4] | 98.3 [94.3, 100.0] | +65.0 [+53.5, +76.1] * |
| Refuses when answerable ↓ | 0.7 [0.0, 2.3] | 0.7 [0.0, 2.3] | +0.0 [+0.0, +0.0] |
| Abstention F1 ↑ | 49.4 [35.2, 62.5] | 98.3 [95.5, 100.0] | +49.0 [+36.8, +62.5] * |
| Claims supported by context ↑ | 74.8 [68.5, 80.7] | 90.1 [86.4, 94.1] | +15.3 [+8.9, +22.2] * |
| Answers with every claim supported ↑ | 70.9 [63.7, 77.2] | 87.1 [81.5, 92.4] | +16.2 [+7.9, +24.3] * |
| Consistent with reference ↑ | 53.2 [44.8, 61.8] | 66.9 [59.2, 75.0] | +13.7 [+5.0, +21.7] * |
| Covers the reference ↑ | 65.9 [58.5, 72.5] | 73.0 [65.9, 79.3] | +7.1 [+0.9, +13.8] * |
| Token F1 vs reference ↑ | 67.6 [63.8, 71.5] | 77.3 [73.8, 80.9] | +9.7 [+6.6, +12.9] * |
| Claims per answer | 1.66 | 1.59 | – |

Values are percentages with 95% bootstrap CIs (1000 resamples). Examples per run: base: 200, sft: 200.
Δ columns are paired bootstrap differences on shared examples; * marks a CI that excludes 0.
