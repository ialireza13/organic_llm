### Scale S

| arm | seeds | final val loss (mean ± std) | per-seed | train FLOPs | metric s (total) | tokens/s | final heads / FFN per layer (seed 0) |
|---|---|---|---|---|---|---|---|
| (b) random growth | 1 | 4.2016 ± nan | 4.2016 | 4.795e+16 | 0 | 114998 | [5, 7, 6, 8, 3, 7] / [34, 23, 25, 21, 23, 18]×64 |
| (c) uniform growth | 1 | 4.2020 ± nan | 4.2020 | 4.795e+16 | 0 | 115423 | [6, 6, 6, 6, 6, 6] / [24, 24, 24, 24, 24, 24]×64 |
| (d) scratch, same tokens | 1 | 4.1764 ± nan | 4.1764 | 5.198e+16 | 0 | 213296 | [6, 6, 6, 6, 6, 6] / [24, 24, 24, 24, 24, 24]×64 |
| (e) scratch, matched FLOPs | 1 | 4.2567 ± nan | 4.2567 | 4.795e+16 | 0 | 158962 | [6, 6, 6, 6, 6, 6] / [24, 24, 24, 24, 24, 24]×64 |
| (f) OpenELM-style widths | 1 | 4.2017 ± nan | 4.2017 | 5.198e+16 | 0 | 128656 | [3, 4, 5, 7, 8, 9] / [12, 17, 22, 26, 31, 36]×64 |

Paired differences (same seed = same data order; negative = first arm better):

| comparison | n | mean ± std | per seed |
|---|---|---|---|
| (b) random growth − (c) uniform growth | 1 | -0.0004 ± nan | -0.0004 |
| (c) uniform growth − (d) scratch, same tokens | 1 | +0.0256 ± nan | +0.0256 |
| (c) uniform growth − (e) scratch, matched FLOPs | 1 | -0.0548 ± nan | -0.0548 |
| (f) OpenELM-style widths − (d) scratch, same tokens | 1 | +0.0253 ± nan | +0.0253 |
