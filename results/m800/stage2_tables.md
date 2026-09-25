### Scale M

| arm | seeds | final val loss (mean ± std) | per-seed | train FLOPs | metric s (total) | tokens/s | final heads / FFN per layer (seed 0) |
|---|---|---|---|---|---|---|---|
| (c) uniform growth | 3 | 3.7388 ± 0.0023 | 3.7364, 3.7391, 3.7410 | 3.517e+17 | 0 | 156687 | [10, 10, 10, 10, 10, 10, 10, 10] / [40, 40, 40, 40, 40, 40, 40, 40]×64 |
| (d) scratch, same tokens | 3 | 3.6739 ± 0.0020 | 3.6717, 3.6756, 3.6743 | 3.936e+17 | 0 | 160807 | [10, 10, 10, 10, 10, 10, 10, 10] / [40, 40, 40, 40, 40, 40, 40, 40]×64 |
| (e) scratch, matched FLOPs | 3 | 3.6978 ± 0.0028 | 3.6947, 3.7002, 3.6984 | 3.517e+17 | 0 | 154430 | [10, 10, 10, 10, 10, 10, 10, 10] / [40, 40, 40, 40, 40, 40, 40, 40]×64 |

Paired differences (same seed = same data order; negative = first arm better):

| comparison | n | mean ± std | paired t-test p | per seed |
|---|---|---|---|---|
| (c) uniform growth − (d) scratch, same tokens | 3 | +0.0649 ± 0.0016 | 0.00 | +0.0647, +0.0635, +0.0667 |
| (c) uniform growth − (e) scratch, matched FLOPs | 3 | +0.0410 ± 0.0020 | 0.00 | +0.0416, +0.0388, +0.0427 |
