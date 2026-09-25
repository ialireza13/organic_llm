### Scale S

| arm | seeds | final val loss (mean ± std) | per-seed | train FLOPs | metric s (total) | tokens/s | final heads / FFN per layer (seed 0) |
|---|---|---|---|---|---|---|---|
| (a) metric growth [m1b] | 3 | 4.2122 ± 0.0126 | 4.2004, 4.2255, 4.2107 | 4.761e+16 | 24 | 163214 | [9, 9, 5, 5, 5, 3] / [22, 29, 26, 23, 24, 20]×64 |
| (a) metric growth [m3neg] | 3 | 4.1967 ± 0.0097 | 4.1905, 4.2079, 4.1916 | 4.847e+16 | 107 | 156778 | [3, 7, 7, 7, 5, 7] / [22, 22, 24, 23, 27, 26]×64 |
| (a) metric growth [m6] | 3 | 4.1929 ± 0.0078 | 4.1842, 4.1994, 4.1951 | 4.860e+16 | 62 | 159195 | [10, 4, 6, 5, 7, 4] / [41, 22, 19, 21, 21, 20]×64 |
| (b) random growth | 3 | 4.2073 ± 0.0134 | 4.2016, 4.2226, 4.1977 | 4.795e+16 | 0 | 131250 | [5, 7, 6, 8, 3, 7] / [34, 23, 25, 21, 23, 18]×64 |
| (c) uniform growth | 3 | 4.2023 ± 0.0075 | 4.2020, 4.2099, 4.1950 | 4.795e+16 | 0 | 131516 | [6, 6, 6, 6, 6, 6] / [24, 24, 24, 24, 24, 24]×64 |
| (d) scratch, same tokens | 3 | 4.1893 ± 0.0276 | 4.1764, 4.1706, 4.2210 | 5.198e+16 | 0 | 216091 | [6, 6, 6, 6, 6, 6] / [24, 24, 24, 24, 24, 24]×64 |
| (e) scratch, matched FLOPs | 2 | 4.2570 ± 0.0004 | 4.2567, 4.2573 | 4.795e+16 | 0 | 189498 | [6, 6, 6, 6, 6, 6] / [24, 24, 24, 24, 24, 24]×64 |
| (e') scratch, matched FLOPs, warmup 101 steps | 3 | 4.1933 ± 0.0074 | 4.1962, 4.1987, 4.1849 | 4.795e+16 | 0 | 214484 | [6, 6, 6, 6, 6, 6] / [24, 24, 24, 24, 24, 24]×64 |
| (f) OpenELM-style widths | 3 | 4.2211 ± 0.0273 | 4.2017, 4.2093, 4.2523 | 5.198e+16 | 0 | 185781 | [3, 4, 5, 7, 8, 9] / [12, 17, 22, 26, 31, 36]×64 |

Paired differences (same seed = same data order; negative = first arm better):

| comparison | n | mean ± std | per seed |
|---|---|---|---|
| (a) metric growth [m1b] − (b) random growth | 3 | +0.0049 ± 0.0073 | -0.0012, +0.0029, +0.0130 |
| (a) metric growth [m1b] − (c) uniform growth | 3 | +0.0099 ± 0.0099 | -0.0016, +0.0156, +0.0157 |
| (a) metric growth [m1b] − (d) scratch, same tokens | 3 | +0.0229 ± 0.0326 | +0.0240, +0.0549, -0.0103 |
| (a) metric growth [m1b] − (e) scratch, matched FLOPs | 2 | -0.0441 ± 0.0174 | -0.0564, -0.0318 |
| (a) metric growth [m1b] − (e') scratch, matched FLOPs, warmup 101 steps | 3 | +0.0189 ± 0.0128 | +0.0042, +0.0268, +0.0259 |
| (a) metric growth [m1b] − (f) OpenELM-style widths | 3 | -0.0089 ± 0.0296 | -0.0013, +0.0162, -0.0416 |
| (a) metric growth [m3neg] − (b) random growth | 3 | -0.0106 ± 0.0043 | -0.0110, -0.0148, -0.0061 |
| (a) metric growth [m3neg] − (c) uniform growth | 3 | -0.0056 ± 0.0051 | -0.0114, -0.0020, -0.0034 |
| (a) metric growth [m3neg] − (d) scratch, same tokens | 3 | +0.0073 ± 0.0339 | +0.0141, +0.0373, -0.0294 |
| (a) metric growth [m3neg] − (e) scratch, matched FLOPs | 2 | -0.0578 ± 0.0119 | -0.0662, -0.0494 |
| (a) metric growth [m3neg] − (e') scratch, matched FLOPs, warmup 101 steps | 3 | +0.0034 ± 0.0080 | -0.0057, +0.0092, +0.0068 |
| (a) metric growth [m3neg] − (f) OpenELM-style widths | 3 | -0.0244 ± 0.0318 | -0.0112, -0.0014, -0.0607 |
| (a) metric growth [m6] − (b) random growth | 3 | -0.0144 ± 0.0106 | -0.0173, -0.0233, -0.0026 |
| (a) metric growth [m6] − (c) uniform growth | 3 | -0.0094 ± 0.0090 | -0.0177, -0.0106, +0.0001 |
| (a) metric growth [m6] − (d) scratch, same tokens | 3 | +0.0036 ± 0.0276 | +0.0078, +0.0288, -0.0259 |
| (a) metric growth [m6] − (e) scratch, matched FLOPs | 2 | -0.0652 ± 0.0103 | -0.0725, -0.0579 |
| (a) metric growth [m6] − (e') scratch, matched FLOPs, warmup 101 steps | 3 | -0.0004 ± 0.0111 | -0.0120, +0.0007, +0.0103 |
| (a) metric growth [m6] − (f) OpenELM-style widths | 3 | -0.0282 ± 0.0254 | -0.0175, -0.0099, -0.0572 |
| (b) random growth − (c) uniform growth | 3 | +0.0050 ± 0.0068 | -0.0004, +0.0127, +0.0027 |
| (c) uniform growth − (d) scratch, same tokens | 3 | +0.0130 ± 0.0345 | +0.0256, +0.0394, -0.0261 |
| (c) uniform growth − (e) scratch, matched FLOPs | 2 | -0.0511 ± 0.0052 | -0.0548, -0.0474 |
| (c) uniform growth − (e') scratch, matched FLOPs, warmup 101 steps | 3 | +0.0090 ± 0.0029 | +0.0058, +0.0112, +0.0101 |
| (e') scratch, matched FLOPs, warmup 101 steps − (e) scratch, matched FLOPs | 2 | -0.0596 ± 0.0014 | -0.0605, -0.0586 |
| (f) OpenELM-style widths − (d) scratch, same tokens | 3 | +0.0318 ± 0.0067 | +0.0253, +0.0387, +0.0313 |
