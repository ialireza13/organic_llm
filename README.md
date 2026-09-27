# Organic growth of transformers

**Idea:** start training a small transformer and grow it (add FFN neurons and attention heads) during training, letting a metric decide *where* to add capacity. The hope was to match a normally trained model with less compute, or beat it at equal size.

**Verdict: it didn't work.** Training the full-size model from scratch was as good or better in every fair comparison.

## What we tried

- **Growth mechanism:** new units enter behind a mask that ramps from 0 to 1, so the model's output is unchanged at the moment of growth. Models start at 50% width and grow in 8 steps between 10% and 60% of training.
- **Seven "where to grow" metrics:** GradMax, mask gradient, splitting eigenvalue, TINY bottleneck, Adam gradient SNR, spectral saturation, block influence.
- **Two model sizes** on FineWeb: S (10M non-embedding params) and M (39M).

## What we found

| Question | Answer |
| --- | --- |
| Does any metric predict where an added unit helps most? | **No.** All correlations with the measured benefit were ≤ 0 (TINY was significantly *negative*). |
| Does metric-guided growth beat growing every layer evenly? | **No.** Within noise at S (6 seeds); slightly worse at M. |
| Does growing beat training the full-size model from scratch with the same compute? | **No.** Growth is worse by 0.013 at S and 0.041 at M (full 800M-token budget, 3/3 seeds). |
| Does it save wall-clock time? | **No.** The static-shape implementation still computes masked units. |

- An early "win" at M came from a run that was too short: the small model learns faster at first, but the full-size model overtakes it once training is long enough.
- Growth itself behaves correctly: there are no loss spikes at growth events. The cost comes from spending a large part of training with less capacity.

## If you revisit this

Change the recipe rather than the scale:
- Grow much earlier (finish by ~15% of training).
- Start closer to full size.
- Initialize new units from existing ones instead of randomly.

Always compare against a from-scratch model with the same compute and the same warmup, using ≥3 seeds.

**Details:** [REPORT.md](REPORT.md) (full results), [DECISIONS.md](DECISIONS.md) (judgement calls), [PLAN.md](PLAN.md) (original brief), `results/` (logs and plots).
