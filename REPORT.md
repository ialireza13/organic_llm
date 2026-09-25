# Organic growth of transformers — Stage 0–2 overnight report

2026-09-25 · overnight autonomous run (00:44–10:44 EDT) · single H100 NVL · status log: [STATUS.md](STATUS.md) · every judgement call: [DECISIONS.md](DECISIONS.md)

## 1. TL;DR

- **Metrics:** no metric predicted the single-unit growth oracle. Pooled Spearman was −0.19 (M1b) to −0.64 (M3, significantly *anti*-correlated), and the oracle itself is only weakly reliable (seed-to-seed ρ 0.20 / 0.64). **Drop M1, M1b, M2, M5, M7 as location selectors.** Keep only **M6** and the **"−M3 / late-layer" allocation** as hypotheses.
- **Metric vs random/uniform growth: not beyond seed noise.** At S, M6 growth beat uniform growth by 0.009 ± 0.007 nats (4 seeds, p = 0.08) and random growth by 0.014 (p = 0.14). M1b, the formally top-ranked metric, was 0.010 *worse* than uniform. At M (1 seed, reduced budget), every metric arm was 0.013–0.017 worse than uniform growth.
- **Growth itself doesn't pay at S:** uniform growth loses to a from-scratch target-size model trained at matched FLOPs (+0.0125 ± 0.007, 4 seeds, p = 0.04). At S, width growth saves only 8% of FLOPs because the tied LM head dominates. At M (1 seed, undertrained) growth beat the matched-FLOPs baseline by 0.05. Promising but unconfirmed.
- **Update after extra seeds (see §8):** with 6 seeds, M6 vs uniform at S shrinks to −0.005 ± 0.010 (p = 0.31), i.e. noise. At M with 2 seeds, uniform growth beats the matched-FLOPs baseline in both seeds (−0.046 ± 0.007, p = 0.07).
- Methodology warning: the first matched-FLOPs baseline was 0.06 worse purely because of its warmup, at an edge-of-stability LR. Comparisons of this size need a more stable LR and ≥4–5 seeds.

## 2. What was built, and test results

Everything is in this repo; experiments ran on `server1` (1× H100 NVL, power-capped at 370 W).

| Module | What it does |
| --- | --- |
| `data_prep/prepare_fineweb.py` | FineWeb `sample-10BT` (first 3 parquet files) → GPT-2 tiktoken → uint16 shards: 1.5B train tokens, 10M val tokens ([0, 8M) evaluation, [8M, 10M) held-out batches for metrics). Took 100 s with 32 workers. |
| `data.py` | Deterministic batches: a permutation of 1024-token windows seeded by the run seed; step *s* always gets the same batch. Arms with the same seed see identical data (paired comparisons); oracle continuations replay the base run's data order. |
| `model/gpt.py` | Pre-LN GPT, GELU FFN, tied embeddings, no linear biases, bf16 autocast, `torch.compile`, flash SDPA. Every layer pre-allocates `max_heads` × 64-dim heads and `max_ffn` neurons; each unit has a mask `clamp((step − birth)/ramp, 0, 1)` (FFN: on the hidden activation; heads: on the head output before O). Static shapes → one compile per run. |
| `growth/operators.py` | `grow_ffn` (+64 neurons) and `grow_head` (+1 head): take the lowest free slots, re-init with the standard init (N(0, 0.02); output side N(0, 0.02/√(2L))), reset AdamW state, set birth = current step (mask 0 → ramps to 1 over 2% of steps). |
| `growth/optimizer.py` | AdamW with a *per-element* step counter, so moments *and* bias correction reset exactly for grown slices. |
| `growth/controller.py` | 8 growth events evenly spaced over 10–60% of training; uniform / random / metric policies (see DECISIONS.md for the allocation rule); logs every event (step, unit, score, params). |
| `metrics/` | M1 GradMax, M1b mask gradient, M2 splitting eigenvalue (FFN only), M3 TINY bottleneck, M5 Adam SNR, M6 spectral saturation × (1 − dormant fraction), M7 sublayer influence. All share `score(model, batches, opt) -> {"ffn:l"/"head:l": score}`. |
| `oracle/` | Stage 1: `run_oracle.py` (grow one candidate, continue with the same data order and LR schedule, evaluate on 4M val tokens), `metrics_at_ckpt.py`, `analyze.py` (Spearman, reliability, random baseline). |
| `train.py` | Training loop for all arms; active-parameter FLOP counter; checkpoints; CSV logs to `results/runs/<run>/`. |
| `analysis/stage2.py` | Stage 2 tables (incl. paired per-seed differences) and plots. |
| `scripts/queue.py` | Restart-safe job queue on the server (tmux), so runs don't depend on the laptop connection. |

**Required tests: all pass** (`pytest tests/`, 14 tests + a 200-step S smoke run, on the server GPU):

| Test | Result |
| --- | --- |
| Function preservation (every operator: FFN chunk and head, every layer; fp32 logits before/after, atol 1e-5) | pass (bit-identical: inactive units are multiplied by an exact 0) |
| Inactive units don't affect the output (inactive weights set to 10× random) | pass |
| New units get zero gradient at mask = 0 and non-zero gradients for every new parameter once the mask ramps above 0 | pass |
| Every parameter is in the optimizer; grown slices get `exp_avg`, `exp_avg_sq` *and* the per-element step reset to 0; other slices untouched; the first post-growth update uses step = 1 bias correction | pass |
| Checkpoint round-trip exact (weights, masks, births, optimizer, controller state incl. RNG and EMA; next training step bit-identical) | pass |
| Crash/restart resume (added after the server restarted mid-run) | pass |
| FLOP counter vs hand calculation: S target = 6·29,933,568 + 6·12·384·1024 = **207,912,960 FLOPs/token**; +1 head +1 chunk adds exactly 6·(4·384·64 + 2·384·64) + 12·64·1024 | pass |
| All 7 metrics return finite scores for every unit and leave no gradients behind | pass |
| 200-step S smoke run with M1-driven growth through all 8 events: no loss spike after any ramp (post-ramp loss < pre-event loss + 0.15), ends at exactly 36 heads / 9216 FFN | pass |

## 3. Stage 1 — oracle validation (S)

**Setup.** One S model at 50% width (3 heads, 768 FFN per layer; lr 4e-3, the full 2543-step cosine schedule), checkpointed at 30% (step 763) and 60% (step 1526). At each checkpoint there are 12 candidates (+64 FFN neurons or +1 head, in each of the 6 layers) plus a no-growth control. Each is grown at step 0 (30-step ramp), trained with the base run's data order and LR schedule, and evaluated on the same fixed 4M validation tokens. Oracle = (control − candidate) val loss per added parameter. There are 2 seeds of new-unit init, and each seed has its own control run.

**Oracle reliability (checked first).** With 200 continuation steps the oracle was noise: at 60% every candidate was within ±2e-4 of control, and two identical control runs differed by 0.9e-4 (GPU nondeterminism amplified by training). As the prompt allows, I increased the continuation **once, to 400 steps** (600 was too slow on the shared GPU; see DECISIONS.md). The 400-step oracle is the reference below.

| Checkpoint | Oracle seed-to-seed Spearman / Pearson (n = 12) | Identical controls differ by |
| --- | --- | --- |
| 30% | **+0.20** / +0.27 (weak) | 2.9e-4 |
| 60% | **+0.64** / +0.51 (moderate) | 1.6e-4 |

Single-unit effects are tiny: at most ~1.8e-3 nats at 30% and ~2.7e-4 at 60%. The only structure that replicates across seeds: at 30%, extra **heads in the upper half (layers 3–5)** help most (+1.0e-3 to +1.8e-3 in both seeds), and so does an FFN chunk in the last layer. Early-layer candidates help little.

**Metric vs oracle (400-step oracle; Spearman between the metric's ranking and the mean-over-seeds oracle ranking).** Each metric is averaged over 3 disjoint held-out sets of 64k tokens. "self-rel." is the Spearman between two of those repeats: **all metrics are very stable (≥0.94), so the low correlations reflect what the metrics measure, not metric noise.** s/eval is wall-clock per 64k-token evaluation, measured while the GPU was shared with 7 other jobs (isolated cost is ~3–5× lower).

| Metric | ρ @30% | ρ @60% | **ρ pooled (24)** | ρ FFN only (12) | ρ heads only (12) | self-rel. | s / eval |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M1 GradMax | −0.55 | −0.33 | **−0.35** | −0.41 | −0.31 | 0.98 | 4.5 |
| M1b mask gradient | −0.20 | −0.30 | **−0.19** | −0.24 | −0.31 | 0.95 | 1.3 |
| M2 splitting (FFN only) | n/a | n/a | n/a | **−0.40** | n/a | 0.94 | 1.7 |
| M3 TINY bottleneck | −0.39 | −0.55 | **−0.64** | −0.57 | −0.68 | 0.99 | 8.3 |
| M5 Adam grad SNR | −0.27 | −0.06 | **−0.31** | −0.51 | −0.65 | 1.00 | 0.3 |
| M6 spectral saturation | +0.06 | −0.32 | **−0.23** | −0.48 | −0.34 | 1.00 | 2.7 |
| M7 block influence | +0.15 | −0.24 | **−0.28** | −0.15 | −0.64 | 0.97 | 1.3 |
| random ranking | 0 ± 0.30 | 0 ± 0.30 | 0 ± 0.21 (95th pct +0.34) | 0 ± 0.30 | 0 ± 0.30 | – | 0 |

The same analysis on the 200-step values of the same runs ([table](results/oracle/stage1_table_oracle400_at200.md)) and on the original 200-step runs ([table](results/oracle/stage1_table_oracle.md)) gives the same picture: no metric is reliably positive, and signs flip between checkpoints and continuation lengths.

**Reading.**
- **No metric passes the Stage-1 go criterion** ("clear positive Spearman"). Every pooled ρ is negative. Five of six are inside or near the random band (±0.21 sd).
- **M3 (TINY) is significantly *anti*-correlated** (−0.64, beyond the random 95th percentile), and M1, M2 and M5 lean the same way. The gradient-based metrics rate the *early* layers as the most "needy" (large residual-stream gradients and bottlenecks at the bottom of the network), while the oracle says the *late* layers benefit most from an extra unit. One plausible reason: a first-order score measures how fast the loss *could* fall along a new direction at init. A unit's actual value after 400 steps of AdamW training depends on what it learns, and upper layers may use new capacity better.
- The oracle itself is weak at 30% and only moderate at 60%. With 12 candidates, even a perfect metric could only reach ρ ≈ √(oracle reliability) ≈ 0.45–0.8 against it. The signal-to-noise problem is structural: one unit is ~0.5% of the model.
- **Selection for Stage 2, per protocol (top 2 by pooled ρ): M1b (−0.19) and M6 (−0.23).** Both are indistinguishable from random ranking, so Stage 2 effectively compares two "non-informative" metric policies against random and uniform growth. I also ran an **exploratory, post-hoc arm with −M3** (grow where the TINY bottleneck is *smallest*). It is the only ranking that tracked the oracle (ρ = +0.64 in-sample), so Stage 2 is its out-of-sample test.

Plots: [oracle values per candidate](results/plots/stage1_oracle_values.png), [metric vs oracle ranks](results/plots/stage1_metric_vs_oracle.png). Raw data: `results/oracle/`.

## 4. Stage 2 — ablation

All arms at a scale share the token budget (except (e)/(e′)), the LR, the data order per seed, and, for growth arms, the growth schedule: 8 events over 10–60% of training, start at 50% width, end at exactly the target total heads/FFN. Final parameter counts are identical across growth arms and equal to (d). "Paired" differences compare arms on the same seed (same data order). The validation loss is on 8M held-out tokens.

### S (6 layers, d = 384; 250M tokens; lr 4e-3; 3 seeds, 4 for the closest comparison)

| Arm | seeds | final val loss, mean ± sd | per seed | train FLOPs | isolated tokens/s ¹ |
| --- | --- | --- | --- | --- | --- |
| (a1) metric growth, **M1b** | 3 | 4.2122 ± 0.0126 | 4.2004, 4.2255, 4.2107 | 4.76e16 | 0.79M |
| (a2) metric growth, **M6** | 4 | **4.1945 ± 0.0071** | 4.1842, 4.1994, 4.1951, 4.1994 | 4.86e16 | 0.79M |
| (a3, exploratory) growth, **−M3** | 3 | 4.1967 ± 0.0097 | 4.1905, 4.2079, 4.1916 | 4.85e16 | 0.79M |
| (b) random growth | 3 | 4.2073 ± 0.0134 | 4.2016, 4.2226, 4.1977 | 4.80e16 | 0.79M |
| (c) uniform growth | 4 | 4.2039 ± 0.0069 | 4.2020, 4.2099, 4.1950, 4.2088 | 4.80e16 | 0.79M |
| (d) scratch at target size, same tokens | 3 | 4.1893 ± 0.0276 | 4.1764, 4.1706, 4.2210 | 5.20e16 | 0.99M |
| (e) scratch, matched FLOPs (as specified: warmup 4% of 2346 steps) | 2 | 4.2570 ± 0.0004 | 4.2567, 4.2573 | 4.80e16 | 0.99M |
| **(e′) scratch, matched FLOPs, same 101-step warmup** | 4 | **4.1914 ± 0.0071** | 4.1962, 4.1987, 4.1849, 4.1858 | 4.80e16 | 0.99M |
| (f) OpenELM-style linear widths (from scratch, same total size) | 3 | 4.2211 ± 0.0273 | 4.2017, 4.2093, 4.2523 | 5.20e16 | 0.99M |

¹ Measured afterwards on the otherwise idle GPU (`scripts/bench.py`; results/bench_S.json). The per-run tokens/s in `summary.json` were measured with 4–8 jobs sharing the GPU and are not comparable across arms. Growth arms are 20% slower per token than a target-size model only because the static-shape implementation always computes the 2×-target pre-allocated weights.

**Paired differences at S** (negative = first arm better; two-sided paired t-test, uncorrected for the ~25 comparisons in [results/stage2_tables.md](results/stage2_tables.md)):

| Comparison | n | Δ mean ± sd | p | seeds where first arm is better |
| --- | --- | --- | --- | --- |
| M6 growth − uniform growth | 4 | −0.0094 ± 0.0073 | 0.08 | 3/4 (one tie) |
| M6 growth − random growth | 3 | −0.0144 ± 0.0106 | 0.14 | 3/3 |
| M1b growth − uniform growth | 3 | +0.0099 ± 0.0099 | 0.23 | 1/3 |
| M1b growth − random growth | 3 | +0.0049 ± 0.0073 | 0.37 | 1/3 |
| −M3 growth − uniform growth (exploratory) | 3 | −0.0056 ± 0.0051 | 0.20 | 3/3 |
| −M3 growth − random growth (exploratory) | 3 | −0.0106 ± 0.0043 | 0.05 | 3/3 |
| random growth − uniform growth | 3 | +0.0050 ± 0.0068 | 0.33 | 1/3 |
| uniform growth − (e′) scratch, matched FLOPs (**Claim A**) | 4 | **+0.0125 ± 0.0073** | **0.04** | 0/4 |
| M6 growth − (e′) scratch, matched FLOPs | 4 | +0.0031 ± 0.0115 | 0.62 | 1/4 |
| uniform growth − (d) scratch, same tokens | 3 | +0.0130 ± 0.0345 | 0.58 | 1/3 |
| (f) OpenELM − (d) scratch | 3 | +0.0318 ± 0.0067 | 0.01 | 0/3 |
| (e′) − (e): same run, warmup 101 vs 93 steps | 2 | −0.0596 ± 0.0014 | 0.01 | 2/2 |

### M (8 layers, d = 640; **300M tokens, 1 seed**, lr 2.5e-3; reduced scope, see §6)

| Arm | final val loss | Δ vs uniform growth | train FLOPs | isolated tokens/s |
| --- | --- | --- | --- | --- |
| (a1) metric growth, M1b | 3.9991 | +0.0137 | 1.31e17 | 0.38M |
| (a2) metric growth, M6 | 3.9986 | +0.0132 | 1.34e17 | 0.38M |
| (a3, exploratory) growth, −M3 | 4.0025 | +0.0171 | 1.33e17 | 0.38M |
| (b) random growth | 3.9920 | +0.0067 | 1.32e17 | 0.38M |
| (c) uniform growth | **3.9853** | 0 | 1.32e17 | 0.38M |
| (d) scratch, same tokens | **3.9767** | −0.0086 | 1.48e17 | 0.51M |
| (e) scratch, matched FLOPs (same 91-step warmup) | 4.0359 | +0.0505 | 1.32e17 | 0.51M |

(f) was not run at M. With one seed, and an S seed-to-seed sd of ~0.007–0.013 for growth arms, the M metric-vs-uniform gaps (0.013–0.017) are suggestive at most. Their sign is *opposite* to S for M6 and −M3.

### What this says

- **Does the metric earn its place? No, not on this evidence.** The protocol-selected metrics split: M6 looks mildly better than uniform/random at S (p = 0.08 / 0.14), while M1b is worse than both. At M, all metric arms lose to uniform growth. Uniform growth ≥ random growth at both scales (by 0.005–0.007).
- **The exploratory −M3 arm**, the only ranking that tracked the Stage-1 oracle, beat random growth in 3/3 S seeds (p = 0.05, uncorrected and post hoc) but lost to uniform at M. That is weak support for the oracle's "grow the upper layers" signal. Not a result.
- **Timing confound.** The per-event head/FFN split is decided by the metric. M6 and −M3 put all 18 new heads into the first 3–4 events; M1b put most heads into the last 2 events (see `growth_events.csv`). At S the head-early arms did better. This may matter more than *which layer* got the unit, and it did not carry over to M. A fixed-split control would separate the two.
- **Allocations** ([heatmap](results/plots/S_allocation_heatmaps.png)) are very reproducible across seeds for metric arms (−M3 gives identical head allocations in all 3 seeds). M6 and M1b both put 9–12 heads (the maximum is 12) into layer 0, and M6 also puts 37–43 FFN chunks there (vs 24 for uniform). −M3 adds heads to layers 1–5 and FFN to the top layers.
- **Claim A (compute efficiency).** At S, no: the FLOP-matched from-scratch model (e′) is 0.0125 better than uniform growth in 4/4 seeds. At S the tied embedding/LM head is ~2/3 of per-token FLOPs, so starting at 50% width saves only 8% of FLOPs, too little to pay for training a smaller model for the first half. At M (1 seed; 11% FLOP saving; undertrained at ~8 tokens/param) growth beats the FLOP-matched baseline by 0.05. That is worth checking with seeds and at GPT-2 scale, where the non-embedding share is larger.
- **Claim B (parameter efficiency at equal FLOPs):** not supported. The best growth arm (M6) ties (e′) at S (+0.003 ± 0.012) with the same parameter count and FLOPs. OpenELM-style fixed non-uniform widths were clearly *worse* than uniform (+0.032, p = 0.01).
- **Noise caveat.** Full-width from-scratch runs at lr 4e-3 are much noisier (sd 0.028) than growth runs (sd ~0.007–0.013), because 4e-3 is near the stability edge: one bad seed (seed 2) hurt both (d) and (f). The (e) vs (e′) warmup artifact is the same phenomenon.

## 5. Plots

- Validation loss vs tokens, per arm (seed mean bold, seeds faint): [S](results/plots/S_val_vs_tokens.png), [M](results/plots/M_val_vs_tokens.png)
- Validation loss vs training FLOPs: [S](results/plots/S_val_vs_flops.png), [M](results/plots/M_val_vs_flops.png)
- Per-layer allocation heatmaps (heads, FFN chunks) for metric-driven and random arms: [S](results/plots/S_allocation_heatmaps.png), [M](results/plots/M_allocation_heatmaps.png)
- Stage 1: [oracle values per candidate](results/plots/stage1_oracle_values.png), [metric vs oracle ranks](results/plots/stage1_metric_vs_oracle.png)

The periodic val points use the first 2M val tokens and the final point uses all 8M, hence the small hook at the end of every curve.

Per-run artifacts: `results/runs/<run>/` (config.json, train_log.csv, val_log.csv, growth_events.csv, metric_log.json, summary.json). LR sweep: `results/lrsweep/`. Stage 1: `results/oracle/`.

## 6. What failed, was cut, or deviated from the prompt

Full log with alternatives: [DECISIONS.md](DECISIONS.md). Timeline and incidents: [STATUS.md](STATUS.md).

**Failed / incidents**
- **Server container restart** (~02:38 EDT, during a 7-minute network outage) killed every running job: 4 Stage-2 runs (one at step 2400/2543) and the 4 oracle processes. About 40 min of GPU time was lost. I then added periodic resume checkpoints and auto-resume to `train.py` (tested).
- **OOM, my fault** (01:35): an ad-hoc diagnostic run I started pushed GPU memory over 93 GB and killed two Stage-1 jobs. They were rerun.
- **200-step oracle was noise**, and 600 steps was too slow on the shared GPU. The oracle was rerun once at 400 steps; it is only weakly reliable at the 30% checkpoint (ρ = 0.20).
- **Arm (e) as first specified is confounded**: with warmup = 4% of its own (shorter) schedule, it trained into a worse regime at this edge-of-stability LR and ended 0.06 worse than the same run with the same *absolute* warmup (e′). (e) is reported, but (e′) is the valid matched-FLOPs baseline; seed 2 of (e) was dropped.

**Cut**
- **M scale reduced**: 300M tokens instead of 800M, **1 seed** instead of 2, batch 128 (2288 steps), LR 2.5e-3 (S optimum scaled ∝ 1/width, not re-tuned).
- M4 skipped (as instructed). M2 splitting was only scored on FFN (by design) and was not selected.
- No isolated per-arm tokens/s benchmark: the GPU was never idle. The logged tokens/s were measured with 4–8 jobs sharing a power-capped GPU and are only roughly comparable (see §4).

**Deviations / judgement calls that matter for interpretation**
- LR 4e-3 (sweep optimum) is close to the instability edge (8e-3 diverged). Full-width from-scratch runs are therefore much noisier across seeds (sd 0.028) than growth runs, which start small (sd ~0.008).
- The per-event split between heads and FFN is decided by the metric (per the prompt), so metric arms add heads at different times than uniform/random (M6 and −M3 early, M1b late) and their training FLOPs differ by −0.7% to +1.4% (attention term). Final parameter counts are identical across growth arms.
- Stage-2 metric selection followed the protocol (top 2 pooled ρ = M1b, M6) even though neither passed the go criterion. The extra −M3 arm is exploratory and was chosen post hoc.
- Metric scores are EMA-smoothed (β = 0.5) over evaluations every ~60 steps on 64k held-out tokens. Metric compute is logged separately (M6: ~60 s per run, −M3: ~110 s, M1b: ~25 s, vs ~25–30 min runs). It is not included in train FLOPs.
- Tokens/s for growth arms reflect the static-shape implementation: every run computes the 2×-target pre-allocated model regardless of how many units are active, so growth runs are ~1.3–1.5× slower per token than a target-size model. This is an implementation cost, not intrinsic to growth.

## 7. Recommended next steps (Stage 3)

1. **Do not carry the current metric set forward as-is.** None of M1, M1b, M2, M3, M5, M6, M7 predicted the single-unit oracle, and M1b (the Stage-2 "winner") did not beat random growth. If a metric goes to Stage 3, carry **M6** (cheap; better than uniform in 3/4 S seeds, p = 0.08, but worse at M) and the **−M3 / "late-layer" allocation** (the only ranking that tracked the oracle; better than random in 3/3 S seeds, p = 0.05 uncorrected, worse at M), and treat both as hypotheses, not results.
2. **Fix the oracle before trusting any metric ranking.** Single units (0.5% of params) give effects the size of run-to-run noise. Options: grow larger candidate units (e.g. +25% of a layer); average ≥4 init seeds; use deterministic kernels so control noise is 0; or measure the oracle at the end of a full schedule instead of after 400 steps.
3. **Settle Claim A before scaling up.** At S, growth does not beat a FLOP-matched scratch model (uniform growth is 0.0125 worse than (e′), 4/4 seeds). At M it did (0.05, 1 seed): rerun M with ≥3 seeds and the full 800M tokens first. Part of the reason: at S the tied LM head is ~2/3 of the FLOPs, so growing only width saves just 8%. At GPT-2 scale (124M/350M) the non-embedding share is larger, so the potential saving is too. Re-test there with a compact (gathered) implementation so wall-clock savings are real.
4. **Use a more stable LR** (or LR re-warmup for new units) in the next round, e.g. 2e-3–3e-3 at S. At 4e-3, seed noise for full-width models (sd 0.028) swamps the effects we want to measure. Use ≥4–5 seeds for any comparison under 0.01 nats.
5. **Separate "when" from "where".** Rerun the metric arms with the head/FFN split fixed per event (as in uniform/random), so the metric chooses only layers. Then The two better-than-random policies made very different choices: M6 piles heads and FFN into layer 0, while −M3 adds heads to layers 1–5 and FFN to the top layers. They also differed in *timing* (heads early). So "any consistent non-uniform allocation, or early heads, beats random" is as plausible as "the metric matters". A fixed-allocation control (e.g. the final −M3 or M6 allocation, trained from scratch) would separate "growth path" from "final shape".

## 8. Added after report (extra seeds, 07:28–08:46 EDT)

After the report was committed I used the remaining time for extra seeds on the closest or most decision-relevant comparisons: S seeds 4–5 for M6 and uniform growth, and M seed 1 for scratch, uniform growth, matched-FLOPs scratch (e) and M6. All finished; tables and plots in `results/` have been regenerated.

**S, M6 vs uniform growth (6 seeds):** M6 4.1966 ± 0.0065, uniform 4.2012 ± 0.0071. Paired Δ = **−0.0046 ± 0.0099, p = 0.31** (per seed: −0.018, −0.011, 0.000, −0.009, +0.010, 0.000). The 4-seed result (−0.009, p = 0.08) does not hold up: **M6 is not distinguishable from uniform growth.**

**M (now 2 seeds for these arms):**

| Arm | seed 0 | seed 1 | mean ± sd |
| --- | --- | --- | --- |
| (d) scratch, same tokens | 3.9767 | 3.9741 | 3.9754 ± 0.0019 |
| (c) uniform growth | 3.9853 | 3.9840 | 3.9847 ± 0.0010 |
| (a2) M6 growth | 3.9986 | 3.9885 | 3.9935 ± 0.0071 |
| (e) scratch, matched FLOPs (91-step warmup) | 4.0359 | 4.0250 | 4.0304 ± 0.0077 |

| Paired comparison (M, 2 seeds) | Δ mean ± sd | p |
| --- | --- | --- |
| M6 − uniform growth | +0.0089 ± 0.0061 | 0.29 |
| uniform growth − (e) matched-FLOPs scratch (**Claim A**) | **−0.0458 ± 0.0067** | 0.07 |
| M6 − (e) matched-FLOPs scratch | −0.0369 ± 0.0006 | 0.01 |
| uniform growth − (d) scratch, same tokens | +0.0092 ± 0.0009 | 0.04 |

**Updated reading.**
- **Metric-driven growth:** no evidence that any metric beats uniform growth at either scale. At S it is a wash over 6 seeds; at M, M6 is slightly worse in both seeds.
- **Claim A:** scale-dependent. At S, growth loses to a FLOP-matched from-scratch model (4 seeds). At M it wins by ~0.046 nats in both seeds, while costing ~0.009 vs a from-scratch model that uses 12% more FLOPs. The M runs are short (300M tokens, ~8 tokens/param), so this may be a short-training effect. Confirming it at the full 800M-token M budget and at 124M is the most useful next experiment.
