# Organic growth of transformers — Stage 0–2 overnight report

2026-09-25 · overnight autonomous run (00:44–10:44 EDT) · single H100 NVL · status log: [STATUS.md](STATUS.md) · every judgement call: [DECISIONS.md](DECISIONS.md)

## 1. TL;DR

__TLDR__

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

__STAGE2__

## 5. Plots

__PLOTS__

## 6. What failed, was cut, or deviated from the prompt

__DEVIATIONS__

## 7. Recommended next steps (Stage 3)

__NEXT__
