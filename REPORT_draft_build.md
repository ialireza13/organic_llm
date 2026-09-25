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
