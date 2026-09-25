# STATUS

**Current state (07:28 EDT):** REPORT.md written and committed. Now running post-report extra seeds (tmux `extra`, scripts/jobs_extra.tsv): M seed 1 for scratch, uniform, e, M6; S seeds 4–5 for M6 and uniform. Launch cutoff 09:14. Checkpoints/logs remain server-side in `~/organic-growth/runs/` (not synced); small results are in `results/`.

**Start time:** 2026-09-25 00:44 EDT (04:44 UTC). **Hard deadline:** 2026-09-25 10:44 EDT.

| Deadline | Milestone |
| --- | --- |
| 03:44 EDT | env, data, model + operators, tests passing |
| 05:44 EDT | metrics + Stage 1 oracle done |
| 09:14 EDT | stop launching Stage 2 runs |
| 09:44 EDT | Stage 2 done |
| 10:44 EDT | REPORT.md committed, GPU idle |

## Server
- `server1`: vast.ai container, 1× H100 NVL (95830 MiB, MIG disabled), driver 595.71.05, CUDA 13.2, 368 GB free disk, 224 visible CPUs (prompt says 28; using ≤32 workers), 1.4 TB RAM visible, Python 3.12.3.

## Log
- 00:44 SSH ok, server inspected. Creating venv (`~/organic-growth/.venv`, uv).
- 00:47 venv ready (torch 2.14.0+cu130, CUDA ok). Data: FineWeb `sample-10BT` files 000–002 → 1.5B train tokens (15×100M uint16 shards) + 10M val tokens (`val_000.bin`: [0,8M) eval, [8M,10M) held-out metric batches). Took 100 s.
- 00:56 **Required tests all pass** on server (pytest): function preservation (every operator, atol 1e-5 fp32), inactive units don't affect output, new units get non-zero grads once mask > 0 (zero at mask 0), all params in optimizer + moment/step reset of grown slices, checkpoint round-trip exact (model, masks, births, optimizer, controller, next step bit-identical), FLOP counter = hand calc (207,912,960 FLOPs/token for S target), all 7 metrics return finite scores, 200-step S smoke run with M1-driven growth stable through all 8 events (loss 10.9 → val 6.46, 340k tok/s).
- 00:57 LR sweep launched (tmux `sweep`): S scratch, lr ∈ {1e-3, 2e-3, 4e-3, 8e-3}, 4 concurrent.
- 01:19 **LR sweep done** (S scratch, 250M tokens): lr 1e-3 → 4.331, 2e-3 → 4.216, **4e-3 → 4.176**, 8e-3 → 5.548 (unstable). Using **4e-3** for every arm. ~213k tok/s per job with 4 concurrent (GPU power-capped at 370 W).
- 01:19 Stage 2 queue launched (tmux `queue`, `scripts/queue.py`, 4 parallel, stops launching at 09:14 EDT): oracle base run + S arms b,c,d,e,f × 3 seeds (seed-major order). Scratch seed 0 = sweep run at 4e-3.
- 01:20 Oracle queue launched (tmux `oq`, 3 parallel): metrics + 2 oracle seeds at the 30% and 60% checkpoints, starting when checkpoints appear.
- 01:31 Oracle base run (S, 50% width, lr 4e-3) done; ckpts at 30% (step 763) and 60% (step 1526).
- 01:35 OOM incident: an ad-hoc check run I started on the GPU pushed memory over 93 GB and killed `metrics_0.3` and `oracle_0.3_seed0` (200-step). Re-queued.
- 01:55 **200-step oracle unreliable**: at 60% every candidate is within ±2e-4 of control (identical controls differ by 0.9e-4); seed-to-seed Spearman 0.50 (n=12). At 30% values are ±7e-4 with mixed signs (noise). → per prompt, re-running the oracle **once** with 600 continuation steps (val also logged at 200/400) for all 4 (ckpt, seed) pairs (tmux `oq3`), micro-batch 12 to fit memory.
- 02:05 Stage 2 seed 0 done for b, c, e, f (d = sweep run): see results/stage2_tables.md.
- 02:06 600-step oracle too slow on shared GPU (~15 min/candidate → ~3 h); killed, relaunched at **400 steps** (val at 200 and 400), tmux `oq4`. Stage 2 queue throttled to 2 slots via hold jobs (`touch runs/resume_stage2` on server to release).
- 02:34–02:41 SSH to server1 unreachable (network); retried with backoff (30s…240s), recovered 02:41. Server jobs unaffected (tmux).
- **~02:38 server container restarted** (uptime reset during the network outage) → every running job was killed: Stage 2 S_grow_random_s1 (was at step 2400/2543), S_grow_uniform_s1, S_scratch_s1, S_scratch_flops_s1, and the 4 oracle-400 processes (4/13 candidates each were saved; the oracle resumes from its JSON). Files on disk survived.
- 02:58 Added periodic resume checkpoints (every 200 steps, `runs/<name>/ckpt_latest.pt`) + auto-resume to train.py (test added, 14/14 tests pass). Relaunched tmux `oq4` (oracle-400) and `queue` (Stage 2, 2 hold slots).
- 03:30 **Stage 1 done** (400-step oracle). Oracle seed reliability: Spearman +0.20 @30%, +0.64 @60% (n=12). No metric positively correlated with the oracle (pooled ρ: M1b −0.19, M6 −0.23, M7 −0.28, M5 −0.31, M1 −0.35, M3 −0.64; M2 FFN-only −0.40). Selected M1b + M6 per protocol. Tables: results/oracle/stage1_table_oracle400*.md.
- 03:32 Released Stage 2 throttle; metric arms (M1b, M6) launched for seed 0. Exploratory −M3 arm (3 seeds) appended at the end of the queue.
- 03:45 M queued (300M tokens, 1 seed, 7 arms, lr 2.5e-3); M smoke test (40 steps, M1b growth) ok, ~16 GB/job.
- 04:10 Interim: arm (e) matched-FLOPs scratch is 0.08 worse than (d) in both seeds, from the first ~200 steps (grad norm 0.35 vs 0.25); only difference is warmup 93 vs 101 steps + 8% fewer steps → LR-edge sensitivity. Queued diagnostic (e') with warmup fixed at 101 steps, seeds 0–1, at the front of the queue.
- 04:20 Diagnostic (e') at step 702: val 4.936 vs (e) 5.183 vs (d) ≈4.95 → the (e) deficit is a warmup/LR-edge artifact; (e') is the valid matched-FLOPs baseline. Added (e') seed 2.
- 04:55 S progress: M1b 3 seeds, M6 3 seeds, e' 2 seeds done; seed 2 b/c/d/f and −M3 ×3 running/pending; M next.
- 05:45 All protocol S runs done (3 seeds; (e) 2 seeds) + exploratory −M3 ×3 + (e′) ×3. M running (scratch, uniform, M1b first; then M6, random, −M3, e). Seed 3 for M6/uniform/e′ queued behind M.
- 06:35 M results so far (1 seed): scratch 3.9767, uniform 3.9853, M1b 3.9991, M6 3.9986. Running: M random, M −M3, M (e), S seed 3 (M6, uniform, e′).
- 07:20 Queue empty: M arms (7) and S seed-3 runs done. Isolated throughput benchmark (idle GPU): S scratch 0.99M tok/s, S growth (2× prealloc) 0.79M, M scratch 0.51M, M growth 0.38M.
- 07:25 Final analysis (results/stage2_tables.md), plots, REPORT.md committed. GPU idle.
