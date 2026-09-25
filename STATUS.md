# STATUS

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
