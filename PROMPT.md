# Overnight task: Organic Growth of Transformers — build + Stage 0–2 ablation

You are working **fully autonomously overnight**. The researcher (Alireza) is asleep and will read your results in the morning. `PLAN.md` in this directory is the project brief — read it fully first. This prompt narrows it to tonight's scope and overrides it where they differ.

## 0. Autonomy rules (most important)

- **Never ask questions and never wait for input.** Nobody will answer until morning. When something is ambiguous, pick the most reasonable option, record it in `DECISIONS.md` (one line: what, why, alternative), and keep going.
- **Never stop early.** If a component fails and you cannot fix it within ~30 minutes, drop or simplify it, log it in `DECISIONS.md`, and continue with the rest. A partial but honest result beats no result.
- **Keep `STATUS.md` current** (timestamped, updated at least every phase and every ~60 min): current phase, what is running, what finished, what failed. After any context compaction, re-read `STATUS.md`, `DECISIONS.md` and `PLAN.md` before continuing.
- **Long jobs never run in the foreground.** Launch training on the server in detached `tmux` sessions with logs to files, then poll with `sleep 300`–`sleep 540` plus `ssh server1 'tail ...; nvidia-smi'`. Your own bash calls have time limits; jobs must survive without you, and must survive a dropped SSH connection.
- **Commit to git** after every working milestone (local repo only; do not push anywhere).
- **Safety:** locally, work only inside this project directory. On the server, work only inside `~/organic-growth` (plus `~/.cache` for downloads). No `sudo`, no deleting anything outside the project on either machine, no opening ports, no sending data anywhere except between this laptop and server1. Use a Python venv on the server. Keep server disk usage under 80%.

## 1. Environment and workflow (laptop + server)

You are running on the researcher's **laptop**. The **code lives here**, in this directory (the git repo). All GPU work runs on `server1` (single H100 NVL 94 GB, 28 CPU cores, 185 GB RAM), reached with `ssh server1`.

- **Edit code only locally.** Never edit code files on the server.
- **Sync code to the server before every test or launch:**
  `rsync -az --delete --exclude .git --exclude .venv --exclude data/ --exclude runs/ --exclude results/ ./ server1:~/organic-growth/`
- **Server-only, never synced back:** the venv, `data/` (tokenized shards), `runs/` (checkpoints, full logs).
- **Results come back to the laptop:** each run writes small artifacts (metrics CSVs, growth-event logs, final summaries, plots) to `~/organic-growth/results/` on the server. Pull them with `rsync -az server1:~/organic-growth/results/ ./results/` after each phase and before writing the report, then commit them.
- **Remote commands** are non-interactive: `ssh server1 'cd ~/organic-growth && source .venv/bin/activate && ...'`.
- **Long jobs:** `ssh server1 "cd ~/organic-growth && tmux new -d -s <name> 'source .venv/bin/activate && <cmd> > runs/<name>.log 2>&1'"`. Keep a launch queue script on the server (e.g. `scripts/queue.sh`) so a batch of runs proceeds even if the laptop loses connection.
- **If SSH fails,** retry with backoff (30s, 1m, 2m, ... up to 30 minutes total), log it in `STATUS.md`, and resume. Jobs already running on the server continue; check their state before relaunching anything, so no run is duplicated.
- `STATUS.md`, `DECISIONS.md` and `REPORT.md` live locally in the repo.

Record the start time in `STATUS.md`. **Hard deadline: start + 10 hours.** Time boxes:

| By | Must be done |
| --- | --- |
| start + 3h | Env, data tokenized, model + growth operators, all required tests passing |
| start + 5h | Cheap metrics implemented, oracle (Stage 1) finished |
| start + 9h | Stage 2 ablation runs finished (stop launching new runs at start + 8.5h) |
| start + 10h | `REPORT.md` written and committed; no stray GPU processes |

If a phase overruns, cut scope (fewer seeds, drop the M-scale, drop expensive metrics) rather than skipping the report.

## 2. Setup

1. Check SSH works (`ssh server1 hostname`). On the server: `nvidia-smi` (confirm full GPU, ~94 GB, no MIG), `df -h`, `nproc`, `free -g`, Python/CUDA versions. Log in `STATUS.md`.
2. Locally: git init this directory (if not already a repo), add a `.gitignore` for `data/`, `runs/`, `.venv/`, checkpoints. On the server: create `~/organic-growth` and a venv; install recent `torch`, `numpy`, `tiktoken`, `datasets`, `scipy`, `pandas`, `matplotlib`, `pytest`. Save the exact package versions to `requirements.txt` locally.
3. **Data (on the server):** FineWeb `sample-10BT` (HF `HuggingFaceFW/fineweb`), GPT-2 tokenizer via tiktoken, multiprocessing across the CPU cores, written as uint16 `.bin` shards (nanoGPT style). Tokenize ~1.5B train tokens + a fixed ~10M-token validation split. Fallbacks in order: FineWeb-Edu sample, OpenWebText. Log what you used.

## 3. Scope decisions already made (do not revisit)

- **Width growth only:** FFN neurons and attention heads. Layer count and `d_model` are fixed. No depth growth, no `d_model` growth, no pruning.
- **Fixed-interval growth schedule;** the metric decides only *where* to grow.
- **Static-shape implementation:** pre-allocate every growable unit at its maximum size and gate units with per-unit masks (FFN: mask on each hidden neuron's activation; attention: mask on each head's output before O). Growing a unit = re-initialize its weights with the standard init, reset its AdamW moments to zero, and ramp its mask 0 → 1 linearly (default ramp: 2% of total steps). Inactive units have mask 0 and must not affect the output. This keeps `torch.compile` shapes static.
- **FLOP accounting:** count FLOPs of *active* parameters only (6·N_active·tokens plus the attention term), updated at every growth event. Log metric-computation FLOPs/time separately.
- GPT-style pre-LN transformer, GELU FFN, seq len 1024, AdamW, cosine LR with warmup, bf16. Untied or tied embeddings — your choice, log it.

## 4. Model sizes

| Name | Layers | d_model | Target heads (hd 64) | Target FFN | Non-embedding params at target |
| --- | --- | --- | --- | --- | --- |
| S | 6 | 384 | 6 | 1536 | ~10.6M |
| M | 8 | 640 | 10 | 2560 | ~39M |

- **Growth arms start at 50% of target** in every layer (S: 3 heads, 768 FFN; M: 5 heads, 1280 FFN) and end at exactly the target *total* heads and FFN neurons.
- **Pre-allocate 2× the per-layer target** (heads and FFN) so the metric can allocate non-uniformly.
- **Growth units:** 64 FFN neurons, or 1 head.
- **Schedule:** 8 growth events evenly spaced between 10% and 60% of training; each event adds 1/8 of the total units to be added, split between FFN chunks and heads by the metric's ranking (score per added parameter).
- **Token budgets (defaults):** S = 250M tokens, M = 800M tokens. Pick batch sizes that give a reasonable step count (~2k–3k steps). Tune LR once for the from-scratch uniform baseline at S (a quick 3–4 value sweep) and reuse it for every arm.

## 5. Required tests (must pass before any experiment)

Write them in `tests/` locally, sync, run them on the server with pytest (GPU tests need the server), and put the results in `STATUS.md`:

1. Function preservation: logits identical (atol ~1e-5, fp32) before and after every growth operation.
2. New units receive non-zero gradients once their mask ramps above 0.
3. Every active parameter is in the optimizer; moments for newly grown units are reset.
4. Checkpoint save/load round-trips exactly, including masks and optimizer state.
5. FLOP counter matches a hand calculation for one config.
6. A 200-step smoke run on S trains stably through at least one growth event (no loss spike after the ramp).

## 6. Metrics (in this order; skip what does not fit in time)

All share the interface `score(model, batches) -> {unit_id: score}`, normalized by added parameters, computed on held-out batches (not training batches), with an EMA over the last few evaluations.

1. **M5** gradient SNR from AdamW state (near-free).
2. **M6** spectral saturation: stable rank per block, dormant-neuron fraction.
3. **M7** block influence 1 − cos(x_in, x_out) (layer-level; use as filter/tie-breaker).
4. **M1** GradMax-style score: for a candidate new unit, the top singular value(s) of E[g_out · x_inᵀ] at that layer (see the GradMax paper). Also implement the cheap variant **M1b**: gradient of the loss w.r.t. the mask of a freshly initialized, currently inactive candidate unit.
5. **M3** TINY-style expressivity bottleneck.
6. **M2** splitting eigenvalue (FFN only; power iteration with Hessian-vector products on a subsample).
7. **M4** — skip tonight.

## 7. Stage 1 — oracle validation (at S)

1. Train one S growth-start model (50% width) to 30% and to 60% of the S budget; save both checkpoints.
2. At each checkpoint, candidates are: +64 FFN neurons in layer ℓ, or +1 head in layer ℓ, for every layer (12 candidates per checkpoint).
3. For each candidate *and* a no-growth control: continue training from the checkpoint with the **same data order**, grow that single unit at step 0 (short ramp, ~30 steps), train ~200 steps, then evaluate on a fixed validation set of ≥2M tokens. Oracle value = control loss − candidate loss, per added parameter. Run 2 seeds (different new-unit init).
4. **Check that the oracle is reliable first:** report the correlation between the two oracle seeds. If it is near zero, increase the continuation steps and/or eval size once, and log that you did.
5. Compute every implemented metric at the checkpoint (before growth) and report the Spearman correlation between each metric's ranking and the mean oracle ranking, per checkpoint and pooled. Include a random-ranking baseline.
6. **Select the top 2 metrics** by pooled Spearman for Stage 2. Log the choice and reasoning.

## 8. Stage 2 — ablation

Arms, all with the same token budget and the same growth schedule for growth arms:

- (a1), (a2): metric-driven growth with the two selected metrics
- (b) growth at random locations (same schedule and unit counts)
- (c) uniform growth (every layer grows equally)
- (d) from-scratch uniform model at the target size, same tokens
- (e) from-scratch uniform model at the target size, with tokens reduced so its total FLOPs match the growth arms
- (f) if time allows: OpenELM-style linearly varying per-layer widths at the same total size, from scratch

Seeds: 3 at S, then 2 at M if time allows (S first, fully, before any M run). Run several S jobs concurrently on the GPU if memory and throughput allow; check `nvidia-smi` utilization.

Log per run: validation loss curve, final validation loss, cumulative active FLOPs, tokens/s, wall-clock, every growth event (step, unit, score), and final per-layer allocation (heads and FFN width per layer).

## 9. Morning deliverable: `REPORT.md`

Pull all results from the server first, then write the report locally, lead with conclusions, and commit it. It must contain:

1. **TL;DR** (≤5 lines): which metrics look worth keeping, which to drop, and whether metric-driven growth beat random and uniform growth beyond seed noise.
2. What was built, and test results.
3. Stage 1 table: metric × Spearman (per checkpoint, pooled), oracle seed-to-seed reliability, metric cost (seconds per evaluation).
4. Stage 2 table: arm × scale → final val loss (mean ± std over seeds), total FLOPs, tokens/s.
5. Plots in `results/plots/`: val loss vs tokens and vs FLOPs per arm; per-layer allocation heatmaps for metric-driven arms. Link them from the report.
6. Everything that failed, was cut, or deviated from this prompt (link `DECISIONS.md`).
7. Recommended next steps for Stage 3, including which metrics to carry forward.

Be honest: if differences are within noise, say so plainly. Do not overstate results.

If time remains after the report is committed, run extra seeds for the closest comparisons and append the results to `REPORT.md` under a clearly marked "Added after report" section. Make sure no GPU jobs are left running at the deadline, except ones you deliberately leave and describe in `STATUS.md`.
