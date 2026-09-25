"""Write scripts/jobs.tsv (queue input). Usage: python scripts/make_jobs.py LR [metric1 metric2]"""
import sys

lr = sys.argv[1]
metrics = sys.argv[2:4]
C = "source .venv/bin/activate && OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py --micro 24 --lr " + lr
jobs = []
# hold jobs occupy 2 of the 4 slots until runs/resume_stage2 exists (prioritizes the Stage-1 oracle)
for h in ("hold1", "hold2"):
    jobs.append((h, "while [ ! -f runs/resume_stage2 ]; do sleep 30; done"))
# Stage 1 oracle base run: 50% width, room for one extra unit per layer; checkpoints at 30% and 60%
jobs.append(("S_oraclebase", f"{C} --scale S --arm base_half --seed 0 --save_at 0.3,0.6 --stop_frac 0.6 --run_name S_oraclebase --results_dir results/oracle_base"))
# diagnostic: arm (e) with the same absolute warmup (101 steps) as the 2543-step runs (0.0431 * 2346 = 101)
for seed in (0, 1):
    jobs.append((f"S_scratch_flops_w101_s{seed}", f"{C} --scale S --arm scratch_flops --seed {seed} --match_growth_flops --warmup_frac 0.0431 --run_name S_scratch_flops_w101_s{seed}"))
for seed in (0, 1, 2):
    for m in metrics:
        jobs.append((f"S_grow_metric_{m}_s{seed}", f"{C} --scale S --arm grow_metric --metric {m} --seed {seed}"))
    for arm in ("grow_random", "grow_uniform", "scratch", "scratch_flops", "openelm"):
        if arm == "scratch" and seed == 0:
            continue  # = LR-sweep run at the chosen LR (copied)
        extra = " --match_growth_flops" if arm == "scratch_flops" else ""
        jobs.append((f"S_{arm}_s{seed}", f"{C} --scale S --arm {arm} --seed {seed}{extra}"))
# exploratory arm (a3), queued after all protocol S runs
for m in sys.argv[4:]:
    for seed in (0, 1, 2):
        jobs.append((f"S_grow_metric_{m}_s{seed}", f"{C} --scale S --arm grow_metric --metric {m} --seed {seed}"))
# M scale (1 seed, reduced tokens, LR scaled by 1/width from the S optimum), queued after all S runs
CM = C.replace("--micro 24 --lr " + lr, "--micro 16 --lr 2.5e-3")
m_arms = [("scratch", ""), ("grow_uniform", ""), ("grow_metric", metrics[0]), ("grow_metric", metrics[1]),
          ("grow_random", ""), ("grow_metric", "m3neg"), ("scratch_flops", "")]
for arm, m in m_arms:
    extra = (f" --metric {m}" if m else "") + (" --match_growth_flops" if arm == "scratch_flops" else "")
    name = f"M_{arm}{'_' + m if m else ''}_s0"
    jobs.append((name, f"{CM} --scale M --arm {arm} --seed 0{extra}"))
with open("scripts/jobs.tsv", "w") as f:
    for n, c in jobs:
        f.write(f"{n}\t{c}\n")
print(len(jobs), "jobs")
