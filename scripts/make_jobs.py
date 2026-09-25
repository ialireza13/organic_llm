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
for seed in (0, 1, 2):
    for m in metrics:
        jobs.append((f"S_grow_metric_{m}_s{seed}", f"{C} --scale S --arm grow_metric --metric {m} --seed {seed}"))
    for arm in ("grow_random", "grow_uniform", "scratch", "scratch_flops", "openelm"):
        if arm == "scratch" and seed == 0:
            continue  # = LR-sweep run at the chosen LR (copied)
        extra = " --match_growth_flops" if arm == "scratch_flops" else ""
        jobs.append((f"S_{arm}_s{seed}", f"{C} --scale S --arm {arm} --seed {seed}{extra}"))
with open("scripts/jobs.tsv", "w") as f:
    for n, c in jobs:
        f.write(f"{n}\t{c}\n")
print(len(jobs), "jobs")
