"""Jobs for the full-budget M test (800M tokens, 3 seeds): uniform growth vs from-scratch at matched FLOPs
(and from scratch at the same tokens as a reference). All arms: batch 128 (6103 steps), lr 2.5e-3, the same
absolute warmup (244 steps = 4% of 6103). Uniform growth pre-allocates exactly the target width (1x).
Writes scripts/jobs_m800.tsv; results go to results/m800/."""
C = ("source .venv/bin/activate && OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
     "python train.py --scale M --micro 16 --lr 2.5e-3 --tokens 8e8 --warmup_steps 244 --results_dir results/m800")
jobs = []
for seed in (0, 1, 2):
    for arm, extra in (("grow_uniform", "--prealloc 1.0"), ("scratch_flops", "--match_growth_flops"), ("scratch", "")):
        name = f"M800_{arm}_s{seed}"
        jobs.append((name, f"{C} --arm {arm} --seed {seed} {extra} --run_name {name}"))
with open("scripts/jobs_m800.tsv", "w") as f:
    for n, c in jobs:
        f.write(f"{n}\t{c}\n")
print(len(jobs), "jobs")
