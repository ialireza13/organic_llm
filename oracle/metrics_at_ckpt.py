"""Compute every metric at a checkpoint (before growth) on held-out batches; record scores and cost.

Each metric is evaluated `--repeats` times on disjoint held-out batch sets (8 x 8 x 1024 tokens each,
the same size as one Stage-2 evaluation); the reported score is the mean over repeats (a stand-in
for the EMA used in Stage 2), and the split-repeat Spearman gives the metric's own reliability.
"""
import argparse
import json
import time

import torch

from metrics import METRICS
from train import Trainer, get_args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--n_batches", type=int, default=8)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--metrics", default=",".join(METRICS))
    a = ap.parse_args()
    sd = torch.load(a.ckpt, map_location="cuda", weights_only=False)
    targs = sd["args"]
    argv = ["--scale", targs["scale"], "--arm", targs["arm"], "--seed", str(targs["seed"]), "--lr", str(targs["lr"]),
            "--run_name", "metric_tmp", "--results_dir", "runs/metric_tmp", "--runs_dir", "runs/metric_tmp",
            "--no_compile"]
    if targs.get("tokens"):
        argv += ["--tokens", str(targs["tokens"])]
    t = Trainer(get_args(argv))
    out = {"ckpt": a.ckpt, "step": sd["step"], "metrics": {}}
    for name in a.metrics.split(","):
        fn = METRICS[name]
        reps, secs = [], []
        for r in range(a.repeats):
            t.load_state(sd)
            t.model.update_masks(sd["step"])
            batches = t.val.metric_batches(a.n_batches, a.bs, offset=r * a.n_batches)
            torch.cuda.synchronize()
            t0 = time.time()
            sc = fn(t.model, batches, opt=t.opt)
            torch.cuda.synchronize()
            secs.append(time.time() - t0)
            reps.append(sc)
        keys = reps[0].keys()
        mean = {k: sum(rr[k] for rr in reps) / len(reps) for k in keys}
        out["metrics"][name] = {"score": mean, "repeats": reps, "seconds": secs}
        print(f"{name}: {secs} s; {mean}", flush=True)
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
