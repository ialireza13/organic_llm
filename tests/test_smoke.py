"""200-step smoke run at scale S with metric-driven growth (compiled); checks stability through growth."""
import csv
import os

import numpy as np
import pytest

from train import Trainer, get_args

HAS_DATA = os.path.exists("data/fineweb/val_000.bin")


@pytest.mark.skipif(not HAS_DATA, reason="needs tokenized data (server)")
def test_smoke_S_200_steps(tmp_path):
    args = get_args(["--scale", "S", "--arm", "grow_metric", "--metric", "m1", "--tokens", str(200 * 96 * 1024),
                     "--lr", "2e-3", "--ramp_frac", "0.02", "--results_dir", str(tmp_path / "res"),
                     "--runs_dir", str(tmp_path / "runs"), "--run_name", "smoke", "--eval_every", "50",
                     "--final_eval_tokens", "1000000", "--eval_tokens", "500000"])
    t = Trainer(args)
    summ = t.run()
    assert summ["status"] == "ok" and summ["steps"] == 200
    assert len(t.ctl.events) > 0
    rows = list(csv.DictReader(open(tmp_path / "res" / "smoke" / "train_log.csv")))
    step = np.array([int(r["step"]) for r in rows])
    loss = np.array([float(r["loss"]) for r in rows])
    # no spike: after every growth event + ramp, logged loss stays within 0.15 of the pre-event loss
    for ev in sorted({e["step"] for e in t.ctl.events}):
        pre = loss[step < ev][-1]
        post = loss[(step > ev) & (step <= ev + t.ramp + 20)]
        assert post.size and post.max() < pre + 0.15, (ev, pre, post)
    assert summ["final_val_loss"] < 7.5
    a = summ["allocation"]
    assert sum(a["heads"]) == 36 and sum(a["ffn"]) == 9216, a
    print("SMOKE", summ)
