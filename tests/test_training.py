import json
import math
import os

import numpy as np
import pytest
import torch

from metrics import METRICS
from train import Trainer, get_args

HAS_DATA = os.path.exists("data/fineweb/val_000.bin")
pytestmark = pytest.mark.skipif(not HAS_DATA, reason="needs tokenized data (server)")


def make_trainer(tmp_path, name, extra=()):
    args = get_args(["--scale", "T", "--arm", "grow_metric", "--metric", "m1", "--no_compile", "--tokens", "163840",
                     "--results_dir", str(tmp_path / "res"), "--runs_dir", str(tmp_path / "runs"),
                     "--run_name", name, "--metric_batches", "2", "--metric_bs", "2", *extra])
    return Trainer(args)


def test_checkpoint_roundtrip_exact(tmp_path):
    torch.backends.cuda.matmul.allow_tf32 = False
    t1 = make_trainer(tmp_path, "a")
    # run through the first growth events so masks, births and optimizer state are non-trivial
    stop = t1.ctl.event_steps[2] + 2
    for s in range(stop):
        t1.ctl.maybe_metric(s)
        t1.ctl.maybe_grow(s)
        t1.model.update_masks(s)
        t1.train_step(s)
        t1.step += 1
    assert len(t1.ctl.events) > 0
    path = tmp_path / "ck.pt"
    t1.save(path)
    t2 = make_trainer(tmp_path, "b")
    t2.load(path)
    sd1, sd2 = t1.model.state_dict(), t2.model.state_dict()
    for k in sd1:
        assert torch.equal(sd1[k], sd2[k]), k
    assert t1.model.active_heads == t2.model.active_heads and t1.model.active_ffn == t2.model.active_ffn
    for (n1, s1), (n2, s2) in zip(t1.opt.state_dict()["state"], t2.opt.state_dict()["state"]):
        assert n1 == n2
        for k in s1:
            assert torch.equal(s1[k], s2[k]), (n1, k)
    assert t1.ctl.events == t2.ctl.events and t1.ctl.ema == t2.ctl.ema
    # one more identical step on both -> identical parameters
    for t in (t1, t2):
        t.model.update_masks(stop)
        t.train_step(stop)
    for (n, p1), (_, p2) in zip(t1.model.named_parameters(), t2.model.named_parameters()):
        assert torch.equal(p1, p2), n


@pytest.mark.parametrize("name", list(METRICS))
def test_metric_returns_finite_scores(tmp_path, name):
    t = make_trainer(tmp_path, f"m_{name}")
    for s in range(3):
        t.model.update_masks(s)
        t.train_step(s)
    batches = t.val.metric_batches(2, 2)
    sc = METRICS[name](t.model, batches, opt=t.opt)
    L = t.cfg.n_layer
    need = [f"ffn:{l}" for l in range(L)] + ([] if name == "m2" else [f"head:{l}" for l in range(L)])
    for u in need:
        assert u in sc and math.isfinite(sc[u]), (u, sc)
    # metric computation must not change the function or leave gradients behind
    assert all(p.grad is None for p in t.model.parameters())


def test_resume_after_crash(tmp_path):
    torch.backends.cuda.matmul.allow_tf32 = False
    extra = ("--ckpt_every", "5", "--final_eval_tokens", "65536", "--eval_tokens", "65536", "--eval_every", "4")
    t = make_trainer(tmp_path, "r", extra)
    orig = t.train_step

    def boom(step):
        if step == 12:
            raise RuntimeError("simulated crash")
        return orig(step)

    t.train_step = boom
    with pytest.raises(RuntimeError):
        t.run()
    assert os.path.exists(tmp_path / "runs" / "r" / "ckpt_latest.pt")
    t2 = make_trainer(tmp_path, "r", extra)
    summ = t2.run()
    assert summ["status"] == "ok" and summ["steps"] == t2.total_steps
    import csv as _csv
    steps = [int(r["step"]) for r in _csv.DictReader(open(tmp_path / "res" / "r" / "train_log.csv"))]
    assert steps == sorted(set(steps)), steps  # no duplicated rows after resume
    ev = list(_csv.DictReader(open(tmp_path / "res" / "r" / "growth_events.csv")))
    assert len(ev) == len(t2.ctl.events)
    assert not os.path.exists(tmp_path / "runs" / "r" / "ckpt_latest.pt")
