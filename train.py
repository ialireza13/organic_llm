"""Training loop for all arms (nanoGPT style) with growth, FLOP accounting, checkpoints, CSV logs.

Example:
  python train.py --scale S --arm grow_metric --metric m1 --seed 0 --lr 2e-3 --run_name S_m1_s0
"""
import argparse
import csv
import json
import math
import os
import time

import numpy as np
import torch

from configs.scales import SCALES, growth_totals, model_config
from data import TrainData, ValData
from growth.controller import GrowthController
from growth.optimizer import GrowableAdamW
from metrics import METRICS
from model.gpt import GPT

DATA_DIR = "data/fineweb"


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="S")
    ap.add_argument("--arm", default="scratch")
    ap.add_argument("--metric", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--warmup_frac", type=float, default=0.04)
    ap.add_argument("--min_lr_frac", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--tokens", type=float, default=None, help="override token budget")
    ap.add_argument("--match_growth_flops", action="store_true", help="scratch_flops: set tokens to match growth FLOPs")
    ap.add_argument("--ramp_frac", type=float, default=0.02)
    ap.add_argument("--eval_every", type=int, default=None)
    ap.add_argument("--eval_tokens", type=int, default=2_000_000)
    ap.add_argument("--final_eval_tokens", type=int, default=8_000_000)
    ap.add_argument("--metric_batches", type=int, default=8)
    ap.add_argument("--metric_bs", type=int, default=8)
    ap.add_argument("--save_at", default="", help="comma-separated fractions of training to checkpoint at")
    ap.add_argument("--stop_frac", type=float, default=1.0)
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--no_compile", action="store_true")
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument("--results_dir", default="results/runs")
    ap.add_argument("--runs_dir", default="runs")
    return ap.parse_args(argv)


def lr_at(step, total, peak, warmup, min_frac):
    if step < warmup:
        return peak * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return peak * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))


def growth_arm_flops(scale, total_steps, tokens_per_step, n_events=8, start=0.1, end=0.6):
    """Total training FLOPs of any growth arm (depends only on total heads/FFN over time)."""
    s = SCALES[scale]
    L, d, hd, T, V = s["n_layer"], s["d_model"], 64, 1024, 50304
    add_h, add_c = growth_totals(scale)
    ev = [int(round(total_steps * (start + i * (end - start) / (n_events - 1)))) for i in range(n_events)]
    H0, F0 = L * (s["heads"] // 2), L * (s["ffn"] // 2)
    total = 0.0
    for step in range(total_steps):
        k = sum(1 for e in ev if e <= step)
        H = H0 + int(round(add_h * k / n_events))
        Fn = F0 + 64 * int(round(add_c * k / n_events))
        fpt = 6 * (4 * d * hd * H + 2 * d * Fn + d * V) + 12 * hd * H * T
        total += fpt * tokens_per_step
    return total


class Trainer:
    def __init__(self, args):
        self.args = args
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        sc = SCALES[args.scale]
        self.B, self.micro = sc["batch"], sc["micro"]
        self.tokens_per_step = self.B * 1024
        tokens = args.tokens or sc["tokens"]
        full_steps = int(tokens // self.tokens_per_step)
        if args.match_growth_flops:
            gf = growth_arm_flops(args.scale, int(sc["tokens"] // self.tokens_per_step), self.tokens_per_step)
            cfg_t = model_config(args.scale, "scratch", 1)
            d, hd, T, V, L = cfg_t.d_model, 64, 1024, cfg_t.vocab_size, cfg_t.n_layer
            fpt = 6 * (sum(4 * d * hd * h + 2 * d * f for h, f in zip(cfg_t.init_heads, cfg_t.init_ffn)) + d * V) \
                + sum(12 * hd * h * T for h in cfg_t.init_heads)
            full_steps = int(gf / fpt // self.tokens_per_step)
            self.matched_flops_target = gf
        self.total_steps = full_steps
        self.ramp = max(1, int(round(args.ramp_frac * full_steps)))
        self.cfg = model_config(args.scale, args.arm, self.ramp)
        self.model = GPT(self.cfg).cuda()
        self.opt = GrowableAdamW(self.model, lr=args.lr, weight_decay=args.wd)
        self.cmodel = self.model if args.no_compile else torch.compile(self.model)
        self.data = TrainData(args.data_dir, 1024, self.B, data_seed=args.seed)
        self.val = ValData(args.data_dir, 1024)
        self.warmup = max(1, int(args.warmup_frac * full_steps))
        policy = {"grow_uniform": "uniform", "grow_random": "random", "grow_metric": "metric"}.get(args.arm, "none")
        add_h, add_c = growth_totals(args.scale)
        mfn = METRICS[args.metric] if (policy == "metric") else None
        self.ctl = GrowthController(
            self.model, self.opt, policy, full_steps, add_h, add_c, metric_fn=mfn, metric_name=args.metric,
            metric_batches_fn=lambda k: self.val.metric_batches(args.metric_batches, args.metric_bs, offset=k * args.metric_batches),
            seed=args.seed)
        self.step = 0
        self.flops = 0.0
        self.tokens = 0
        self.train_time = 0.0
        name = args.run_name or f"{args.scale}_{args.arm}{'_' + args.metric if args.metric else ''}_s{args.seed}"
        self.name = name
        self.res_dir = os.path.join(args.results_dir, name)
        self.ckpt_dir = os.path.join(args.runs_dir, name)
        os.makedirs(self.res_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.save_steps = {int(round(float(f) * full_steps)): f for f in args.save_at.split(",") if f}
        self.eval_every = args.eval_every or max(1, full_steps // 20)

    # ------------------------------------------------------------------
    def train_step(self, step):
        n_micro = self.B // self.micro
        loss_acc = 0.0
        for mi in range(n_micro):
            x, y = self.data.get(step, mi, n_micro)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = self.cmodel(x, y)
            (loss / n_micro).backward()
            loss_acc += loss.detach()
        gn = torch.nn.utils.clip_grad_norm_(self.opt.all_params(), 1.0)
        lr = lr_at(step, self.total_steps, self.args.lr, self.warmup, self.args.min_lr_frac)
        self.opt.step(lr)
        self.opt.zero_grad()
        return loss_acc / n_micro, gn, lr

    @torch.no_grad()
    def evaluate(self, n_tokens):
        tot, n = 0.0, 0
        for x, y in self.val.eval_batches(n_tokens, bs=32):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = self.cmodel(x, y)
            tot += loss.float() * x.numel()
            n += x.numel()
        return float(tot / n)

    def state(self):
        return {"model": self.model.state_dict(), "opt": self.opt.state_dict(), "ctl": self.ctl.state_dict(),
                "active_heads": self.model.active_heads, "active_ffn": self.model.active_ffn, "step": self.step,
                "flops": self.flops, "tokens": self.tokens, "cfg": self.cfg.to_dict(), "args": vars(self.args),
                "total_steps": self.total_steps}

    def save(self, path):
        torch.save(self.state(), path)

    def load(self, path):
        sd = torch.load(path, map_location="cuda", weights_only=False)
        self.load_state(sd)

    def load_state(self, sd):
        self.model.load_state_dict(sd["model"])
        self.opt.load_state_dict(sd["opt"])
        self.ctl.load_state_dict(sd["ctl"])
        self.model.active_heads = list(sd["active_heads"])
        self.model.active_ffn = list(sd["active_ffn"])
        self.step, self.flops, self.tokens = sd["step"], sd["flops"], sd["tokens"]

    # ------------------------------------------------------------------
    def run(self):
        a = self.args
        json.dump({"args": vars(a), "cfg": self.cfg.to_dict(), "total_steps": self.total_steps,
                   "tokens_per_step": self.tokens_per_step, "event_steps": self.ctl.event_steps,
                   "warmup": self.warmup, "ramp_steps": self.ramp},
                  open(os.path.join(self.res_dir, "config.json"), "w"), indent=1)
        ftrain = open(os.path.join(self.res_dir, "train_log.csv"), "w", newline="")
        wtrain = csv.writer(ftrain)
        wtrain.writerow(["step", "loss", "grad_norm", "lr", "tokens", "flops", "tok_per_s", "elapsed_s"])
        fval = open(os.path.join(self.res_dir, "val_log.csv"), "w", newline="")
        wval = csv.writer(fval)
        wval.writerow(["step", "tokens", "flops", "val_loss", "active_nonemb_params"])
        fev = open(os.path.join(self.res_dir, "growth_events.csv"), "w", newline="")
        wev = csv.writer(fev)
        wev.writerow(["step", "event", "kind", "layer", "score", "params", "flops_per_token_after"])
        stop_step = int(round(a.stop_frac * self.total_steps))
        t_start = time.time()
        t_last, tok_last = time.time(), 0
        status = "ok"
        while self.step < stop_step:
            s = self.step
            if s in self.save_steps:
                self.save(os.path.join(self.ckpt_dir, f"ckpt_{self.save_steps[s]}.pt"))
            self.ctl.maybe_metric(s)
            for ev in self.ctl.maybe_grow(s):
                wev.writerow([ev["step"], ev["event"], ev["kind"], ev["layer"], ev["score"], ev["params"],
                              self.model.flops_per_token()])
                fev.flush()
            self.model.update_masks(s)
            if s % self.eval_every == 0:
                vl = self.evaluate(a.eval_tokens)
                wval.writerow([s, self.tokens, self.flops, vl, self.model.active_nonembedding_params()])
                fval.flush()
            loss, gn, lr = self.train_step(s)
            self.flops += self.model.flops_per_token() * self.tokens_per_step
            self.tokens += self.tokens_per_step
            self.step += 1
            if s % 10 == 0 or self.step == stop_step:
                lv = float(loss)
                if not math.isfinite(lv):
                    status = f"nonfinite loss at step {s}"
                    print(status, flush=True)
                    break
                torch.cuda.synchronize()
                now = time.time()
                tps = (self.tokens - tok_last) / max(1e-9, now - t_last)
                t_last, tok_last = now, self.tokens
                wtrain.writerow([s, lv, float(gn), lr, self.tokens, self.flops, tps, now - t_start])
                ftrain.flush()
                if s % 100 == 0:
                    print(f"step {s}/{self.total_steps} loss {lv:.4f} lr {lr:.2e} tok/s {tps:.0f} "
                          f"alloc {self.model.allocation()}", flush=True)
        if self.step in self.save_steps:
            self.save(os.path.join(self.ckpt_dir, f"ckpt_{self.save_steps[self.step]}.pt"))
        wall = time.time() - t_start
        final = self.evaluate(a.final_eval_tokens) if status == "ok" else float("nan")
        wval.writerow([self.step, self.tokens, self.flops, final, self.model.active_nonembedding_params()])
        for f in (ftrain, fval, fev):
            f.close()
        json.dump(self.ctl.metric_log, open(os.path.join(self.res_dir, "metric_log.json"), "w"))
        summary = {
            "name": self.name, "scale": a.scale, "arm": a.arm, "metric": a.metric, "seed": a.seed, "lr": a.lr,
            "status": status, "steps": self.step, "total_steps": self.total_steps, "tokens": self.tokens,
            "final_val_loss": final, "train_flops": self.flops,
            "metric_seconds": self.ctl.metric_seconds, "metric_tokens": self.ctl.metric_tokens,
            "wall_clock_s": wall, "tokens_per_s": self.tokens / wall,
            "allocation": self.model.allocation(),
            "active_nonemb_params": self.model.active_nonembedding_params(),
        }
        json.dump(summary, open(os.path.join(self.res_dir, "summary.json"), "w"), indent=1)
        print("SUMMARY", json.dumps(summary), flush=True)
        return summary


if __name__ == "__main__":
    Trainer(get_args()).run()
