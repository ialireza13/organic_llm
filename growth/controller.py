"""Growth controller: fixed-interval schedule; the policy decides only *where* to grow.

Schedule: `n_events` events evenly spaced between start_frac and end_frac of training. Totals to add
(heads, FFN chunks) are fixed so every growth arm ends at exactly the target totals.

Policies
  uniform: event e adds round(H*e/8)-round(H*(e-1)/8) heads and C*e/8-... chunks, assigned round-robin
           over layers (each layer ends with the same width).
  random:  same per-event counts per type as uniform; layers drawn uniformly at random among those
           with free slots.
  metric:  per-event parameter budget = 1/8 of total added parameters. Units are picked greedily by
           metric score per parameter with D'Hondt diminishing returns (score / (1 + units already
           added to that (type, layer) at this event)), within the remaining per-type totals; the
           last event adds everything that remains. Scores are EMA-smoothed over metric evaluations
           run every `metric_every` steps on held-out batches.
"""
import json
import time

import numpy as np
import torch

from growth.operators import can_grow, grow_unit, unit_params
from metrics.common import parse_uid, uid


class GrowthController:
    def __init__(self, model, opt, policy, total_steps, add_heads, add_chunks, n_events=8,
                 start_frac=0.1, end_frac=0.6, metric_fn=None, metric_name=None, metric_batches_fn=None,
                 metric_every=None, ema=0.5, seed=0, log_dir=None):
        self.model, self.opt, self.policy = model, opt, policy
        self.total_steps = total_steps
        self.add_heads, self.add_chunks = add_heads, add_chunks
        self.n_events = n_events
        self.event_steps = [int(round(total_steps * (start_frac + i * (end_frac - start_frac) / (n_events - 1))))
                            for i in range(n_events)]
        self.metric_fn, self.metric_name = metric_fn, metric_name
        self.metric_batches_fn = metric_batches_fn
        spacing = self.event_steps[1] - self.event_steps[0] if n_events > 1 else total_steps
        self.metric_every = metric_every or max(1, spacing // 3)
        self.first_metric_step = self.event_steps[0] - 2 * self.metric_every
        self.ema_beta = ema
        self.ema = None
        self.n_metric_evals = 0
        self.metric_seconds = 0.0
        self.metric_tokens = 0
        self.rng = np.random.default_rng(seed + 777)
        self.gen = torch.Generator(device=next(model.parameters()).device)
        self.gen.manual_seed(seed + 999)
        self.events = []  # dicts: step, event, kind, layer, score, params
        self.added = {"head": 0, "ffn": 0}
        self.rr = {"head": 0, "ffn": 0}  # round-robin pointers
        self.log_dir = log_dir
        self.metric_log = []

    # ------------------------------------------------------------------ metric
    def maybe_metric(self, step):
        if self.policy != "metric" or step < self.first_metric_step or step > self.event_steps[-1]:
            return
        if (step - self.first_metric_step) % self.metric_every != 0 and step not in self.event_steps:
            return
        if self.metric_log and self.metric_log[-1]["step"] == step:
            return
        batches = self.metric_batches_fn(self.n_metric_evals)
        torch.cuda.synchronize()
        t0 = time.time()
        sc = self.metric_fn(self.model, batches, opt=self.opt)
        torch.cuda.synchronize()
        dt = time.time() - t0
        self.metric_seconds += dt
        self.metric_tokens += sum(x.numel() for x, _ in batches)
        self.n_metric_evals += 1
        if self.ema is None:
            self.ema = dict(sc)
        else:
            for k, v in sc.items():
                self.ema[k] = self.ema_beta * self.ema.get(k, v) + (1 - self.ema_beta) * v
        self.metric_log.append({"step": step, "seconds": dt, "raw": sc, "ema": dict(self.ema)})

    # ------------------------------------------------------------------ growth
    def event_index(self, step):
        return self.event_steps.index(step) if step in self.event_steps else None

    def _quota(self, e, total):
        """Units of one type to add at event e (0-based) under an even split."""
        return int(round(total * (e + 1) / self.n_events)) - int(round(total * e / self.n_events))

    def maybe_grow(self, step):
        e = self.event_index(step)
        if e is None or self.policy in ("none", None):
            return []
        if self.policy == "uniform":
            plan = self._plan_uniform(e)
        elif self.policy == "random":
            plan = self._plan_random(e)
        elif self.policy == "metric":
            plan = self._plan_metric(e)
        else:
            raise ValueError(self.policy)
        out = []
        for kind, layer, score in plan:
            grow_unit(self.model, self.opt, (kind, layer), step, self.gen)
            self.added[kind] += 1
            ev = {"step": step, "event": e, "kind": kind, "layer": layer, "score": score,
                  "params": unit_params(self.model, kind)}
            self.events.append(ev)
            out.append(ev)
        return out

    def _plan_uniform(self, e):
        L = self.model.cfg.n_layer
        plan = []
        for kind, total in (("head", self.add_heads), ("ffn", self.add_chunks)):
            for _ in range(self._quota(e, total)):
                for _try in range(L):
                    l = self.rr[kind] % L
                    self.rr[kind] += 1
                    if can_grow(self.model, (kind, l)) and not self._full_after(plan, kind, l):
                        plan.append((kind, l, None))
                        break
        return plan

    def _full_after(self, plan, kind, l):
        """Would one more unit of `kind` in layer l exceed its free slots given the pending plan?"""
        from growth.operators import inactive_ffn_slots, inactive_head_slots, FFN_CHUNK
        pending = sum(1 for k, ll, _ in plan if k == kind and ll == l)
        free = (len(inactive_ffn_slots(self.model, l)) // FFN_CHUNK) if kind == "ffn" else len(inactive_head_slots(self.model, l))
        return pending + 1 > free

    def _plan_random(self, e):
        L = self.model.cfg.n_layer
        plan = []
        for kind, total in (("head", self.add_heads), ("ffn", self.add_chunks)):
            for _ in range(self._quota(e, total)):
                ok = [l for l in range(L) if not self._full_after(plan, kind, l)]
                plan.append((kind, int(self.rng.choice(ok)), None))
        return plan

    def _plan_metric(self, e):
        L = self.model.cfg.n_layer
        sc = self.ema
        cands = [(k, l) for k in ("head", "ffn") for l in range(L) if uid(k, l) in sc]
        vals = np.array([sc[uid(k, l)] for k, l in cands], dtype=np.float64)
        lo, hi = vals.min(), vals.max()
        w = {c: ((v - lo) / (hi - lo) if hi > lo else 1.0) + 0.1 for c, v in zip(cands, vals)}
        # metrics without head scores (M2): heads fall back to the uniform per-event quota
        heads_scored = any(k == "head" for k, _ in cands)
        remaining = {"head": self.add_heads - self.added["head"], "ffn": self.add_chunks - self.added["ffn"]}
        last = e == self.n_events - 1
        total_params = self.add_heads * unit_params(self.model, "head") + self.add_chunks * unit_params(self.model, "ffn")
        added_params = self.added["head"] * unit_params(self.model, "head") + self.added["ffn"] * unit_params(self.model, "ffn")
        budget = total_params * (e + 1) / self.n_events - added_params
        plan = []
        if not heads_scored:
            for _ in range(self._quota(e, self.add_heads)):
                ok = [l for l in range(L) if not self._full_after(plan, "head", l)]
                plan.append(("head", ok[self.rr["head"] % len(ok)], None))
                self.rr["head"] += 1
                remaining["head"] -= 1
                budget -= unit_params(self.model, "head")
        count = {}
        while True:
            best, best_v = None, -1.0
            for c in cands:
                k, l = c
                if remaining[k] <= 0 or self._full_after(plan, k, l):
                    continue
                if not last and unit_params(self.model, k) > budget + 0.5 * unit_params(self.model, k):
                    continue
                v = w[c] / (1 + count.get(c, 0))
                if v > best_v:
                    best, best_v = c, v
            if best is None:
                break
            k, l = best
            plan.append((k, l, float(sc[uid(k, l)])))
            count[best] = count.get(best, 0) + 1
            remaining[k] -= 1
            budget -= unit_params(self.model, k)
            if not last and budget <= 0:
                break
        return plan

    # ------------------------------------------------------------------ state
    def state_dict(self):
        return {"ema": self.ema, "n_metric_evals": self.n_metric_evals, "metric_seconds": self.metric_seconds,
                "metric_tokens": self.metric_tokens, "events": self.events, "added": self.added, "rr": self.rr,
                "rng": self.rng.bit_generator.state, "gen": self.gen.get_state(), "metric_log": self.metric_log}

    def load_state_dict(self, sd):
        for k in ("ema", "n_metric_evals", "metric_seconds", "metric_tokens", "events", "added", "rr", "metric_log"):
            setattr(self, k, sd[k])
        self.rng.bit_generator.state = sd["rng"]
        self.gen.set_state(sd["gen"].cpu())
