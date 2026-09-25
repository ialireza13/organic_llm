"""Shared helpers for growth metrics.

Every metric exposes score(model, batches, opt=None) -> {unit_id: score}, unit_id in
{"ffn:<l>", "head:<l>"}. Higher = grow here. "Extensive" metrics (estimated loss decrease for adding
one unit: M1, M1b, M2, M3) are divided by the unit's parameter count; "intensive" metrics (per-layer
need signals: M5, M6, M7) are already per-parameter quantities and are left as is.
"""
import torch

from growth.operators import FFN_CHUNK, unit_params


def uid(kind, layer):
    return f"{kind}:{layer}"


def parse_uid(u):
    k, l = u.split(":")
    return k, int(l)


def active_ffn_idx(model, l):
    return (model.blocks[l].ffn.mask > 0).nonzero().flatten()


def active_head_idx(model, l):
    return (model.blocks[l].attn.mask > 0).nonzero().flatten()


def capture_pass(model, x, y, want_grads=True, extra=()):
    """Eager forward with captures; returns (loss, caps, grads dict keyed by (layer, name))."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, loss = model(x, y, capture=True)
    caps = model.captures
    model.captures = None
    grads = {}
    if want_grads:
        names = ["attn_out", "ffn_out", "ffn_a"] + list(extra)
        tens, keys = [], []
        for l, c in enumerate(caps):
            for n in names:
                tens.append(c[n])
                keys.append((l, n))
        gs = torch.autograd.grad(loss, tens, allow_unused=True)
        grads = {k: (g.float() if g is not None else None) for k, g in zip(keys, gs)}
    return loss.detach(), caps, grads


def causal_mean(x):
    """x: [B,T,d] -> running mean over positions (uniform causal attention of a fresh head)."""
    T = x.shape[1]
    denom = torch.arange(1, T + 1, device=x.device, dtype=x.dtype).view(1, T, 1)
    return x.cumsum(1) / denom


def per_param(model, scores):
    out = {}
    for u, s in scores.items():
        k, _ = parse_uid(u)
        out[u] = s / unit_params(model, k)
    return out


def topk_sq_sv(M, k=FFN_CHUNK):
    s = torch.linalg.svdvals(M.float())
    return float((s[:k] ** 2).sum())
