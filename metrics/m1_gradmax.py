"""M1: GradMax score, and M1b: mask gradient of a freshly initialized inactive unit.

M1 (Evci et al., 2022): new units enter with zero fan-out; with a norm-bounded fan-in, the largest
achievable gradient norm of the new fan-out weights is the sum of the top-k squared singular values
of E[g_out x_in^T] (k = units added).
  * FFN chunk (64 neurons): x_in = FFN input (LN2 output), g_out = dL/d(FFN output); GELU linearized
    around 0 (slope 1/2) -> factor 1/4.
  * Head: a fresh head (small random Q,K) attends ~uniformly over the causal prefix, so its value
    input is the running mean of the LN1 output; g_out = dL/d(attention output); k = head_dim = 64.
M1b: for a candidate slot re-initialized with the standard init (mask still 0), |E[dL/dmask]| summed
over the unit's neurons (sign can be learned, so magnitude is the potential first-order gain).
"""
import torch

from growth.operators import FFN_CHUNK, inactive_ffn_slots, inactive_head_slots, reinit_ffn_slots, reinit_head_slot
from metrics.common import capture_pass, causal_mean, per_param, topk_sq_sv, uid


def score_m1(model, batches, opt=None):
    L, d = model.cfg.n_layer, model.cfg.d_model
    dev = next(model.parameters()).device
    Mf = [torch.zeros(d, d, device=dev) for _ in range(L)]
    Ma = [torch.zeros(d, d, device=dev) for _ in range(L)]
    for x, y in batches:
        _, caps, grads = capture_pass(model, x, y)
        for l, c in enumerate(caps):
            xf = c["ffn_in"].detach().float().flatten(0, 1)
            gf = grads[(l, "ffn_out")].flatten(0, 1)
            Mf[l] += gf.T @ xf
            xa = causal_mean(c["attn_in"].detach().float()).flatten(0, 1)
            ga = grads[(l, "attn_out")].flatten(0, 1)
            Ma[l] += ga.T @ xa
    n = len(batches)
    scores = {}
    for l in range(L):
        scores[uid("ffn", l)] = 0.25 * topk_sq_sv(Mf[l] / n, FFN_CHUNK)
        scores[uid("head", l)] = topk_sq_sv(Ma[l] / n, model.cfg.head_dim)
    return per_param(model, scores)


def score_m1b(model, batches, opt=None, seed=1234):
    L = model.cfg.n_layer
    dev = next(model.parameters()).device
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    cand_f, cand_h = {}, {}
    for l in range(L):
        fs = inactive_ffn_slots(model, l)
        if len(fs) >= FFN_CHUNK:
            cand_f[l] = fs[:FFN_CHUNK]
            reinit_ffn_slots(model, l, cand_f[l], gen)  # inactive: does not change the function
        hs = inactive_head_slots(model, l)
        if len(hs) >= 1:
            cand_h[l] = int(hs[0])
            reinit_head_slot(model, l, cand_h[l], gen)
    gf = [None] * L
    gh = [None] * L
    for x, y in batches:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y, capture=True)
        caps = model.captures
        model.captures = None
        tens = [c["ffn_mask"] for c in caps] + [c["head_mask"] for c in caps]
        gs = torch.autograd.grad(loss, tens)
        for l in range(L):
            gf[l] = gs[l] if gf[l] is None else gf[l] + gs[l]
            gh[l] = gs[L + l] if gh[l] is None else gh[l] + gs[L + l]
    scores = {}
    for l in range(L):
        scores[uid("ffn", l)] = float(gf[l][cand_f[l]].abs().sum()) / len(batches) if l in cand_f else 0.0
        scores[uid("head", l)] = float(gh[l][cand_h[l]].abs()) / len(batches) if l in cand_h else 0.0
    return per_param(model, scores)
