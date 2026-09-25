"""M2: splitting eigenvalue (Liu et al., 2019; Firefly), FFN only.

For active neuron i with input weights w_i: S_i = E_x[ g_i(x) * gelu''(w_i^T x) * x x^T ], where
g_i = dL/d(activation_i). Splitting along the eigenvector of lambda_min(S_i) < 0 decreases the loss by
~ eps^2/2 * |lambda_min|. The layer score for adding a 64-neuron chunk is the sum of the 64 most
negative lambda_min (as positive gains), divided by the chunk's parameters.
lambda_min is found for all neurons at once by batched power iteration using matvecs
S_i v = sum_t c_ti (x_t . v) x_t on a token subsample (no explicit d x d matrices).
Heads are not scored (returns only ffn:* keys).
"""
import math

import torch

from growth.operators import FFN_CHUNK
from metrics.common import active_ffn_idx, capture_pass, per_param, uid


def gelu_dd(u):
    phi = torch.exp(-0.5 * u * u) / math.sqrt(2 * math.pi)
    return phi * (2 - u * u)


def _matvec(X, C, V):
    # X [N,d], C [N,F], V [F,d] -> S_i v_i for each neuron i: [F,d]
    return (C * (X @ V.T)).T @ X


def _power(X, C, V, iters, shift=None):
    for _ in range(iters):
        W = _matvec(X, C, V)
        if shift is not None:
            W = W - shift.view(-1, 1) * V
        V = W / W.norm(dim=1, keepdim=True).clamp_min(1e-30)
    W = _matvec(X, C, V)
    if shift is not None:
        W = W - shift.view(-1, 1) * V
    return (W * V).sum(1), V  # Rayleigh quotients


def score_m2(model, batches, opt=None, tokens_per_batch=2048, iters=30, seed=0):
    L = model.cfg.n_layer
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    Xs = [[] for _ in range(L)]
    Cs = [[] for _ in range(L)]
    idx = [active_ffn_idx(model, l) for l in range(L)]
    ntot = 0
    for x, y in batches:
        _, caps, grads = capture_pass(model, x, y)
        n = x.numel()
        sel = torch.randperm(n, generator=gen, device="cuda")[:tokens_per_batch]
        ntot += n
        for l, c in enumerate(caps):
            X = c["ffn_in"].detach().float().flatten(0, 1)[sel]
            u = c["ffn_u"].detach().float().flatten(0, 1)[sel][:, idx[l]]
            g = grads[(l, "ffn_a")].flatten(0, 1)[sel][:, idx[l]] * n  # per-token dL/da (undo mean)
            Xs[l].append(X)
            Cs[l].append(g * gelu_dd(u))
    scores = {}
    for l in range(L):
        X = torch.cat(Xs[l])
        C = torch.cat(Cs[l]) / X.shape[0]
        Fa = C.shape[1]
        V0 = torch.randn(Fa, X.shape[1], device=X.device, generator=gen)
        V0 = V0 / V0.norm(dim=1, keepdim=True)
        mu, _ = _power(X, C, V0, iters)
        shift = mu.abs()
        lam_shifted, _ = _power(X, C, V0, iters, shift=shift)
        lam_min = lam_shifted + shift
        gains = (-lam_min).clamp_min(0)
        scores[uid("ffn", l)] = float(gains.topk(min(FFN_CHUNK, Fa)).values.sum())
    return per_param(model, scores)
