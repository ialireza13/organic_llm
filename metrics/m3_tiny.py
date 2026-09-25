"""M3: TINY-style expressivity bottleneck (Verbockhaven et al., 2024).

Desired update of a sublayer's output: v = -dL/d(out). The best update of the existing output
projection (W2 over active hidden units h, or O over active head outputs z) removes the part of v
that is linear in h; what remains, r = v - dW* h, is the expressivity bottleneck. New units with
fan-in on b (FFN input for neurons; causal running mean of LN1 output for a fresh head) can reduce
it by sum_k sigma_k^2 of S_bb^{-1/2} E[b r^T] (top-k, k = 64 new neurons or one 64-dim head).
All statistics are accumulated as second moments in one pass.
"""
import torch

from growth.operators import FFN_CHUNK
from metrics.common import active_ffn_idx, active_head_idx, capture_pass, causal_mean, per_param, uid


def _inv_sqrt_psd(S, rel_eps=1e-4):
    e, U = torch.linalg.eigh(S.double())
    e = e.clamp_min(rel_eps * e.max().clamp_min(1e-30))
    return (U * e.rsqrt()) @ U.T


def _bottleneck_gain(Sbb, Sbv, Sbh, Shh, Shv, k, rel_eps=1e-4):
    Shh = Shh.double()
    ridge = rel_eps * Shh.diagonal().mean().clamp_min(1e-30)
    Shh_reg = Shh + ridge * torch.eye(Shh.shape[0], device=Shh.device, dtype=Shh.dtype)
    Sbr = Sbv.double() - Sbh.double() @ torch.linalg.solve(Shh_reg, Shv.double())
    N = _inv_sqrt_psd(Sbb) @ Sbr
    s = torch.linalg.svdvals(N)
    return float((s[:k] ** 2).sum())


def score_m3(model, batches, opt=None):
    L = model.cfg.n_layer
    st = [dict() for _ in range(L)]
    ntok = 0
    idx_f = [active_ffn_idx(model, l) for l in range(L)]
    hd = model.cfg.head_dim
    idx_z = []
    for l in range(L):
        hs = active_head_idx(model, l)
        idx_z.append((hs.view(-1, 1) * hd + torch.arange(hd, device=hs.device).view(1, -1)).flatten())

    def acc(s, key, A, B):
        v = A.T @ B
        s[key] = s[key] + v if key in s else v

    for x, y in batches:
        _, caps, grads = capture_pass(model, x, y)
        ntok += x.numel()
        for l, c in enumerate(caps):
            s = st[l]
            # FFN
            b = c["ffn_in"].detach().float().flatten(0, 1)
            h = (c["ffn_a"].detach().float() * c["ffn_mask"].detach()).flatten(0, 1)[:, idx_f[l]]
            v = -grads[(l, "ffn_out")].flatten(0, 1) * x.numel()  # undo the 1/N of the mean loss
            acc(s, "f_bb", b, b); acc(s, "f_bv", b, v); acc(s, "f_bh", b, h); acc(s, "f_hh", h, h); acc(s, "f_hv", h, v)
            # heads
            b = causal_mean(c["attn_in"].detach().float()).flatten(0, 1)
            z = c["attn_z"].detach().float().flatten(0, 1)[:, idx_z[l]]
            v = -grads[(l, "attn_out")].flatten(0, 1) * x.numel()
            acc(s, "a_bb", b, b); acc(s, "a_bv", b, v); acc(s, "a_bz", b, z); acc(s, "a_zz", z, z); acc(s, "a_zv", z, v)
    scores = {}
    for l in range(L):
        s = {k: v / ntok for k, v in st[l].items()}
        scores[uid("ffn", l)] = 0.25 * _bottleneck_gain(s["f_bb"], s["f_bv"], s["f_bh"], s["f_hh"], s["f_hv"], FFN_CHUNK)
        scores[uid("head", l)] = _bottleneck_gain(s["a_bb"], s["a_bv"], s["a_bz"], s["a_zz"], s["a_zv"], hd)
    return per_param(model, scores)
