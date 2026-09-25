"""Cheap metrics M5 (gradient SNR from AdamW state), M6 (spectral saturation), M7 (block influence).

These are per-layer "need" signals (intensive), not loss-decrease estimates, so they are not divided
by unit parameters.
"""
import torch

from growth.operators import head_rows
from metrics.common import active_ffn_idx, active_head_idx, uid


@torch.no_grad()
def score_m5(model, batches=None, opt=None):
    """Mean over active units of ||m_u||^2 / sum(v_u) (Adam first/second moments of the unit's params)."""
    assert opt is not None, "M5 needs the optimizer"
    scores = {}
    for l, b in enumerate(model.blocks):
        st1, st2 = opt.state[b.ffn.w1.weight], opt.state[b.ffn.w2.weight]
        i = active_ffn_idx(model, l)
        m2 = st1["exp_avg"][i].pow(2).sum(1) + st2["exp_avg"][:, i].pow(2).sum(0)
        v = st1["exp_avg_sq"][i].sum(1) + st2["exp_avg_sq"][:, i].sum(0)
        scores[uid("ffn", l)] = float((m2 / v.clamp_min(1e-30)).mean())
        sq, so = opt.state[b.attn.qkv.weight], opt.state[b.attn.o.weight]
        snr = []
        for h in active_head_idx(model, l).tolist():
            rows, cols = head_rows(model, l, h)
            m2 = sq["exp_avg"][rows].pow(2).sum() + so["exp_avg"][:, cols].pow(2).sum()
            v = sq["exp_avg_sq"][rows].sum() + so["exp_avg_sq"][:, cols].sum()
            snr.append(m2 / v.clamp_min(1e-30))
        scores[uid("head", l)] = float(torch.stack(snr).mean())
    return scores


def _stable_rank_frac(W):
    s = torch.linalg.svdvals(W.float())
    return float((s.pow(2).sum() / s[0].pow(2)) / min(W.shape))


@torch.no_grad()
def score_m6(model, batches, opt=None, dormant_tau=0.1):
    """Normalized stable rank of the active weights x (1 - dormant-unit fraction)."""
    L = model.cfg.n_layer
    act_f = [0.0] * L
    act_h = [0.0] * L
    for x, y in batches:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(x, y, capture=True)
        caps = model.captures
        model.captures = None
        for l, c in enumerate(caps):
            act_f[l] = act_f[l] + c["ffn_a"].float().abs().mean((0, 1))
            act_h[l] = act_h[l] + c["head_y"].float().norm(dim=-1).mean((0, 2))
    scores = {}
    for l, b in enumerate(model.blocks):
        i = active_ffn_idx(model, l)
        a = act_f[l][i]
        dormant = float((a / a.mean() <= dormant_tau).float().mean())
        scores[uid("ffn", l)] = _stable_rank_frac(b.ffn.w1.weight[i]) * (1 - dormant)
        hs = active_head_idx(model, l)
        hd = model.cfg.head_dim
        cols = (hs.view(-1, 1) * hd + torch.arange(hd, device=hs.device).view(1, -1)).flatten()
        ah = act_h[l][hs]
        dormant_h = float((ah / ah.mean() <= dormant_tau).float().mean())
        scores[uid("head", l)] = _stable_rank_frac(b.attn.o.weight[:, cols]) * (1 - dormant_h)
    return scores


@torch.no_grad()
def score_m7(model, batches, opt=None):
    """Sublayer influence 1 - cos(x_in, x_out) on the residual stream (attention and FFN separately)."""
    L = model.cfg.n_layer
    sa, sf = [0.0] * L, [0.0] * L
    for x, y in batches:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(x, y, capture=True)
        caps = model.captures
        model.captures = None
        for l, c in enumerate(caps):
            r0, r1, r2 = (c[k].float() for k in ("resid0", "resid1", "resid2"))
            sa[l] += float(1 - torch.nn.functional.cosine_similarity(r0, r1, dim=-1).mean())
            sf[l] += float(1 - torch.nn.functional.cosine_similarity(r1, r2, dim=-1).mean())
    n = len(batches)
    scores = {}
    for l in range(L):
        scores[uid("ffn", l)] = sf[l] / n
        scores[uid("head", l)] = sa[l] / n
    return scores
