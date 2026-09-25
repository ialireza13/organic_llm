"""GPT-style pre-LN transformer with static-shape, mask-gated growable units.

Every layer pre-allocates `max_heads[l]` attention heads (head_dim 64) and `max_ffn[l]` FFN neurons.
Each unit has a scalar mask (FFN: on the hidden activation; attention: on the head output before O).
A unit's mask is clamp((step - birth) / ramp, 0, 1):
  * initially active units: birth = -BIG  -> mask 1
  * inactive units:         birth = +BIG  -> mask 0 (no effect on the output)
  * grown at step s:        birth = s     -> mask 0 at step s, ramps linearly to 1 over `ramp` steps
"""
import math
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

BIG = 1e9


@dataclass
class GPTConfig:
    n_layer: int = 6
    d_model: int = 384
    head_dim: int = 64
    vocab_size: int = 50304  # GPT-2 vocab padded to a multiple of 64
    seq_len: int = 1024
    max_heads: list = field(default_factory=lambda: [6] * 6)
    max_ffn: list = field(default_factory=lambda: [1536] * 6)
    init_heads: list = field(default_factory=lambda: [6] * 6)
    init_ffn: list = field(default_factory=lambda: [1536] * 6)
    ramp_steps: int = 50
    tie_embeddings: bool = True

    def to_dict(self):
        return asdict(self)


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig, H: int):
        super().__init__()
        self.H, self.hd = H, cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * H * cfg.head_dim, bias=False)
        self.o = nn.Linear(H * cfg.head_dim, cfg.d_model, bias=False)
        self.register_buffer("mask", torch.zeros(H))
        self.register_buffer("birth", torch.full((H,), BIG))

    def forward(self, x, cap=None):
        B, T, _ = x.shape
        H, hd = self.H, self.hd
        q, k, v = self.qkv(x).split(H * hd, dim=2)
        q = q.view(B, T, H, hd).transpose(1, 2)
        k = k.view(B, T, H, hd).transpose(1, 2)
        v = v.view(B, T, H, hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # B,H,T,hd
        mask = self.mask
        if cap is not None:
            mask = mask.detach().clone().requires_grad_(True)
            cap["head_mask"] = mask
            cap["head_y"] = y  # pre-mask head outputs
        y = y * mask.view(1, H, 1, 1).to(y.dtype)
        z = y.transpose(1, 2).reshape(B, T, H * hd)
        if cap is not None:
            cap["attn_z"] = z
        out = self.o(z)
        if cap is not None:
            cap["attn_out"] = out
        return out


class FFN(nn.Module):
    def __init__(self, cfg: GPTConfig, Fm: int):
        super().__init__()
        self.F = Fm
        self.w1 = nn.Linear(cfg.d_model, Fm, bias=False)
        self.w2 = nn.Linear(Fm, cfg.d_model, bias=False)
        self.register_buffer("mask", torch.zeros(Fm))
        self.register_buffer("birth", torch.full((Fm,), BIG))

    def forward(self, x, cap=None):
        u = self.w1(x)
        a = F.gelu(u)
        mask = self.mask
        if cap is not None:
            mask = mask.detach().clone().requires_grad_(True)
            cap["ffn_mask"] = mask
            cap["ffn_u"], cap["ffn_a"] = u, a
        h = a * mask.to(a.dtype)
        out = self.w2(h)
        if cap is not None:
            cap["ffn_out"] = out
        return out


class Block(nn.Module):
    def __init__(self, cfg, H, Fm):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg, H)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FFN(cfg, Fm)

    def forward(self, x, cap=None):
        if cap is None:
            x = x + self.attn(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
            return x
        xa = self.ln1(x)
        cap["attn_in"], cap["resid0"] = xa, x
        x1 = x + self.attn(xa, cap)
        xf = self.ln2(x1)
        cap["ffn_in"], cap["resid1"] = xf, x1
        x2 = x1 + self.ffn(xf, cap)
        cap["resid2"] = x2
        return x2


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        L = cfg.n_layer
        assert len(cfg.max_heads) == L and len(cfg.max_ffn) == L
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, cfg.max_heads[l], cfg.max_ffn[l]) for l in range(L)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.wte.weight
        self.apply(self._init)
        for b in self.blocks:
            self.init_residual_(b.attn.o.weight)
            self.init_residual_(b.ffn.w2.weight)
        # initially active units
        for l, b in enumerate(self.blocks):
            b.attn.birth[: cfg.init_heads[l]] = -BIG
            b.ffn.birth[: cfg.init_ffn[l]] = -BIG
        # python-side active counts (units grown so far, incl. ramping ones) for FLOP accounting
        self.active_heads = list(cfg.init_heads)
        self.active_ffn = list(cfg.init_ffn)
        self.update_masks(0)
        self.captures = None  # list of per-layer dicts when capturing

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def init_residual_(self, w, gen=None, cols=None):
        std = 0.02 / math.sqrt(2 * self.cfg.n_layer)
        if cols is None:
            nn.init.normal_(w, 0.0, std)
        else:
            w[:, cols] = torch.randn(w.shape[0], len(cols), generator=gen, device=w.device) * std

    @torch.no_grad()
    def update_masks(self, step):
        r = max(1, self.cfg.ramp_steps)
        for b in self.blocks:
            for mod in (b.attn, b.ffn):
                mod.mask.copy_(((step - mod.birth) / r).clamp_(0.0, 1.0))

    def forward(self, idx, targets=None, capture=False):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        if capture:
            self.captures = [dict() for _ in self.blocks]
            for b, cap in zip(self.blocks, self.captures):
                x = b(x, cap)
        else:
            for b in self.blocks:
                x = b(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    # ---------------- accounting ----------------
    def active_matmul_params(self):
        """Parameters that participate in matmuls for active units (non-embedding + lm_head)."""
        d, hd = self.cfg.d_model, self.cfg.head_dim
        n = sum(4 * d * hd * h + 2 * d * f for h, f in zip(self.active_heads, self.active_ffn))
        return n + d * self.cfg.vocab_size

    def active_nonembedding_params(self):
        d, hd = self.cfg.d_model, self.cfg.head_dim
        n = sum(4 * d * hd * h + 2 * d * f for h, f in zip(self.active_heads, self.active_ffn))
        n += (4 * d) * self.cfg.n_layer + 2 * d  # layer norms
        return n

    def flops_per_token(self):
        """Training FLOPs per token (fwd+bwd) for active units: 6*N_matmul + 12*sum_l(d_attn_l)*T."""
        hd, T = self.cfg.head_dim, self.cfg.seq_len
        attn = sum(12 * hd * h * T for h in self.active_heads)
        return 6 * self.active_matmul_params() + attn

    def allocation(self):
        return {"heads": list(self.active_heads), "ffn": list(self.active_ffn)}
