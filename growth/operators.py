"""Function-preserving growth operators: add FFN neuron chunks or attention heads behind a 0-mask.

Growing = pick the lowest-index inactive slots in the layer, re-initialize their weights with the
standard init, reset their AdamW moments, and set birth = step so the mask ramps 0 -> 1.
"""
import torch

FFN_CHUNK = 64


def _gen(model, seed):
    dev = next(model.parameters()).device
    g = torch.Generator(device=dev)
    g.manual_seed(int(seed))
    return g


def inactive_ffn_slots(model, layer):
    ffn = model.blocks[layer].ffn
    return (ffn.birth >= 1e8).nonzero().flatten()


def inactive_head_slots(model, layer):
    at = model.blocks[layer].attn
    return (at.birth >= 1e8).nonzero().flatten()


@torch.no_grad()
def reinit_ffn_slots(model, layer, idx, gen):
    ffn = model.blocks[layer].ffn
    d = ffn.w1.weight.shape[1]
    ffn.w1.weight[idx] = torch.randn(len(idx), d, generator=gen, device=idx.device) * 0.02
    model.init_residual_(ffn.w2.weight, gen=gen, cols=idx)


@torch.no_grad()
def head_rows(model, layer, h):
    at = model.blocks[layer].attn
    H, hd = at.H, at.hd
    base = torch.arange(h * hd, (h + 1) * hd, device=at.qkv.weight.device)
    return torch.cat([base, base + H * hd, base + 2 * H * hd]), base


@torch.no_grad()
def reinit_head_slot(model, layer, h, gen):
    at = model.blocks[layer].attn
    rows, cols = head_rows(model, layer, h)
    d = at.qkv.weight.shape[1]
    at.qkv.weight[rows] = torch.randn(len(rows), d, generator=gen, device=rows.device) * 0.02
    model.init_residual_(at.o.weight, gen=gen, cols=cols)


@torch.no_grad()
def grow_ffn(model, opt, layer, step, n_neurons=FFN_CHUNK, gen=None):
    free = inactive_ffn_slots(model, layer)
    if len(free) < n_neurons:
        raise ValueError(f"layer {layer}: only {len(free)} free FFN slots")
    idx = free[:n_neurons]
    ffn = model.blocks[layer].ffn
    reinit_ffn_slots(model, layer, idx, gen)
    if opt is not None:
        opt.reset(ffn.w1.weight, rows=idx)
        opt.reset(ffn.w2.weight, cols=idx)
    ffn.birth[idx] = float(step)
    ffn.mask[idx] = 0.0
    model.active_ffn[layer] += n_neurons
    return idx


@torch.no_grad()
def grow_head(model, opt, layer, step, gen=None):
    free = inactive_head_slots(model, layer)
    if len(free) < 1:
        raise ValueError(f"layer {layer}: no free head slots")
    h = int(free[0])
    at = model.blocks[layer].attn
    reinit_head_slot(model, layer, h, gen)
    if opt is not None:
        rows, cols = head_rows(model, layer, h)
        opt.reset(at.qkv.weight, rows=rows)
        opt.reset(at.o.weight, cols=cols)
    at.birth[h] = float(step)
    at.mask[h] = 0.0
    model.active_heads[layer] += 1
    return h


def grow_unit(model, opt, unit, step, gen):
    """unit = ('ffn', layer) or ('head', layer)."""
    kind, layer = unit
    if kind == "ffn":
        return grow_ffn(model, opt, layer, step, gen=gen)
    return grow_head(model, opt, layer, step, gen=gen)


def unit_params(model, kind):
    d, hd = model.cfg.d_model, model.cfg.head_dim
    return 2 * d * FFN_CHUNK if kind == "ffn" else 4 * d * hd


def can_grow(model, unit):
    kind, layer = unit
    if kind == "ffn":
        return len(inactive_ffn_slots(model, layer)) >= FFN_CHUNK
    return len(inactive_head_slots(model, layer)) >= 1
