"""Scale presets and arm -> model-config construction."""
from growth.operators import FFN_CHUNK
from model.gpt import GPTConfig

SCALES = {
    # name: layers, d_model, target heads/layer, target ffn/layer, token budget, batch (seqs), micro batch
    "S": dict(n_layer=6, d_model=384, heads=6, ffn=1536, tokens=250_000_000, batch=96, micro=48),
    "M": dict(n_layer=8, d_model=640, heads=10, ffn=2560, tokens=800_000_000, batch=288, micro=48),
    # tiny config for tests
    "T": dict(n_layer=2, d_model=128, heads=2, ffn=512, tokens=2_000_000, batch=8, micro=8),
}

ARMS = ["scratch", "scratch_flops", "grow_uniform", "grow_random", "grow_metric", "openelm", "base_half"]


def openelm_widths(L, target, unit, lo=0.5, hi=1.5):
    """Linearly varying per-layer widths (in units) from lo*target to hi*target, summing to L*target."""
    t = target // unit
    raw = [lo * t + (hi - lo) * t * l / (L - 1) for l in range(L)]
    w = [int(round(r)) for r in raw]
    diff = L * t - sum(w)
    i = L - 1
    while diff != 0:  # fix rounding so the total is exact
        w[i] += 1 if diff > 0 else -1
        diff += -1 if diff > 0 else 1
        i = (i - 1) % L
    return [x * unit for x in w]


def model_config(scale, arm, ramp_steps):
    s = SCALES[scale]
    L, H, Fw = s["n_layer"], s["heads"], s["ffn"]
    kw = dict(n_layer=L, d_model=s["d_model"], ramp_steps=ramp_steps)
    if arm in ("scratch", "scratch_flops"):
        kw.update(max_heads=[H] * L, max_ffn=[Fw] * L, init_heads=[H] * L, init_ffn=[Fw] * L)
    elif arm in ("grow_uniform", "grow_random", "grow_metric"):
        kw.update(max_heads=[2 * H] * L, max_ffn=[2 * Fw] * L, init_heads=[H // 2] * L, init_ffn=[Fw // 2] * L)
    elif arm == "base_half":  # oracle base: 50% width + room for exactly one more head and one FFN chunk
        kw.update(max_heads=[H // 2 + 1] * L, max_ffn=[Fw // 2 + FFN_CHUNK] * L, init_heads=[H // 2] * L,
                  init_ffn=[Fw // 2] * L)
    elif arm == "openelm":
        hs = openelm_widths(L, H, 1)
        fs = openelm_widths(L, Fw, FFN_CHUNK)
        kw.update(max_heads=hs, max_ffn=fs, init_heads=hs, init_ffn=fs)
    else:
        raise ValueError(arm)
    return GPTConfig(**kw)


def growth_totals(scale):
    s = SCALES[scale]
    L = s["n_layer"]
    add_heads = L * (s["heads"] - s["heads"] // 2)
    add_chunks = L * (s["ffn"] - s["ffn"] // 2) // FFN_CHUNK
    return add_heads, add_chunks
