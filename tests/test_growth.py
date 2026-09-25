import copy
import math

import pytest
import torch

from configs.scales import model_config
from growth.operators import FFN_CHUNK, grow_ffn, grow_head, head_rows
from growth.optimizer import GrowableAdamW
from model.gpt import GPT, GPTConfig

DEV = "cuda"


def tiny_growth_model(seed=0, ramp=4):
    torch.manual_seed(seed)
    cfg = GPTConfig(n_layer=3, d_model=128, max_heads=[4, 4, 4], max_ffn=[512, 512, 512],
                    init_heads=[1, 2, 1], init_ffn=[128, 192, 64], ramp_steps=ramp, seq_len=64)
    return GPT(cfg).to(DEV)


def batch(seed=0, B=4, T=64):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 50257, (B, T + 1), generator=g).to(DEV)
    return x[:, :-1], x[:, 1:]


@pytest.fixture(autouse=True)
def fp32():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yield


def test_function_preservation_every_operator():
    m = tiny_growth_model()
    opt = GrowableAdamW(m)
    x, y = batch()
    gen = torch.Generator(device=DEV).manual_seed(1)
    step = 10
    m.update_masks(step)
    with torch.no_grad():
        ref = m(x).clone()
    for l in range(3):
        grow_ffn(m, opt, l, step, gen=gen)
        m.update_masks(step)
        with torch.no_grad():
            out = m(x)
        assert torch.allclose(out, ref, atol=1e-5, rtol=0), f"ffn grow layer {l}: {(out - ref).abs().max()}"
        grow_head(m, opt, l, step, gen=gen)
        m.update_masks(step)
        with torch.no_grad():
            out = m(x)
        assert torch.allclose(out, ref, atol=1e-5, rtol=0), f"head grow layer {l}: {(out - ref).abs().max()}"


def test_inactive_units_do_not_affect_output():
    m = tiny_growth_model()
    x, _ = batch()
    with torch.no_grad():
        ref = m(x).clone()
        for b in m.blocks:
            inact = (b.ffn.mask == 0).nonzero().flatten()
            b.ffn.w1.weight[inact] = torch.randn_like(b.ffn.w1.weight[inact]) * 10
            b.ffn.w2.weight[:, inact] = torch.randn_like(b.ffn.w2.weight[:, inact]) * 10
            b.attn.qkv.weight.mul_(1.0)
            for h in (b.attn.mask == 0).nonzero().flatten().tolist():
                rows, cols = head_rows(m, list(m.blocks).index(b), h)
                b.attn.qkv.weight[rows] = torch.randn_like(b.attn.qkv.weight[rows]) * 10
                b.attn.o.weight[:, cols] = torch.randn_like(b.attn.o.weight[:, cols]) * 10
        out = m(x)
    assert torch.allclose(out, ref, atol=1e-5, rtol=0)


def test_new_units_get_gradients_once_ramp_starts():
    m = tiny_growth_model(ramp=4)
    opt = GrowableAdamW(m)
    x, y = batch()
    gen = torch.Generator(device=DEV).manual_seed(2)
    step = 5
    idx = grow_ffn(m, opt, 1, step, gen=gen)
    h = grow_head(m, opt, 1, step, gen=gen)
    rows, cols = head_rows(m, 1, h)
    b = m.blocks[1]
    # at the growth step the mask is exactly 0 -> no gradient into the new unit
    m.update_masks(step)
    _, loss = m(x, y)
    loss.backward()
    assert b.ffn.w1.weight.grad[idx].abs().max() == 0
    assert b.attn.o.weight.grad[:, cols].abs().max() == 0
    opt.zero_grad()
    # first step after growth: mask = 1/ramp > 0 -> non-zero gradients for every new parameter
    m.update_masks(step + 1)
    assert math.isclose(float(b.ffn.mask[idx[0]]), 0.25)
    _, loss = m(x, y)
    loss.backward()
    assert (b.ffn.w1.weight.grad[idx].abs().sum(1) > 0).all()
    assert (b.ffn.w2.weight.grad[:, idx].abs().sum(0) > 0).all()
    assert (b.attn.qkv.weight.grad[rows].abs().sum(1) > 0).all()
    assert (b.attn.o.weight.grad[:, cols].abs().sum(0) > 0).all()
    m.update_masks(step + 4)
    assert float(b.ffn.mask[idx[0]]) == 1.0


def test_optimizer_covers_params_and_resets_moments():
    m = tiny_growth_model()
    opt = GrowableAdamW(m)
    in_opt = {id(p) for p in opt.all_params()}
    for n, p in m.named_parameters():
        assert id(p) in in_opt, f"{n} not in optimizer"
    x, y = batch()
    for s in range(3):
        m.update_masks(s)
        _, loss = m(x, y)
        loss.backward()
        opt.step(1e-3)
        opt.zero_grad()
    b = m.blocks[0]
    st1 = opt.state[b.ffn.w1.weight]
    before = {k: v.clone() for k, v in st1.items()}
    gen = torch.Generator(device=DEV).manual_seed(3)
    idx = grow_ffn(m, opt, 0, 3, gen=gen)
    h = grow_head(m, opt, 0, 3, gen=gen)
    rows, cols = head_rows(m, 0, h)
    for k in ("exp_avg", "exp_avg_sq", "step"):
        assert st1[k][idx].abs().max() == 0
        assert opt.state[b.ffn.w2.weight][k][:, idx].abs().max() == 0
        assert opt.state[b.attn.qkv.weight][k][rows].abs().max() == 0
        assert opt.state[b.attn.o.weight][k][:, cols].abs().max() == 0
    keep = torch.ones(st1["step"].shape[0], dtype=torch.bool, device=DEV)
    keep[idx] = False
    for k in ("exp_avg", "exp_avg_sq", "step"):
        assert torch.equal(st1[k][keep], before[k][keep])
    # active (initially active) params have been updated 3 times
    act = (b.ffn.birth < 0).nonzero().flatten()
    assert float(st1["step"][act].min()) == 3.0
    # first update of the new unit uses correct bias correction: step counter restarts at 1
    m.update_masks(4)
    _, loss = m(x, y)
    loss.backward()
    opt.step(1e-3)
    assert float(st1["step"][idx].max()) == 1.0


def test_flop_counter_hand_calculation():
    cfg = model_config("S", "scratch", 1)
    m = GPT(cfg)
    # hand calculation (S at target): L=6, d=384, 6 heads x 64, FFN 1536, V=50304, T=1024
    N = 6 * (4 * 384 * 64 * 6 + 2 * 384 * 1536) + 384 * 50304
    assert N == 29_933_568
    attn = 6 * 12 * (64 * 6) * 1024
    assert m.flops_per_token() == 6 * N + attn == 207_912_960
    # growth config at 50%: grow one head + one chunk in layer 0 -> adds 6*(4*384*64 + 2*384*64) + 12*64*1024
    g = GPT(model_config("S", "grow_uniform", 1))
    f0 = g.flops_per_token()
    grow_head(g, None, 0, 0, gen=torch.Generator().manual_seed(0))
    grow_ffn(g, None, 0, 0, gen=torch.Generator().manual_seed(0))
    assert g.flops_per_token() - f0 == 6 * (4 * 384 * 64 + 2 * 384 * 64) + 12 * 64 * 1024
