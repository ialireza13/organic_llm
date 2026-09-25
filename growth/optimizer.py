"""AdamW with a per-element step counter, so moments of newly grown units can be reset exactly.

Resetting a slice sets exp_avg, exp_avg_sq and step to 0 for those elements; the next update then
gets the correct bias correction, as if the unit's parameters were freshly added to the optimizer.
"""
import torch


class GrowableAdamW:
    def __init__(self, model, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
        self.betas, self.eps, self.wd = betas, eps, weight_decay
        self.lr = lr
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        # decay 2D weights only (matmuls + embeddings); tied weights appear once in named_parameters
        self.groups = [
            {"params": [p for n, p in named if p.dim() >= 2], "names": [n for n, p in named if p.dim() >= 2], "wd": weight_decay},
            {"params": [p for n, p in named if p.dim() < 2], "names": [n for n, p in named if p.dim() < 2], "wd": 0.0},
        ]
        self.state = {}
        for g in self.groups:
            for p in g["params"]:
                self.state[p] = {
                    "exp_avg": torch.zeros_like(p, memory_format=torch.preserve_format),
                    "exp_avg_sq": torch.zeros_like(p, memory_format=torch.preserve_format),
                    "step": torch.zeros_like(p, memory_format=torch.preserve_format),
                }

    def all_params(self):
        return [p for g in self.groups for p in g["params"]]

    def zero_grad(self):
        for p in self.all_params():
            p.grad = None

    @torch.no_grad()
    def step(self, lr=None):
        lr = self.lr if lr is None else lr
        b1, b2 = self.betas
        for g in self.groups:
            ps = [p for p in g["params"] if p.grad is not None]
            if not ps:
                continue
            grads = [p.grad for p in ps]
            ms = [self.state[p]["exp_avg"] for p in ps]
            vs = [self.state[p]["exp_avg_sq"] for p in ps]
            ss = [self.state[p]["step"] for p in ps]
            torch._foreach_add_(ss, 1.0)
            torch._foreach_lerp_(ms, grads, 1 - b1)
            torch._foreach_mul_(vs, b2)
            torch._foreach_addcmul_(vs, grads, grads, 1 - b2)
            if g["wd"] > 0:
                torch._foreach_mul_(ps, 1 - lr * g["wd"])
            bc1 = torch._foreach_pow(b1, ss)  # b1 ** step
            torch._foreach_neg_(bc1)
            torch._foreach_add_(bc1, 1.0)  # 1 - b1**step
            bc2 = torch._foreach_pow(b2, ss)
            torch._foreach_neg_(bc2)
            torch._foreach_add_(bc2, 1.0)
            denom = torch._foreach_div(vs, bc2)
            torch._foreach_sqrt_(denom)
            torch._foreach_add_(denom, self.eps)
            num = torch._foreach_div(ms, bc1)
            torch._foreach_div_(num, denom)
            torch._foreach_add_(ps, num, alpha=-lr)

    @torch.no_grad()
    def reset(self, p, rows=None, cols=None):
        """Zero moments and step for p[rows, :] or p[:, cols] (index tensors/lists)."""
        st = self.state[p]
        for k in ("exp_avg", "exp_avg_sq", "step"):
            if rows is not None:
                st[k][rows] = 0
            if cols is not None:
                st[k][:, cols] = 0

    def state_dict(self):
        out = {"lr": self.lr, "betas": self.betas, "eps": self.eps, "wd": self.wd, "state": []}
        for g in self.groups:
            for n, p in zip(g["names"], g["params"]):
                out["state"].append((n, {k: v.clone() for k, v in self.state[p].items()}))
        return out

    def load_state_dict(self, sd):
        by_name = dict(sd["state"])
        for g in self.groups:
            for n, p in zip(g["names"], g["params"]):
                for k, v in by_name[n].items():
                    self.state[p][k].copy_(v)
