"""Stage 1 plots: oracle values per candidate (both seeds) and metric vs oracle rank scatter."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

S = json.load(open("results/oracle/stage1_summary_oracle400.json"))
OUT = "results/plots"
os.makedirs(OUT, exist_ok=True)

cks = list(S["per_ckpt"])
fig, axes = plt.subplots(1, len(cks), figsize=(6.5 * len(cks), 3.8), squeeze=False)
for ax, ck in zip(axes[0], cks):
    e = S["per_ckpt"][ck]
    c = e["candidates"]
    raw = np.array(e["oracle_raw_per_seed"]) * 1e4
    x = np.arange(len(c))
    ax.bar(x - 0.2, raw[0], 0.4, label="seed 0")
    ax.bar(x + 0.2, raw[1], 0.4, label="seed 1")
    ax.axhline(0, color="k", lw=0.8)
    ctrl = e["control_losses"]
    ax.axhspan(-abs(ctrl[0] - ctrl[1]) * 1e4, abs(ctrl[0] - ctrl[1]) * 1e4, color="gray", alpha=0.2,
               label="|control seed0 − seed1|")
    ax.set_xticks(x, c, rotation=60, fontsize=8)
    ax.set_ylabel("oracle: control − candidate val loss (×1e-4)")
    ax.set_title(f"checkpoint {ck} (400 steps); seed ρ = {e['seed_reliability_spearman']:+.2f}")
    ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "stage1_oracle_values.png"), dpi=120)
plt.close(fig)

metrics = [m for m in S["per_ckpt"][cks[0]]["metrics"] if m != "m3neg"]
fig, axes = plt.subplots(1, len(metrics), figsize=(2.6 * len(metrics), 2.8), squeeze=False)
for ax, m in zip(axes[0], metrics):
    for ck, mk in zip(cks, ("o", "s")):
        e = S["per_ckpt"][ck]
        sc = e["metrics"][m]["scores"]
        ok = [i for i, v in enumerate(sc) if v is not None]
        orc = np.array(e["oracle_per_param"])[ok]
        ms = np.array([sc[i] for i in ok])
        col = ["C0" if e["candidates"][i].startswith("ffn") else "C3" for i in ok]
        ax.scatter(rankdata(ms), rankdata(orc), c=col, marker=mk, s=18, alpha=0.8)
    ax.set_title(f"{m}: ρ={S['pooled'][m]['spearman_all'] if not np.isnan(S['pooled'][m]['spearman_all']) else S['pooled'][m]['spearman_ffn']:+.2f}", fontsize=9)
    ax.set_xlabel("metric rank", fontsize=8)
axes[0][0].set_ylabel("oracle rank", fontsize=8)
fig.suptitle("Stage 1: metric vs oracle rank per checkpoint (o = 30%, □ = 60%; blue FFN, red heads)", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "stage1_metric_vs_oracle.png"), dpi=120)
plt.close(fig)
print("ok")
