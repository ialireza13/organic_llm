"""Stage 2 tables and plots from results/runs/*/ (summary.json, val_log.csv, growth_events.csv)."""
import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ORDER = ["grow_metric", "grow_random", "grow_uniform", "scratch", "scratch_flops", "openelm"]
LABEL = {"grow_random": "(b) random growth", "grow_uniform": "(c) uniform growth", "scratch": "(d) scratch, same tokens",
         "scratch_flops": "(e) scratch, matched FLOPs", "openelm": "(f) OpenELM-style widths"}


def arm_key(s):
    return f"grow_metric:{s['metric']}" if s["arm"] == "grow_metric" else s["arm"]


def arm_label(k):
    if k.startswith("grow_metric:"):
        return f"(a) metric growth [{k.split(':')[1]}]"
    return LABEL.get(k, k)


def load(res_dir):
    rows = []
    for f in glob.glob(os.path.join(res_dir, "*", "summary.json")):
        s = json.load(open(f))
        s["dir"] = os.path.dirname(f)
        s["key"] = arm_key(s)
        rows.append(s)
    return rows


def sort_keys(keys):
    return sorted(keys, key=lambda k: (ORDER.index(k.split(":")[0]) if k.split(":")[0] in ORDER else 99, k))


def table(rows, scale):
    rs = [r for r in rows if r["scale"] == scale and r["status"] == "ok"]
    keys = sort_keys({r["key"] for r in rs})
    lines = ["| arm | seeds | final val loss (mean ± std) | per-seed | train FLOPs | metric s (total) | tokens/s | final heads / FFN per layer (seed 0) |",
             "|---|---|---|---|---|---|---|---|"]
    stats = {}
    for k in keys:
        g = sorted([r for r in rs if r["key"] == k], key=lambda r: r["seed"])
        v = np.array([r["final_val_loss"] for r in g])
        sd = v.std(ddof=1) if len(v) > 1 else float("nan")
        stats[k] = {"mean": float(v.mean()), "std": float(sd), "n": len(v), "per_seed": {r["seed"]: r["final_val_loss"] for r in g}}
        al = g[0]["allocation"]
        lines.append(f"| {arm_label(k)} | {len(g)} | {v.mean():.4f} ± {sd:.4f} | " + ", ".join(f"{x:.4f}" for x in v) +
                     f" | {np.mean([r['train_flops'] for r in g]):.3e} | {np.mean([r['metric_seconds'] for r in g]):.0f} | "
                     f"{np.mean([r['tokens_per_s'] for r in g]):.0f} | {al['heads']} / {[f // 64 for f in al['ffn']]}×64 |")
    return "\n".join(lines), stats


def paired(stats, a, b):
    """Paired (same seed = same data order) difference a - b."""
    sa, sb = stats[a]["per_seed"], stats[b]["per_seed"]
    common = sorted(set(sa) & set(sb))
    d = np.array([sa[s] - sb[s] for s in common])
    if len(d) == 0:
        return None
    return {"n": len(d), "mean": float(d.mean()), "std": float(d.std(ddof=1)) if len(d) > 1 else float("nan"), "diffs": d.tolist()}


def plots(rows, scale, out):
    os.makedirs(out, exist_ok=True)
    rs = [r for r in rows if r["scale"] == scale and r["status"] == "ok"]
    keys = sort_keys({r["key"] for r in rs})
    cmap = plt.get_cmap("tab10")
    for xcol, fname in (("tokens", "val_vs_tokens"), ("flops", "val_vs_flops")):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        for i, k in enumerate(keys):
            for r in [r for r in rs if r["key"] == k]:
                df = pd.read_csv(os.path.join(r["dir"], "val_log.csv"))
                df = df[df.step > 0]
                for ax in axes:
                    ax.plot(df[xcol], df.val_loss, color=cmap(i), alpha=0.8, lw=1.2,
                            label=arm_label(k) if r["seed"] == min(x["seed"] for x in rs if x["key"] == k) else None)
        axes[0].set_ylim(None, 5.0)
        # zoom on the end of training
        allv = [pd.read_csv(os.path.join(r["dir"], "val_log.csv")).val_loss.iloc[-1] for r in rs]
        axes[1].set_ylim(min(allv) - 0.02, max(allv) + 0.12)
        xmax = max(pd.read_csv(os.path.join(r["dir"], "val_log.csv"))[xcol].max() for r in rs)
        axes[1].set_xlim(0.5 * xmax, 1.02 * xmax)
        for ax in axes:
            ax.set_xlabel("training tokens" if xcol == "tokens" else "training FLOPs (active params)")
            ax.set_ylabel("val loss")
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        axes[1].set_title("zoom: second half")
        fig.suptitle(f"Scale {scale}: validation loss vs {xcol}")
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"{scale}_{fname}.png"), dpi=120)
        plt.close(fig)
    # allocation heatmaps for growth arms (non-uniform ones)
    gk = [k for k in keys if k.startswith("grow_metric") or k == "grow_random"]
    if gk:
        seeds = sorted({r["seed"] for r in rs})
        fig, axes = plt.subplots(2, len(gk), figsize=(3.2 * len(gk), 6), squeeze=False)
        for j, k in enumerate(gk):
            g = sorted([r for r in rs if r["key"] == k], key=lambda r: r["seed"])
            H = np.array([r["allocation"]["heads"] for r in g])
            Fm = np.array([[f // 64 for f in r["allocation"]["ffn"]] for r in g])
            for i, (M, nm) in enumerate(((H, "heads"), (Fm, "FFN chunks (×64)"))):
                ax = axes[i, j]
                im = ax.imshow(M.T, aspect="auto", cmap="viridis")
                for (a, b), v in np.ndenumerate(M.T):
                    ax.text(b, a, int(v), ha="center", va="center", color="w", fontsize=8)
                ax.set_xticks(range(len(g)), [f"s{r['seed']}" for r in g])
                ax.set_ylabel("layer")
                ax.set_title(f"{arm_label(k)}\n{nm}", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"{scale}_allocation_heatmaps.png"), dpi=120)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results/runs")
    ap.add_argument("--out", default="results/plots")
    a = ap.parse_args()
    rows = load(a.res)
    all_stats = {}
    md = []
    for scale in sorted({r["scale"] for r in rows}):
        t, st = table(rows, scale)
        all_stats[scale] = st
        md.append(f"### Scale {scale}\n\n{t}\n")
        comps = []
        mk = [k for k in st if k.startswith("grow_metric")]
        for m in mk:
            for b in ("grow_random", "grow_uniform", "scratch", "scratch_flops", "openelm"):
                if b in st:
                    p = paired(st, m, b)
                    if p:
                        comps.append(f"| {arm_label(m)} − {arm_label(b)} | {p['n']} | {p['mean']:+.4f} ± {p['std']:.4f} | " +
                                     ", ".join(f"{x:+.4f}" for x in p["diffs"]) + " |")
        for a1, b in (("grow_random", "grow_uniform"), ("grow_uniform", "scratch"), ("grow_uniform", "scratch_flops"),
                      ("openelm", "scratch")):
            if a1 in st and b in st:
                p = paired(st, a1, b)
                if p:
                    comps.append(f"| {arm_label(a1)} − {arm_label(b)} | {p['n']} | {p['mean']:+.4f} ± {p['std']:.4f} | " +
                                 ", ".join(f"{x:+.4f}" for x in p["diffs"]) + " |")
        if comps:
            md.append("Paired differences (same seed = same data order; negative = first arm better):\n\n"
                      "| comparison | n | mean ± std | per seed |\n|---|---|---|---|\n" + "\n".join(comps) + "\n")
        plots(rows, scale, a.out)
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(os.path.dirname(a.out.rstrip("/")), "stage2_tables.md"), "w").write("\n".join(md))
    json.dump(all_stats, open(os.path.join(os.path.dirname(a.out.rstrip("/")), "stage2_stats.json"), "w"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
