"""Stage 1 analysis: oracle reliability and Spearman(metric, oracle) per checkpoint and pooled.

Oracle value of a candidate = (control val loss - candidate val loss) / added params (per seed);
the mean over seeds is the reference ranking.
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy.stats import spearmanr

from growth.operators import FFN_CHUNK

D = {"S": 384}


def params(u, d=384, hd=64):
    return 2 * d * FFN_CHUNK if u.startswith("ffn") else 4 * d * hd


def rho(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).correlation)


def random_baseline(n, trials=20000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.arange(n)
    r = np.array([spearmanr(x, rng.permutation(n)).correlation for _ in range(trials)])
    return {"mean": float(r.mean()), "std": float(r.std()), "p95": float(np.quantile(r, 0.95))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/oracle")
    ap.add_argument("--ckpts", default="0.3,0.6")
    ap.add_argument("--prefix", default="oracle", help="oracle file prefix, e.g. oracle or oracle400")
    ap.add_argument("--at", default=None, help="use the intermediate val loss at this many steps (val_at)")
    a = ap.parse_args()
    out = {"per_ckpt": {}, "pooled": {}}
    pooled_oracle, pooled_metric = [], {}
    pooled_cands = []
    lines = []
    for ck in a.ckpts.split(","):
        seeds = sorted(glob.glob(os.path.join(a.dir, f"{a.prefix}_{ck}_seed*.json")))
        runs = [json.load(open(f))["runs"] for f in seeds]
        if a.at:
            runs = [{c: {"val_loss": v["val_at"][a.at]} for c, v in r.items()} for r in runs]
        cands = [c for c in runs[0] if c != "control" and all(c in r for r in runs)]
        per_seed = []
        for r in runs:
            per_seed.append(np.array([(r["control"]["val_loss"] - r[c]["val_loss"]) / params(c) for c in cands]))
        raw = [np.array([r["control"]["val_loss"] - r[c]["val_loss"] for c in cands]) for r in runs]
        oracle = np.mean(per_seed, 0)
        rel = rho(per_seed[0], per_seed[1]) if len(per_seed) >= 2 else float("nan")
        ctrl = [r["control"]["val_loss"] for r in runs]
        m = json.load(open(os.path.join(a.dir, f"metrics_{ck}.json")))["metrics"]
        entry = {"candidates": cands, "oracle_per_param": oracle.tolist(), "oracle_raw_per_seed": [x.tolist() for x in raw],
                 "seed_reliability_spearman": rel,
                 "seed_reliability_pearson": float(np.corrcoef(per_seed[0], per_seed[1])[0, 1]) if len(per_seed) >= 2 else None,
                 "control_losses": ctrl, "metrics": {}}
        fidx = [i for i, c in enumerate(cands) if c.startswith("ffn")]
        hidx = [i for i, c in enumerate(cands) if c.startswith("head")]
        for name, v in m.items():
            sc = v["score"]
            have = [i for i, c in enumerate(cands) if c in sc]
            ms = np.array([sc[cands[i]] for i in have])
            orc = oracle[have]
            reps = v["repeats"]
            self_rel = rho([reps[0][cands[i]] for i in have], [reps[1][cands[i]] for i in have]) if len(reps) > 1 else None
            entry["metrics"][name] = {
                "spearman_all": rho(ms, orc) if len(have) == len(cands) else float("nan"),
                "spearman_ffn": rho(np.array([sc[cands[i]] for i in fidx]), oracle[fidx]),
                "spearman_head": rho(np.array([sc[cands[i]] for i in hidx]), oracle[hidx]) if all(cands[i] in sc for i in hidx) else float("nan"),
                "spearman_seed0": rho(ms, per_seed[0][have]), "spearman_seed1": rho(ms, per_seed[1][have]) if len(per_seed) > 1 else None,
                "self_reliability": self_rel,
                "seconds_per_eval": float(np.median(v["seconds"])),
                "scores": [sc.get(c) for c in cands],
            }
            pooled_metric.setdefault(name, []).extend([sc.get(c, np.nan) for c in cands])
        pooled_oracle.extend(oracle.tolist())
        pooled_cands.extend([f"{ck}/{c}" for c in cands])
        out["per_ckpt"][ck] = entry
    po = np.array(pooled_oracle)
    kinds = np.array([c.split("/")[1].split(":")[0] for c in pooled_cands])
    for name, vals in pooled_metric.items():
        v = np.array(vals, dtype=float)
        ok = ~np.isnan(v)
        out["pooled"][name] = {
            "spearman_all": rho(v, po) if ok.all() else float("nan"),
            "spearman_ffn": rho(v[kinds == "ffn"], po[kinds == "ffn"]),
            "spearman_head": rho(v[kinds == "head"], po[kinds == "head"]) if ok[kinds == "head"].all() else float("nan"),
            "mean_per_ckpt_all": float(np.nanmean([out["per_ckpt"][ck]["metrics"][name]["spearman_all"] for ck in out["per_ckpt"]])),
        }
    n = len(out["per_ckpt"][a.ckpts.split(",")[0]]["candidates"])
    out["random_baseline"] = {"n12": random_baseline(n), "n24": random_baseline(len(po)), "n6": random_baseline(n // 2),
                              "n12_ffn_pooled": random_baseline(int((kinds == "ffn").sum()))}
    # ranking for Stage-2 selection: pooled Spearman over all candidates (metrics with head scores only)
    elig = {k: v["spearman_all"] for k, v in out["pooled"].items() if not np.isnan(v["spearman_all"])}
    out["ranking"] = sorted(elig, key=lambda k: -elig[k])
    json.dump(out, open(os.path.join(a.dir, f"stage1_summary_{a.prefix}{'_at' + a.at if a.at else ''}.json"), "w"), indent=1)
    # markdown table
    cks = list(out["per_ckpt"])
    hdr = "| metric | " + " | ".join(f"ρ all @{c}" for c in cks) + " | ρ all pooled | ρ FFN pooled | ρ heads pooled | self-rel. | s/eval |"
    lines.append(hdr)
    lines.append("|" + "---|" * (hdr.count("|") - 1))
    for name in pooled_metric:
        pe = out["pooled"][name]
        per = [out["per_ckpt"][c]["metrics"][name]["spearman_all"] for c in cks]
        sr = np.nanmean([out["per_ckpt"][c]["metrics"][name]["self_reliability"] or np.nan for c in cks])
        sec = np.mean([out["per_ckpt"][c]["metrics"][name]["seconds_per_eval"] for c in cks])
        f = lambda x: "n/a" if x is None or np.isnan(x) else f"{x:+.2f}"
        lines.append(f"| {name} | " + " | ".join(f(x) for x in per) +
                     f" | {f(pe['spearman_all'])} | {f(pe['spearman_ffn'])} | {f(pe['spearman_head'])} | {f(sr)} | {sec:.2f} |")
    rb = out["random_baseline"]
    lines.append(f"| random | " + " | ".join(f"0 ± {rb['n12']['std']:.2f}" for _ in cks) +
                 f" | 0 ± {rb['n24']['std']:.2f} (p95 {rb['n24']['p95']:.2f}) | 0 ± {rb['n12_ffn_pooled']['std']:.2f} | | | 0 |")
    lines.append("")
    lines.append("Oracle seed-to-seed reliability (Spearman / Pearson of per-param oracle values): " + ", ".join(
        f"@{c}: {out['per_ckpt'][c]['seed_reliability_spearman'] or float('nan'):+.2f} / {out['per_ckpt'][c]['seed_reliability_pearson'] or float('nan'):+.2f}" for c in cks))
    open(os.path.join(a.dir, f"stage1_table_{a.prefix}{'_at' + a.at if a.at else ''}.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("ranking:", out["ranking"])


if __name__ == "__main__":
    main()
