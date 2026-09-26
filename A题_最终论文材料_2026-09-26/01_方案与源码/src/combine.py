# -*- coding: utf-8 -*-
"""N5 战役记账:逐例取 max(posthoc, refined, strand),汇总均值与增益分解。"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
POSTHOC = Path(r"C:/shumo_live/02_求解/A题_2026/a_lab/records/POSTHOC_BEST.json")


def combine(q=3, K=5):
    posthoc = json.load(open(POSTHOC))
    pool = {}
    src = {}
    for i in range(1, 101):
        case = f"case_{i:03d}"
        key = f"{case}|q{q}|N{K}"
        base = posthoc.get(key)
        if base is not None:
            pool[case] = base
            src[case] = "posthoc"
        fp = HERE / "refined" / f"{case}_q{q}_N{K}.json"
        if fp.exists():
            d = json.load(open(fp))
            if d.get("sp", 0) > pool.get(case, 0):
                pool[case] = d["sp"]
                src[case] = "refined"
        fp = HERE / "strand_n5" / f"{case}_q{q}_N{K}.json"
        if fp.exists():
            d = json.load(open(fp))
            if d.get("sp", 0) > pool.get(case, 0):
                pool[case] = d["sp"]
                src[case] = "strand"
        for extra, tag in (("refined2", "refine2"),
                           ("refined2_q2", "refine2"),
                           ("refined2_mega", "refine2"),
                           ("refined2_light", "refine2"),
                           ("refined3", "refine3"),
                           ("cross_best", "cross"),
                           ("audit_B_best", "auditB"),
                           ("bxcpu_best", "bxcpu"),
                           ("refined5", "r5"),
                           ("refined6", "greedy"),
                           ("refined7", "greedy2"),
                           ("refined7b", "greedy2"),
                           ("refined8", "prod")):
            fp = HERE / extra / f"{case}_q{q}_N{K}.json"
            if fp.exists():
                d = json.load(open(fp))
                if d.get("sp", 0) > pool.get(case, 0):
                    pool[case] = d["sp"]
                    src[case] = tag
    vals = list(pool.values())
    mean = sum(vals) / len(vals) if vals else 0
    gain = sum(v - posthoc.get(f"{c}|q{q}|N{K}", v) for c, v in pool.items())
    from collections import Counter
    print(f"q{q} N{K}: {len(vals)}例 池均值 {mean:.4f} | 相对posthoc总增量 "
          f"{gain:.1f} 点 (=均值+{gain/len(vals):.4f})")
    print("来源分布:", dict(Counter(src.values())))
    wins = sorted(((c, pool[c] - posthoc.get(f"{c}|q{q}|N{K}", pool[c]))
                   for c in pool), key=lambda x: -x[1])[:15]
    print("增益Top15:")
    for c, g in wins:
        if g > 0.001:
            print(f"  {c}: +{g:.3f} ({src[c]}) -> {pool[c]:.3f}")
    return pool


if __name__ == "__main__":
    q = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    combine(q=q)
