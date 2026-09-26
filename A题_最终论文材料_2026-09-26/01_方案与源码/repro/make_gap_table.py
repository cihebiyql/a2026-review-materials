# -*- coding: utf-8 -*-
"""任务2: 冠军池 vs 合规轨 Gap 对照表.

池侧: superlinear_analysis/{case}_q{Q}_mechA.json 的 sp (与 package_plans.py 同源同优先级).
合规侧: compliant_out/compliant_q*.jsonl — 仅取 N=5 记录, tag 非空跳过(消融/实验),
        同键 (case,Q) 取 mk 最小者.
输出 deliver_final/gap_table.csv: case,Q,pool_sp,compliant_sp,gap(池-合规),gap_pct
  gap_pct = 100 * gap / pool_sp
末尾三行汇总: 每问 平均池sp / 平均合规sp / 平均gap.
"""
import csv
import glob
import json
import os
import sys

from package_plans import BASE, QUESTIONS, find_pool_file, list_cases

COMPLIANT_GLOB = os.path.join(BASE, "compliant_out", "compliant_q*.jsonl")
OUT_CSV = os.path.join(BASE, "deliver_final", "gap_table.csv")


def load_pool_sp(cases):
    """(case,Q) -> 池 sp; 缺文件跳过."""
    pool = {}
    for q in QUESTIONS:
        for case in sorted(cases):
            p = find_pool_file(case, q)
            if p is None:
                continue
            d = json.load(open(p, encoding="utf-8"))
            pool[(case, q)] = d["sp"]
    return pool


def load_compliant():
    """(case,Q) -> (mk, sp); 只收 N=5 且 tag 为空/缺键 的记录, 同键取 mk 最小."""
    best = {}
    n_all = n_used = n_skip_tag = n_skip_n = 0
    for p in sorted(glob.glob(COMPLIANT_GLOB)):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n_all += 1
            if r.get("tag"):
                n_skip_tag += 1
                continue
            if r.get("N") != 5:
                n_skip_n += 1
                continue
            n_used += 1
            k = (r["case"], int(r["Q"]))
            if k not in best or r["mk"] < best[k][0]:
                best[k] = (r["mk"], r["sp"])
    stats = dict(n_all=n_all, n_used=n_used, n_skip_tag=n_skip_tag, n_skip_n=n_skip_n,
                 files=[os.path.basename(p) for p in sorted(glob.glob(COMPLIANT_GLOB))])
    return best, stats


def main():
    cases = list_cases()
    pool = load_pool_sp(cases)
    comp, cstats = load_compliant()

    rows = []
    for q in QUESTIONS:
        for case in sorted(cases):
            if (case, q) not in pool or (case, q) not in comp:
                continue
            psp, csp = pool[(case, q)], comp[(case, q)][1]
            gap = psp - csp
            gpct = 100.0 * gap / psp if psp else float("nan")
            rows.append([case, q, round(psp, 4), round(csp, 4),
                         round(gap, 4), round(gpct, 2)])

    summary = []
    for q in QUESTIONS:
        sub = [r for r in rows if r[1] == q]
        if not sub:
            continue
        mp = sum(r[2] for r in sub) / len(sub)
        mc = sum(r[3] for r in sub) / len(sub)
        mg = sum(r[4] for r in sub) / len(sub)
        summary.append(["AVG", q, round(mp, 4), round(mc, 4), round(mg, 4),
                        round(100.0 * mg / mp, 2)])

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case", "Q", "pool_sp", "compliant_sp", "gap", "gap_pct"])
        w.writerows(rows)
        w.writerows(summary)

    print("== make_gap_table 统计 ==")
    print(f"合规轨读取: {cstats['files']}")
    print(f"  记录总数 {cstats['n_all']}, N=5且tag空 {cstats['n_used']}, "
          f"tag非空跳过 {cstats['n_skip_tag']}, 非N5跳过 {cstats['n_skip_n']}")
    print(f"gap_table 行数(不含汇总): {len(rows)}")
    for q in QUESTIONS:
        n_pool = sum(1 for c in cases if (c, q) in pool)
        n_comp = sum(1 for c in cases if (c, q) in comp)
        n_row = sum(1 for r in rows if r[1] == q)
        neg = [r for r in rows if r[1] == q and r[4] < 0]
        print(f"q{q}: 池 {n_pool}/100, 合规 {n_comp}/100, 成行 {n_row}; "
              f"gap<0(合规反超池) {len(neg)} 例" +
              ("" if not neg else " 例:" + ",".join(r[0] for r in neg[:6])))
    print("== 三问汇总 (AVG 行) ==")
    for s in summary:
        print(f"  q{s[1]}: 平均池sp={s[2]}, 平均合规sp={s[3]}, 平均gap={s[4]} ({s[5]}%)")
    print(f"输出: {OUT_CSV}")


if __name__ == "__main__":
    sys.exit(main())
