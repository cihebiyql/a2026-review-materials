# -*- coding: utf-8 -*-
"""任务1: 附录B逐用例方案打包.

从冠军池 superlinear_analysis/{case}_q{Q}_mechA.json (备选 {case}_q{Q}_N5_mechA.json)
提取 plan 的两个附录B顶层字段 node_to_subgraph / core_schedules, 原样写出
deliver_final/plans/{case}_q{Q}_N5.json, 并生成 manifest.csv (case,Q,mk,sp,source,plan_file).

自检(与官方约束一致): 每方案 node_to_subgraph 覆盖该用例全部非COPY算子且键无重复
(仅查键唯一性与数量, 不对比图结构).
本地缺失的 (case,Q) 跳过并记录清单: stdout + deliver_final/missing_plans_report.json.
"""
import csv
import glob
import json
import os
import re
import sys

BASE = os.path.dirname(os.path.abspath(__file__))  # .../n5_push
POOL_DIR = os.path.join(BASE, "superlinear_analysis")
DATA_DIR = r"C:/shumo_live/a_data/data"  # 官方用例图
OUT_DIR = os.path.join(BASE, "deliver_final", "plans")
QUESTIONS = (1, 2, 3)


def list_cases():
    """官方 100 用例清单 + 每例非COPY算子数(用于数量自检)."""
    cases = {}
    for p in sorted(glob.glob(os.path.join(DATA_DIR, "case_*.json"))):
        m = re.match(r"(case_\d{3})\.json$", os.path.basename(p))
        if not m:
            continue
        d = json.load(open(p, encoding="utf-8"))
        n_noncopy = sum(1 for o in d["ops"] if "COPY" not in o["op"])
        cases[m.group(1)] = n_noncopy
    return cases


def find_pool_file(case, q):
    """主名 {case}_q{Q}_mechA.json 优先, 备选 {case}_q{Q}_N5_mechA.json."""
    for name in (f"{case}_q{q}_mechA.json", f"{case}_q{q}_N5_mechA.json"):
        p = os.path.join(POOL_DIR, name)
        if os.path.exists(p):
            return p
    return None


def self_check(n2s, expected_noncopy):
    """返回错误列表; 空 = 通过. 键唯一(JSON dict 天然保证, 仍显式断言) + 数量一致."""
    errs = []
    if not isinstance(n2s, dict):
        return [f"node_to_subgraph 非 dict: {type(n2s).__name__}"]
    keys = list(n2s.keys())
    if len(keys) != len(set(keys)):
        errs.append(f"键有重复: {len(keys)}键/{len(set(keys))}唯一")
    if len(keys) != expected_noncopy:
        errs.append(f"数量不符: 覆盖{len(keys)} != 非COPY算子{expected_noncopy}")
    return errs


def main():
    cases = list_cases()
    if len(cases) != 100:
        print(f"[警告] 官方用例数 {len(cases)} != 100")
    os.makedirs(OUT_DIR, exist_ok=True)

    rows, missing, check_fail, warn_cs = [], {q: [] for q in QUESTIONS}, [], []
    for q in QUESTIONS:
        for case in sorted(cases):
            src = find_pool_file(case, q)
            if src is None:
                missing[q].append(case)
                continue
            try:
                d = json.load(open(src, encoding="utf-8"))
                plan = d["plan"]
                n2s = plan["node_to_subgraph"]
                cs = plan["core_schedules"]
                mk, sp = d["mk"], d["sp"]
                source = d["source"]
            except Exception as e:  # 结构损坏 → 记为自检失败并跳过
                check_fail.append((case, q, os.path.basename(src), f"结构异常: {e}"))
                continue
            errs = self_check(n2s, cases[case])
            if errs:
                check_fail.append((case, q, os.path.basename(src), "; ".join(errs)))
                continue
            if not isinstance(cs, list) or len(cs) != 5:
                warn_cs.append((case, q, os.path.basename(src),
                                len(cs) if isinstance(cs, list) else type(cs).__name__))
            out_name = f"{case}_q{q}_N5.json"
            with open(os.path.join(OUT_DIR, out_name), "w", encoding="utf-8") as f:
                json.dump({"node_to_subgraph": n2s, "core_schedules": cs},
                          f, ensure_ascii=False, separators=(",", ":"))
            rows.append([case, q, mk, sp, source, out_name])

    with open(os.path.join(OUT_DIR, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case", "Q", "mk", "sp", "source", "plan_file"])
        w.writerows(rows)

    report = {"missing": {f"q{q}": missing[q] for q in QUESTIONS},
              "self_check_fail": check_fail}
    with open(os.path.join(BASE, "deliver_final", "missing_plans_report.json"),
              "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"== package_plans 统计 ==")
    print(f"manifest 行数: {len(rows)}")
    for q in QUESTIONS:
        print(f"q{q}: 打包 {sum(1 for r in rows if r[1] == q)}/100, "
              f"本地缺失 {len(missing[q])}" +
              ("" if not missing[q] else " -> " + " ".join(missing[q])))
    if warn_cs:
        print(f"[软警告] core_schedules 非5核 {len(warn_cs)} 例(不拦截): {warn_cs[:5]}")
    else:
        print("core_schedules 均为5核列表")
    if check_fail:
        print(f"[自检失败] {len(check_fail)} 例(已跳过):")
        for c in check_fail:
            print("   ", c)
    else:
        print("自检全部通过: node_to_subgraph 键唯一且覆盖全部非COPY算子")
    print(f"输出目录: {OUT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
