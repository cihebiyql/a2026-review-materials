# -*- coding: utf-8 -*-
"""q2/q3 池互评:方案跨问题通用,把两池最优方案在对方问题下评估。"""
import sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/shumo_live/02_求解/A题_2026/a_lab/solvers/exp_strand")
sys.path.insert(0, r"C:/shumo_live/02_求解/A题_2026/fast_eval")
sys.path.insert(0, r"C:/shumo_live/a_data/code")
from common import load_case, ev_p2, ev_p3

HERE = __import__("pathlib").Path(__file__).parent
(HERE/"cross_best").mkdir(exist_ok=True)
pools = {q: json.load(open(HERE / f"POOL_q{q}_N5.json")) for q in (2, 3)}
# 找到每例每池当前最优方案文件
def best_plan_file(case, q):
    for d, nm in ((HERE/"refined2_q2", f"{case}_q{q}_N5.json"),
                  (HERE/"refined2", f"{case}_q{q}_N5.json"),
                  (HERE/"refined2_light", f"{case}_q{q}_N5.json"),
                  (HERE/"refined2_mega", f"{case}_q{q}_N5.json"),
                  (HERE/"refined", f"{case}_q{q}_N5.json"),
                  (HERE/"strand_n5", f"{case}_q{q}_N5.json")):
        if (d/nm).exists():
            d_ = json.load(open(d/nm))
            if d_.get("plan"): return d_["plan"], d_.get("sp", 0)
    import os
    arch = os.environ.get("A2026_ARCHIVE", r"C:/shumo_live/02_求解/A题_2026/a_lab/registry/plan_archive/plans")
    for tag in ("v3_main","v2_cpc"):
        fp = __import__("pathlib").Path(arch)/f"{case}_q{q}_N5_{tag}.json"
        if fp.exists():
            d_ = json.load(open(fp))
            return {"node_to_subgraph": d_["node_to_subgraph"], "core_schedules": d_["core_schedules"]}, None
    return None, None

scdir = r"C:/shumo_live/02_求解/A题_2026/results/singlecore"
out = {2: {}, 3: {}}
for i in range(1, 101):
    case = f"case_{i:03d}"
    sc = json.load(open(f"{scdir}/{case}_sc.json"))["makespan"]
    g = load_case(case)
    plans = {}
    for q in (2,3):
        p, sp = best_plan_file(case, q)
        if p: plans[q] = p
    for q_eval in (2,3):
        ev = ev_p2 if q_eval==2 else ev_p3
        best = pools[q_eval].get(case, 0); best_plan=None
        for q_src, p in plans.items():
            if q_src == q_eval: continue
            try:
                r, _ = ev(g, p)
                sp = sc/r["makespan"]
                if sp > best: best, best_plan = sp, p
            except Exception:
                pass
        out[q_eval][case] = best if best else pools[q_eval].get(case, None)
        if best_plan is not None:
            import json as _j
            _j.dump({"plan": best_plan, "sp": out[q_eval][case]},
                    open(HERE/"cross_best"/f"{case}_q{q_eval}_N5.json", "w"))
for q in (2,3):
    vals = [v for v in out[q].values() if v]
    print(f"q{q} 互评后均值: {sum(vals)/len(vals):.4f} (原 {sum(pools[q].values())/len(pools[q]):.4f})")
json.dump(out[2], open(HERE/"POOL_q2_N5_cross.json","w"))
json.dump(out[3], open(HERE/"POOL_q3_N5_cross.json","w"))
