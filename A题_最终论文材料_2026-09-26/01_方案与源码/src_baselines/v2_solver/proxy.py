"""v2 两级代理模型（场景A）。

- fast_proxy：段级 DAG + 事件模拟，段时长 = κ·Σcycles（κ 用单核基准
  makespan / 总 cycles 整图校准）。零官方计算，毫秒级，用于组合筛选与
  局部搜索内层。直接建模接力效应（依赖链串行 + 双等待周期）。
- official_proxy：官方 _build_scene_a_tasks（每子图 step1/2/3 精确预算
  local_makespan）+ 同一套事件模拟，整图比值校准（executor/step3 比）。

两级都输出同一结构 dict，方便 Spearman 对照。
"""
import os
import sys
from pathlib import Path

ATT = Path(os.environ.get(
    "A2026_ATT",
    r"D:/work/数模竞赛/中文题目_外层/中文题目/A题/"
    r"通用神经网络处理器下的多核调度问题  附件_解压"))
CODE = ATT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from stub_multicore_cut_and_schedule import derive_multicore_plan  # noqa: E402
from structure_split import cycles_map  # noqa: E402

SAME_WAIT = 100
CROSS_WAIT = 1000


def task_sim(durations, core_of, core_order, preds):
    """Task 级事件模拟：每核按 core_orders 序推进；Task 等全部前驱完成
    （执行器 task_release_time 语义），核间接续加 CROSS_WAIT、同核接续
    加 SAME_WAIT。返回 makespan。"""
    end = {}
    core_free = {c: 0.0 for c in core_order}
    core_idx = {c: 0 for c in core_order}
    pending = set(durations)
    while pending:
        progress = False
        for c, order in core_order.items():
            while core_idx[c] < len(order):
                t = order[core_idx[c]]
                if not all(p in end for p in preds.get(t, ())):
                    break
                start = core_free[c] + (SAME_WAIT if core_idx[c] > 0 else 0)
                for p in preds.get(t, ()):
                    if core_of[p] != c:
                        start = max(start, end[p] + CROSS_WAIT)
                end[t] = start + durations[t]
                core_free[c] = end[t]
                core_idx[c] += 1
                pending.discard(t)
                progress = True
        if not progress:
            raise RuntimeError("task_sim deadlock（同核依赖序非法）")
    return max(end.values())


def fast_proxy(graph, plan, kappa):
    """κ 校准快速代理。kappa = 单核makespan / 总cycles。"""
    view = derive_multicore_plan(graph, plan)
    cyc = cycles_map(graph)
    w_by_sg = {}
    for sg, nodes in view["nodes_by_subgraph"].items():
        w_by_sg[sg] = sum(cyc[v] for v in nodes)
    durations = {sg: kappa * w for sg, w in w_by_sg.items()}
    core_of = view["core_by_subgraph"]
    core_order = view["core_orders"]
    preds = {sg: set(ps) for sg, ps in view["subgraph_preds"].items()}
    mk = task_sim(durations, core_of, core_order, preds)
    return {"makespan": mk, "n_tasks": len(durations),
            "dep_edges": len(view["dependency_pairs"]),
            "max_task_w": max(w_by_sg.values())}


def official_proxy(graph, plan, cal_ratio=1.0):
    """官方预算代理：local_makespan × 整图校准比 cal_ratio。

    cal_ratio = 单核执行器 makespan / step3 整图模拟 makespan
    （第二轮核验：case_001 0.890 / case_050 0.835 / case_014 0.934）。
    逐任务 spill 通过运行时拦截 step2_spill_insertion 的返回获得
    （不改官方文件、不影响官方逻辑与输出）。
    """
    import multicore_cut_evaluate_problem_1 as mce
    from multicore_cut_evaluate_problem_1 import _build_scene_a_tasks
    bw, cap = 60, {"L1": 524288, "UB": 131072}
    captured = []
    orig_step2 = mce.step2_spill_insertion

    def wrapped(graph, seq, capacity=None):
        res = orig_step2(graph, seq, capacity=capacity)
        captured.append(res["spill_records"])
        return res

    mce.step2_spill_insertion = wrapped
    try:
        tasks, ct, traffic, view = _build_scene_a_tasks(graph, plan, bw, cap)
    finally:
        mce.step2_spill_insertion = orig_step2
    durations = {tid: cal_ratio * t["step3"]["makespan"] for tid, t in tasks.items()}
    core_of = view["core_by_subgraph"]
    core_order = view["core_orders"]
    preds = {sg: set(ps) for sg, ps in view["subgraph_preds"].items()}
    mk = task_sim(durations, core_of, core_order, preds)
    subgraph_ids = sorted(view["subgraph_ids"])  # _build 的迭代序
    per_task_spill = {}
    for i, tid in enumerate(subgraph_ids):
        recs = captured[i] if i < len(captured) else []
        per_task_spill[tid] = sum(
            r["size"] * (1 + int(r.get("spill_out_copies_data", 0)))
            for r in recs) / 1e6
    return {"makespan": mk, "n_tasks": len(durations),
            "dep_edges": len(view["dependency_pairs"]),
            "spill_MB": traffic["spill_added_copy_bytes"] / 1e6,
            "added_MB": traffic["added_copy_bytes"] / 1e6,
            "per_task_spill_MB": per_task_spill}
