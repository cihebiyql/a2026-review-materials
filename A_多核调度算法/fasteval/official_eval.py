"""v3 求解器公共层：环境、官方评估封装、图视图、结构诊断。

设计纪律：
- 只 import 官方 code/ 原样模块，不 import v2/solver/baseline（保留独立可复现）；
- ATT 可用环境变量 A2026_ATT 覆盖（本机默认 C:/shumo_live/a_data）；
- 评估常数与官方 config.txt 一致（第二轮核验已逐项对齐）。
"""
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

_PKG = Path(__file__).resolve().parent.parent
ATT = Path(os.environ.get("A2026_OFFICIAL", str(_PKG / "official")))
CODE = ATT / "code"
DATA = ATT / "data"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from stub_multicore_cut_and_schedule import (  # noqa: E402
    derive_multicore_plan, _build_op_adjacency,
    _contract_excluded_copy_nodes, EXCLUDED_COPY_TYPES)

BW = 60
CAP = {"L1": 524288, "UB": 131072}
CROSS_WAIT_A = 1000
SAME_WAIT_A = 100
DELAY_B = 500
CACHE_CAP = 1048576
CACHE_BW = 250

L1_CAP, UB_CAP = 524288, 131072


def load_case(case):
    with open(DATA / f"{case}.json", encoding="utf-8") as f:
        return json.load(f)


def sc_makespan(case, sc_dir):
    for name in (f"{case}_sc.json", f"{case}.json"):
        p = Path(sc_dir) / name
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return json.load(f)["makespan"]
    raise FileNotFoundError(f"no singlecore baseline for {case} in {sc_dir}")


def ev_p1(graph, plan):
    from multicore_cut_evaluate_problem_1 import evaluate_scene_a
    t0 = time.perf_counter()
    r = evaluate_scene_a(graph, plan, bandwidth=BW, capacity=CAP,
                         cross_core_wait=CROSS_WAIT_A,
                         same_core_wait=SAME_WAIT_A)
    return r, time.perf_counter() - t0


def ev_p2(graph, plan):
    from multicore_cut_evaluate_problem_2 import evaluate_scene_b
    t0 = time.perf_counter()
    r = evaluate_scene_b(graph, plan, bandwidth=BW, capacity=CAP,
                         cross_core_copy_delay=DELAY_B)
    return r, time.perf_counter() - t0


def ev_p3(graph, plan):
    from multicore_cut_evaluate_problem_3 import evaluate_problem_3
    t0 = time.perf_counter()
    r = evaluate_problem_3(graph, plan, bandwidth=BW, capacity=CAP,
                           cross_core_copy_delay=DELAY_B,
                           cache_capacity_bytes=CACHE_CAP,
                           cache_bandwidth_bytes_per_cycle=CACHE_BW)
    return r, time.perf_counter() - t0


# ---------------- 图视图 ----------------

def op_dag(graph):
    preds, succs = _build_op_adjacency(graph)
    ids = sorted(o["id"] for o in graph["ops"]
                 if o.get("op") not in EXCLUDED_COPY_TYPES)
    cp, cs = _contract_excluded_copy_nodes(ids, succs)
    return ids, cp, cs


def cycles_map(graph):
    return {o["id"]: max(1, o.get("cycles", 1)) for o in graph["ops"]
            if o.get("op") not in EXCLUDED_COPY_TYPES}


def pipe_map(graph):
    return {o["id"]: o.get("pipe", "PIPE_V") for o in graph["ops"]
            if o.get("op") not in EXCLUDED_COPY_TYPES}


def tensor_views(graph):
    """op->生产张量 / op->消费张量 / 张量大小 / 张量位置。"""
    op_ids = {o["id"] for o in graph["ops"]}
    prod, cons = defaultdict(set), defaultdict(set)
    tsize = {t["id"]: t["size"] for t in graph["tensors"]}
    tpos = {t["id"]: t.get("pos", "UB") for t in graph["tensors"]}
    for e in graph["edges"]:
        s, t = e["source"], e["target"]
        if s in op_ids and t not in op_ids:
            prod[s].add(t)
        elif t in op_ids and s not in op_ids:
            cons[t].add(s)
    return prod, cons, tsize, tpos


def op_weight_edges(graph):
    """op-op 边带张量通信量（经张量中转）。返回 [(src, dst, bytes)]。"""
    prod, cons, tsize, tpos = tensor_views(graph)
    tprod = {}
    for o, ts in prod.items():
        for t in ts:
            tprod[t] = o
    edges = []
    for o, ts in cons.items():
        for t in ts:
            p = tprod.get(t)
            if p is not None and p != o:
                edges.append((p, o, tsize.get(t, 0)))
    return edges


def make_plan(op_lists, n_cores, core_of_list=None):
    """op_lists: 每个子图的 op 列表。core_of_list: 子图->核。"""
    n2s = {}
    for sg, nodes in enumerate(op_lists):
        for v in nodes:
            n2s[v] = sg
    if core_of_list is None:
        core_of_list = [sg % n_cores for sg in range(len(op_lists))]
    cores = [[] for _ in range(n_cores)]
    for sg in range(len(op_lists)):
        cores[core_of_list[sg]].append(sg)
    return {"node_to_subgraph": n2s, "core_schedules": cores}


def topo_check_plan(graph, plan):
    """快速合法性预检（依赖无环 + 同核序合法）。"""
    view = derive_multicore_plan(graph, plan)
    return view


def summarize(r, sc):
    mk = r["makespan"]
    dm = r["data_movement_bytes"]
    out = {"makespan": mk, "speedup": round(sc / mk, 3),
           "added_MB": round(dm["added_copy_bytes"] / 1e6, 2),
           "spill_MB": round(dm["spill_added_copy_bytes"] / 1e6, 2)}
    if "cache_stats" in r:
        out["hit_rate"] = round(r["cache_stats"]["hit_rate"], 4)
        out["hit_MB"] = round(r["cache_stats"]["hit_bytes"] / 1e6, 2)
        out["copy_in_MB"] = round(
            (r["cache_stats"]["hit_bytes"] + r["cache_stats"]["miss_bytes"])
            / 1e6, 2)
    return out
