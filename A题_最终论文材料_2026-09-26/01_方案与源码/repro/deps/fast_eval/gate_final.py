# -*- coding: utf-8 -*-
"""最终门禁：FastEvalP1.evaluate vs 官方 evaluate_scene_a 端到端 bit-exact 对拍
+ 性能基准。

对拍项：makespan / added / spill / partition / scheduled / original bytes /
cross_task_traffic / num_cores。
方案族：stub(2 seeds) × N∈{2,4} + v2 structure_aware_plan(cpc 1/2)。
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "v2_solver"))
sys.path.insert(0, r"C:/shumo_live/a_data/code")

from fast_eval_p1 import FastEvalP1  # noqa: E402
from multicore_cut_evaluate_problem_1 import evaluate_scene_a  # noqa: E402
import stub_multicore_cut_and_schedule as stub  # noqa: E402

CAP = {"L1": 524288, "UB": 131072}


def one_plan(graph, plan, fe, tag, n_fail):
    t0 = time.perf_counter()
    r = evaluate_scene_a(graph, plan, bandwidth=60, capacity=CAP,
                         cross_core_wait=1000, same_core_wait=100)
    t_off = time.perf_counter() - t0
    t0 = time.perf_counter()
    mk, info = fe.evaluate(plan)
    t_mine = time.perf_counter() - t0
    dmo = r["data_movement_bytes"]
    checks = [
        ("makespan", mk, r["makespan"]),
        ("added", info["added_copy_bytes"], dmo["added_copy_bytes"]),
        ("spill", info["spill_added_copy_bytes"],
         dmo["spill_added_copy_bytes"]),
        ("partition", info["partition_added_copy_bytes"],
         dmo["partition_added_copy_bytes"]),
        ("scheduled", info["scheduled_copy_bytes"],
         dmo["scheduled_copy_bytes"]),
        ("original", info["original_copy_bytes"],
         dmo["original_graph_copy_bytes"]),
        ("cross", info["cross_task_traffic"], r["cross_task_traffic"]),
        ("ncores", info["num_cores"], r["num_cores"]),
    ]
    ok = True
    for name, a, b in checks:
        if a != b:
            ok = False
            n_fail[0] += 1
            print(f"  [FAIL {name}] {tag}: mine={a} official={b}")
    print(f"  {tag}: mk={mk} {'OK' if ok else 'FAIL'} "
          f"official={t_off:.2f}s mine={t_mine:.3f}s "
          f"({t_off / max(t_mine, 1e-9):.0f}x)")
    return ok


def main():
    from structure_split import structure_aware_plan
    data = Path(r"C:/shumo_live/a_data/data")
    cases = sys.argv[1:] or ["case_019", "case_001", "case_082",
                             "case_050", "case_005"]
    n_fail = [0]
    n_plan = 0
    for case in cases:
        graph = json.load(open(data / f"{case}.json", encoding="utf-8"))
        fe = FastEvalP1(graph)
        print(f"== {case} (ops={len(graph['ops'])})")
        for seed in (0, 1):
            plan = stub.generate_multicore_plan(
                graph, num_cores=4, seed=seed, min_subgraph_size=50,
                max_subgraph_size=100)
            one_plan(graph, plan, fe, f"stub{seed}", n_fail)
            n_plan += 1
        for cpc in (1, 2):
            plan, _ = structure_aware_plan(graph, 4, chunks_per_core=cpc)
            one_plan(graph, plan, fe, f"v2cpc{cpc}", n_fail)
            n_plan += 1
    print(f"GATE_FINAL: {'PASS' if n_fail[0] == 0 else 'FAIL'} "
          f"({n_plan} plans, {n_fail[0]} mismatches)")
    sys.exit(0 if n_fail[0] == 0 else 1)


if __name__ == "__main__":
    main()
