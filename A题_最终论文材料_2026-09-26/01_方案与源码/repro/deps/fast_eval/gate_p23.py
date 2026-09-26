# -*- coding: utf-8 -*-
"""P2/P3 门禁：FastEvalP2 / FastEvalP3 vs 官方端到端 bit-exact 对拍。

对拍项：makespan / added / spill / partition / cross_task_traffic /
P3 缓存统计（hits/misses/hit_bytes/hit_rate）。
方案族：stub(2 seeds) + v2 structure_aware_plan(cpc 1/2)。
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "v2_solver"))
sys.path.insert(0, r"C:/shumo_live/a_data/code")

from fast_eval_p2 import FastEvalP2, FastEvalP3  # noqa: E402
from multicore_cut_evaluate_problem_2 import evaluate_scene_b  # noqa: E402
from multicore_cut_evaluate_problem_3 import evaluate_problem_3  # noqa: E402
import stub_multicore_cut_and_schedule as stub  # noqa: E402

CAP = {"L1": 524288, "UB": 131072}


def one(graph, plan, tag, n_fail):
    t0 = time.perf_counter()
    r2 = evaluate_scene_b(graph, plan, bandwidth=60, capacity=CAP,
                          cross_core_copy_delay=500)
    r3 = evaluate_problem_3(graph, plan, bandwidth=60, capacity=CAP,
                            cross_core_copy_delay=500,
                            cache_capacity_bytes=1048576,
                            cache_bandwidth_bytes_per_cycle=250)
    t_off = time.perf_counter() - t0
    t0 = time.perf_counter()
    mk2, i2 = FastEvalP2(graph).evaluate(plan)
    mk3, i3 = FastEvalP3(graph).evaluate(plan)
    t_mine = time.perf_counter() - t0
    ok = True
    d2 = r2["data_movement_bytes"]
    for name, a, b in [
            ("P2 mk", mk2, r2["makespan"]),
            ("P2 added", i2["added_copy_bytes"], d2["added_copy_bytes"]),
            ("P2 spill", i2["spill_added_copy_bytes"],
             d2["spill_added_copy_bytes"]),
            ("P2 cross", i2["cross_task_traffic"], r2["cross_task_traffic"]),
            ("P3 mk", mk3, r3["makespan"]),
            ("P3 hits", i3["cache_stats"]["hits"], r3["cache_stats"]["hits"]),
            ("P3 misses", i3["cache_stats"]["misses"],
             r3["cache_stats"]["copy_in_misses"]),
            ("P3 hb", i3["cache_stats"]["hit_bytes"],
             r3["cache_stats"]["hit_bytes"]),
            ("P3 rate", round(i3["cache_stats"]["hit_rate"], 6),
             round(r3["cache_stats"]["hit_rate"], 6))]:
        if a != b:
            ok = False
            n_fail[0] += 1
            print(f"  [FAIL {name}] {tag}: mine={a} official={b}")
    print(f"  {tag}: P2={mk2} P3={mk3} {'OK' if ok else 'FAIL'} "
          f"official={t_off:.2f}s mine={t_mine:.3f}s "
          f"({t_off / max(t_mine, 1e-9):.0f}x)")


def main():
    from structure_split import structure_aware_plan
    data = Path(r"C:/shumo_live/a_data/data")
    cases = sys.argv[1:] or ["case_019", "case_001", "case_082",
                             "case_050", "case_005"]
    n_fail = [0]
    for case in cases:
        graph = json.load(open(data / f"{case}.json", encoding="utf-8"))
        print(f"== {case} (ops={len(graph['ops'])})", flush=True)
        for seed in (0, 1):
            plan = stub.generate_multicore_plan(
                graph, num_cores=4, seed=seed, min_subgraph_size=50,
                max_subgraph_size=100)
            one(graph, plan, f"stub{seed}", n_fail)
        for cpc in (1, 2):
            plan, _ = structure_aware_plan(graph, 4, chunks_per_core=cpc)
            one(graph, plan, f"v2cpc{cpc}", n_fail)
    print(f"GATE_P23: {'PASS' if n_fail[0] == 0 else 'FAIL'} "
          f"({n_fail[0]} mismatches)")
    sys.exit(0 if n_fail[0] == 0 else 1)


if __name__ == "__main__":
    main()
