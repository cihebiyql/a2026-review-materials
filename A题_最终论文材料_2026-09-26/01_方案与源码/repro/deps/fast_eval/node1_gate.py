# -*- coding: utf-8 -*-
"""node1（numba 0.60 / numpy 1.26）兼容性 + 正确性 + 性能门禁。

用法（node1）：
  cd /data1/qlyu/tmp_shumo/fast_eval
  A2026_ATT=/data1/qlyu/a2026/att /data/qlyu/anaconda3/bin/python \
      node1_gate.py case_001 case_082 [case_014]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ATT = Path(__import__("os").environ.get("A2026_ATT",
                                        r"C:/shumo_live/a_data"))
sys.path.insert(0, str(ATT / "code"))

from fast_eval_p1 import FastEvalP1  # noqa: E402
from multicore_cut_evaluate_problem_1 import evaluate_scene_a  # noqa: E402
import stub_multicore_cut_and_schedule as stub  # noqa: E402

CAP = {"L1": 524288, "UB": 131072}
DATA = ATT / "data"


def main():
    cases = sys.argv[1:] or ["case_001", "case_019"]
    n_fail = 0
    for case in cases:
        graph = json.load(open(DATA / f"{case}.json", encoding="utf-8"))
        fe = FastEvalP1(graph)
        print(f"== {case} ops={len(graph['ops'])}", flush=True)
        for seed in (0, 1):
            plan = stub.generate_multicore_plan(
                graph, num_cores=4, seed=seed, min_subgraph_size=50,
                max_subgraph_size=100)
            t0 = time.perf_counter()
            r = evaluate_scene_a(graph, plan, bandwidth=60, capacity=CAP,
                                 cross_core_wait=1000, same_core_wait=100)
            t_off = time.perf_counter() - t0
            t0 = time.perf_counter()
            mk, info = fe.evaluate(plan)
            t_mine = time.perf_counter() - t0
            dmo = r["data_movement_bytes"]
            ok = (mk == r["makespan"]
                  and info["added_copy_bytes"] == dmo["added_copy_bytes"]
                  and info["spill_added_copy_bytes"]
                  == dmo["spill_added_copy_bytes"]
                  and info["cross_task_traffic"] == r["cross_task_traffic"])
            if not ok:
                n_fail += 1
            print(f"  stub{seed}: mk={mk} official={r['makespan']} "
                  f"{'OK' if ok else 'FAIL'} off={t_off:.2f}s "
                  f"mine={t_mine:.3f}s ({t_off / max(t_mine, 1e-9):.0f}x)",
                  flush=True)
    print("NODE1_GATE:", "PASS" if n_fail == 0 else f"FAIL ({n_fail})")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
