# -*- coding: utf-8 -*-
"""GPU step2 对拍：CUDA Belady spill vs CPU 复刻。

node1（cuda12.1）：
  A2026_ATT=/data1/qlyu/a2026/att CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH=v2solver /data/qlyu/anaconda3/envs/cuda12.1/bin/python \
  gpu_step2_gate.py case_019 case_001 case_082 case_050
"""
import json
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "v2solver"))

ATT = os.environ.get("A2026_ATT",
                     r"C:/shumo_live/a_data")
if not os.path.isdir(ATT):
    ATT = "/data1/qlyu/a2026/att"
sys.path.insert(0, os.path.join(ATT, "code"))

from fast_eval_p1 import GraphCodec, TaskBuild, stage_step1, stage_step2
import stub_multicore_cut_and_schedule as stub
from structure_split import structure_aware_plan


def build_uses(tb, seqs):
    results = []
    for k in range(tb.n_tasks):
        lo, hi = int(tb.t_op_ptr[k]), int(tb.t_op_ptr[k + 1])
        tlo, thi = int(tb.t_ten_ptr[k]), int(tb.t_ten_ptr[k + 1])
        n = hi - lo
        n_ten = thi - tlo
        pairs = []
        for p, op in enumerate(seqs[k]):
            u = int(op)
            for e in range(tb.in_ptr[u], tb.in_ptr[u + 1]):
                pairs.append((int(tb.in_ten[e]) - tlo, p, u))
            for e in range(tb.out_ptr[u], tb.out_ptr[u + 1]):
                pairs.append((int(tb.out_ten[e]) - tlo, p, u))
        pairs.sort()
        uses_tid = []
        uses_step = []
        uses_op = []
        prev_key = (-1, -1)
        for (tid, st, op) in pairs:
            if (tid, st) != prev_key:
                prev_key = (tid, st)
                uses_tid.append(tid)
                uses_step.append(st)
                uses_op.append(op)
        uses_ptr = np.zeros(n_ten + 1, dtype=np.int64)
        for tid in uses_tid:
            uses_ptr[tid + 1] += 1
        uses_ptr = np.cumsum(uses_ptr)
        has_use = np.diff(uses_ptr) > 0
        lifecycle = (tb.ten_pos[tlo:thi] != 0) & has_use
        entries = []
        rank = 0
        for tid in range(n_ten):
            if lifecycle[tid]:
                up0, up1 = int(uses_ptr[tid]), int(uses_ptr[tid + 1])
                for u in range(up0, up1):
                    entries.append((int(uses_step[u]), rank, u - up0, tid))
                rank += 1
        entries.sort()
        uat_ptr = np.zeros(n + 1, dtype=np.int64)
        for ent in entries:
            uat_ptr[ent[0] + 1] += 1
        uat_ptr = np.cumsum(uat_ptr)
        results.append({
            "n": n, "n_ten": n_ten,
            "uses_ptr": uses_ptr,
            "uses_step": np.array(uses_step, dtype=np.int64),
            "uses_op": np.array(uses_op, dtype=np.int64),
            "uat_ptr": uat_ptr,
            "uat_tid": np.array([e[3] for e in entries], dtype=np.int64),
            "uat_idx": np.array([e[2] for e in entries], dtype=np.int64),
            "ten_is_l1": tb.ten_pos[tlo:thi] == 1,
            "ten_size": tb.ten_size[tlo:thi].copy(),
        })
    return results


def pack_uses(uses_list):
    B = len(uses_list)
    ten_base = np.zeros(B + 1, dtype=np.int64)
    for i, u in enumerate(uses_list):
        ten_base[i + 1] = ten_base[i] + u["n_ten"]
    all_ten = int(ten_base[-1])
    uses_ptr_f = np.zeros(all_ten + 1, dtype=np.int64)
    uses_step_f = []
    uses_op_f = []
    uat_tid_f = []
    uat_idx_f = []
    step_base = np.zeros(B + 1, dtype=np.int64)
    for i, u in enumerate(uses_list):
        step_base[i + 1] = step_base[i] + u["n"] + 1
    uat_ptr_f = np.zeros(int(step_base[-1]), dtype=np.int64)
    for i, u in enumerate(uses_list):
        tb_ = int(ten_base[i])
        nt = u["n_ten"]
        eb = len(uses_step_f)
        uses_ptr_f[tb_:tb_ + nt + 1] = u["uses_ptr"] + eb
        uses_step_f.extend(u["uses_step"].tolist())
        uses_op_f.extend(u["uses_op"].tolist())
        sb_ = int(step_base[i])
        uat_ptr_f[sb_:sb_ + u["n"] + 1] = u["uat_ptr"] + len(uat_tid_f)
        uat_tid_f.extend(u["uat_tid"].tolist())
        uat_idx_f.extend(u["uat_idx"].tolist())
    return (ten_base, step_base,
            uses_ptr_f, np.array(uses_step_f, dtype=np.int64),
            np.array(uses_op_f, dtype=np.int64),
            uat_ptr_f, np.array(uat_tid_f, dtype=np.int64),
            np.array(uat_idx_f, dtype=np.int64))


def run_case(case, seed, use_v2=False, cpc=1):
    from numba import cuda
    from gpu_step2 import step2_batch_kernel
    graph = json.load(open(os.path.join(ATT, "data", f"{case}.json")))
    if use_v2:
        plan, _ = structure_aware_plan(graph, 4, chunks_per_core=cpc)
    else:
        plan = stub.generate_multicore_plan(graph, num_cores=4, seed=seed,
                                            min_subgraph_size=50,
                                            max_subgraph_size=100)
    gc = GraphCodec(graph)
    pv = stub.derive_multicore_plan(graph, plan)
    tb = TaskBuild(gc, pv)
    seqs = stage_step1(tb)
    cpu_ps = stage_step2(tb, seqs)
    uses_list = build_uses(tb, seqs)
    B = len(uses_list)
    (ten_base, step_base, uses_ptr_f, uses_step_f, uses_op_f,
     uat_ptr_f, uat_tid_f, uat_idx_f) = pack_uses(uses_list)
    all_ten = int(ten_base[-1])
    max_ten = max(u["n_ten"] for u in uses_list)
    max_ops = max(u["n"] for u in uses_list)
    max_steps = max(u["n"] + 1 for u in uses_list)
    ps_cap = 4 * max_ten + 64
    act_cap = ps_cap
    ten_is_l1 = np.zeros(all_ten, dtype=np.int64)
    ten_size = np.zeros(all_ten, dtype=np.int64)
    for i, u in enumerate(uses_list):
        t = int(ten_base[i])
        ten_is_l1[t:t + u["n_ten"]] = u["ten_is_l1"]
        ten_size[t:t + u["n_ten"]] = u["ten_size"]
    op_base = np.zeros(B + 1, dtype=np.int64)
    for i, u in enumerate(uses_list):
        op_base[i + 1] = op_base[i] + u["n"]

    d = cuda.to_device
    d_op = d(op_base)
    d_ten_b = d(ten_base)
    d_step_b = d(step_base)
    d_uses_ptr = d(uses_ptr_f)
    d_uses_step = d(uses_step_f)
    d_uses_op = d(uses_op_f)
    d_uat_ptr = d(uat_ptr_f)
    d_uat_tid = d(uat_tid_f)
    d_uat_idx = d(uat_idx_f)
    d_l1 = d(ten_is_l1)
    d_sz = d(ten_size)
    d_alive = d(np.zeros(B * max_ten, dtype=np.bool_))
    d_nu = d(np.zeros(B * max_ten, dtype=np.int64))
    d_ui = d(np.zeros(B * max_ten, dtype=np.int64))
    d_rank = d(np.zeros(B * max_ten, dtype=np.int64))
    d_order = d(np.zeros(B * act_cap, dtype=np.int64))
    d_resid = d(np.zeros(B * 2, dtype=np.int64))
    d_cg = d(np.zeros(B * max_ten, dtype=np.int64))
    d_rel = d(np.zeros(B * max_ops, dtype=np.int64))
    d_sh = d(np.full(B * max_steps, -1, dtype=np.int64))
    d_st = d(np.full(B * max_steps, -1, dtype=np.int64))
    d_sn = d(np.full(B * ps_cap, -1, dtype=np.int64))
    d_stid = d(np.zeros(B * ps_cap, dtype=np.int64))
    d_sui = d(np.zeros(B * ps_cap, dtype=np.int64))
    d_pv = d(np.zeros(B * ps_cap, dtype=np.int64))
    d_pps = d(np.zeros(B * ps_cap, dtype=np.int64))
    d_pns = d(np.zeros(B * ps_cap, dtype=np.int64))
    d_ppo = d(np.zeros(B * ps_cap, dtype=np.int64))
    d_pno = d(np.zeros(B * ps_cap, dtype=np.int64))
    d_pui = d(np.zeros(B * ps_cap, dtype=np.int64))
    d_ptype = d(np.zeros(B * ps_cap, dtype=np.int64))
    d_psize = d(np.zeros(B * ps_cap, dtype=np.int64))
    d_pn = d(np.full(B, -999, dtype=np.int64))

    t0 = time.perf_counter()
    grid = (B + 127) // 128
    step2_batch_kernel[grid, 128](
        d_op, d_ten_b, d_step_b,
        d_uses_ptr, d_uses_step, d_uses_op,
        d_uat_ptr, d_uat_tid, d_uat_idx,
        d_l1, d_sz,
        np.int64(524288), np.int64(131072),
        d_alive, d_nu, d_ui, d_rank, d_order,
        d_resid, d_cg, d_rel,
        d_sh, d_st, d_sn, d_stid, d_sui,
        d_pv, d_pps, d_pns, d_ppo, d_pno,
        d_pui, d_ptype, d_psize, d_pn,
        np.int64(max_ops), np.int64(max_ten), np.int64(ps_cap),
        np.int64(act_cap))
    cuda.synchronize()
    t1 = time.perf_counter()

    ps_n_h = d_pn.copy_to_host()
    vic_h = d_pv.copy_to_host()

    n_bad = 0
    for k in range(B):
        cn = cpu_ps[k]["n_ps"]
        gn = int(ps_n_h[k])
        if cn != gn:
            n_bad += 1
            print(f"  MISMATCH task{k}: CPU={cn} GPU={gn}")
        elif cn > 0:
            for i in range(cn):
                cv = int(cpu_ps[k]["victim"][i])
                gv = int(vic_h[k * ps_cap + i])
                if cv != gv:
                    n_bad += 1
                    print(f"  VICTIM DIFF task{k} #{i}: CPU={cv} GPU={gv}")
    return n_bad, B, t1 - t0


def main():
    cases = sys.argv[1:] or ["case_019"]
    total = 0
    for case in cases:
        for seed in (0, 1):
            n_bad, B, t = run_case(case, seed)
            total += n_bad
            print(f"{case} stub{seed}: {B} tasks, bad={n_bad} "
                  f"gpu={t*1000:.1f}ms", flush=True)
        n_bad, B, t = run_case(case, 0, use_v2=True, cpc=1)
        total += n_bad
        print(f"{case} v2cpc1: {B} tasks, bad={n_bad} "
              f"gpu={t*1000:.1f}ms", flush=True)
    print("GPU_STEP2_GATE:", "PASS" if total == 0 else f"FAIL ({total})")
    sys.exit(0 if total == 0 else 1)


if __name__ == "__main__":
    main()
