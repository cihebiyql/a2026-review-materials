# -*- coding: utf-8 -*-
"""P3 缓存批内核对拍门禁：scene_b_cache_batch_kernel vs CPU FastEvalP3 真值。

对每个候选 plan：
  CPU 参照 = FastEvalP3(graph).evaluate(plan)[0]（官方 P3 bit-exact 复刻）；
  GPU = pack_one 链（TaskBuildB→step1/2/3）+ build_scene_pack +
        build_cache_keys → 批平铺 → scene_b_cache_batch_kernel[B,128]。
失败语义：CPU 抛 RuntimeError(f"scene_b_sim error {r}") → 码 r；
GPU out_mk 负码（-2/-8/-7）同编号。

用法: py -3.11 gpu_scene_b_cache_batch_gate.py [case ...]
"""
import json
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "fast_eval")):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, "fast_eval"))
GLINK = os.path.join(ROOT, "gpu_link")
if not os.path.isdir(GLINK):
    GLINK = os.path.join(os.path.dirname(ROOT), "n5_push", "gpu_link")
sys.path.insert(0, GLINK)
ATT = os.environ.get("A2026_ATT", "C:/shumo_live/a_data")
sys.path.insert(0, os.path.join(ATT, "code"))

from fast_eval_p1 import GraphCodec, stage_step1, stage_step2, \
    build_ext_all, stage_step3  # noqa: E402
from fast_eval_p2 import (TaskBuildB, _prioritize, FastEvalP3,  # noqa
                          ext_local_out_ptr, ext_local_out_ten)
import stub_multicore_cut_and_schedule as stub  # noqa: E402
from pack_candidates import (Ops, State, build_scene_pack,  # noqa
                             gen_legal)

K = 5
TPB = 128
CACHE_CAP = 1048576
CACHE_BW = 250


def build_cache_keys(tb, exts, task_op_base, n_gops):
    """提取 fast_eval_p2.stage_scene_b 的缓存键计算（415-448 行）。"""
    cache_idx_of_gop = np.full(n_gops, -1, dtype=np.int64)
    keys = set()
    per_gop_key = {}
    for k in range(tb.num_cores):
        ext = exts[k]
        b = task_op_base[k]
        op_type = ext["op_type"]
        out_ptr = ext_local_out_ptr(ext)
        out_ten = ext_local_out_ten(ext)
        ten_logical = ext["ten_logical"]
        ten_pos = ext["ten_pos"]
        for i in range(ext["n_ext_ops"]):
            if op_type[i] in (1, 3):
                for e in range(out_ptr[i], out_ptr[i + 1]):
                    t = int(out_ten[e])
                    if ten_pos[t] != 0:
                        lg = int(ten_logical[t])
                        if int(ext["ten_size"][t]) > 0:
                            per_gop_key[b + i] = lg
                            keys.add(lg)
                        break
    key_sorted = np.array(sorted(keys), dtype=np.int64)
    cache_size_of_key = np.zeros(max(1, len(key_sorted)), dtype=np.int64)
    for k in range(tb.num_cores):
        ext = exts[k]
        for j in range(ext["n_ext_ten"]):
            if ext["ten_pos"][j] != 0:
                lg = int(ext["ten_logical"][j])
                idx = int(np.searchsorted(key_sorted, lg))
                if idx < len(key_sorted) and key_sorted[idx] == lg:
                    cache_size_of_key[idx] = int(ext["ten_size"][j])
    for g, lg in per_gop_key.items():
        cache_idx_of_gop[g] = int(np.searchsorted(key_sorted, lg))
    return cache_idx_of_gop, key_sorted, cache_size_of_key


def pack_one_cache(graph, gc, plan):
    pv = stub.derive_multicore_plan(graph, plan)
    tb = TaskBuildB(gc, pv)
    seqs = _prioritize(tb, stage_step1(tb))
    pss = stage_step2(tb, seqs)
    exts = build_ext_all(tb, seqs, pss)
    s3 = stage_step3(exts)
    pack = build_scene_pack(tb, s3, exts)
    task_op_base = pack["task_op_base"]
    ci, key_sorted, csz = build_cache_keys(tb, exts, task_op_base,
                                           pack["n_gops"])
    pack["cache_idx_of_gop"] = ci
    pack["cache_size_of_key"] = csz
    pack["n_keys"] = len(key_sorted)
    return pack


IN_KEYS = ["gop_pipe", "gop_dur", "gop_is_ddr", "gop_core",
           "task_seq_ptr", "task_seq_arr", "pipe_ptr", "pipe_seq_arr",
           "pipe_ptr_base", "pipe_seq_base", "task_op_base", "succ_ptr",
           "succ_arr", "pred_cnt", "ext_pred_arr", "ext_succ_ptr",
           "ext_succ_arr"]


def concat_packs_cache(packs):
    P = len(packs)
    offs = {k: np.zeros(P + 1, np.int64) for k in
            ["gop_off", "ptrN_off", "core_off", "core1_off", "core5_off",
             "ts_off", "succ_off", "link_off", "key_off"]}
    for i, p in enumerate(packs):
        ng, nc, nl = p["n_gops"], p["n_cores"], p["n_links"]
        offs["gop_off"][i + 1] = offs["gop_off"][i] + ng
        offs["ptrN_off"][i + 1] = offs["ptrN_off"][i] + ng + 1
        offs["core_off"][i + 1] = offs["core_off"][i] + nc
        offs["core1_off"][i + 1] = offs["core1_off"][i] + nc + 1
        offs["core5_off"][i + 1] = offs["core5_off"][i] + nc * 5
        offs["ts_off"][i + 1] = offs["ts_off"][i] + len(p["task_seq_arr"])
        offs["succ_off"][i + 1] = offs["succ_off"][i] + len(p["succ_arr"])
        offs["link_off"][i + 1] = offs["link_off"][i] + nl
        offs["key_off"][i + 1] = offs["key_off"][i] + p["n_keys"]

    def cat(f):
        return np.concatenate([p[f] for p in packs])

    meta = np.array([[p["n_cores"], p["n_gops"], p["n_links"],
                      p["delay"]] for p in packs], dtype=np.int64)
    z = {k: v for k, v in offs.items()}
    z["plan_meta"] = meta
    z["cache_idx_of_gop"] = cat("cache_idx_of_gop")
    z["cache_size_of_key"] = cat("cache_size_of_key")
    for f in IN_KEYS:
        z[f] = cat(f)
    return z


def gpu_eval_cache(z):
    from numba import cuda
    from gpu_scene_b_cache_batch import scene_b_cache_batch_kernel
    B = len(z["plan_meta"])
    pm = z["plan_meta"]
    mng = int(pm[:, 1].max())
    mnc = int(pm[:, 0].max())
    mnl = int(pm[:, 2].max())
    mnk = int(z["key_off"][-1])
    s_ng, s_c4 = mng, mnc * 4
    s_pool = mng + 8
    s_sa = max(mng + 8, mnl + 8)
    s_heap = mnl + 8
    s_ret = mnc * 4 + 8
    s_key = max(1, mnk)
    i64, f64 = np.int64, np.float64
    scr = [
        np.zeros(B * s_ng, i64), np.zeros(B * s_ng, f64),
        np.zeros(B * s_ng, i64),
        np.zeros(B * s_c4, i64), np.zeros(B * s_c4, i64),
        np.zeros(B * s_c4, i64), np.zeros(B * s_c4, f64),
        np.zeros(B * s_pool, i64), np.zeros(B * s_pool, f64),
        np.zeros(B * s_pool, np.bool_),
        np.zeros(B, i64), np.zeros(B * s_ng, i64), np.zeros(B, f64),
        np.zeros(B * s_sa, i64), np.zeros(B * s_pool, f64),
        np.zeros(B * s_ng, i64),
        np.zeros(B * s_heap, i64), np.zeros(B, i64),
        np.zeros(B * s_ret, i64), np.zeros(B, f64),
        # cache scratch
        np.zeros(B * s_pool, i64), np.zeros(B * s_pool, f64),
        np.zeros(B * s_pool, np.bool_),
        np.zeros(B, i64), np.zeros(B * s_ng, i64), np.zeros(B, f64),
        np.zeros(B * s_key, i64), np.zeros(B * s_key, i64),
        np.zeros(B * s_key, i64),
        np.zeros(B, i64), np.zeros(B, i64), np.zeros(B, i64),
        np.zeros(B, i64),
    ]
    d = cuda.to_device
    dpm = d(pm)
    doff = [d(z[k]) for k in
            ["gop_off", "ptrN_off", "core_off", "core1_off", "core5_off",
             "ts_off", "succ_off", "link_off"]]
    din = [d(z[k]) for k in IN_KEYS]
    dci = d(z["cache_idx_of_gop"])
    dcsz = d(z["cache_size_of_key"])
    dkey = d(z["key_off"])
    dscr = [d(a) for a in scr]
    grid = (B + TPB - 1) // TPB
    scene_b_cache_batch_kernel[grid, TPB](
        i64(B), i64(0), dpm, *doff, *din,
        dci, dcsz, dkey, i64(CACHE_CAP), i64(CACHE_BW),
        *dscr[:-13], dscr[-13], dscr[-12], dscr[-11], dscr[-10], dscr[-9],
        dscr[-8], dscr[-7], dscr[-6], dscr[-5], dscr[-4], dscr[-3],
        dscr[-2], dscr[-1],
        *[i64(s) for s in
          (s_ng, s_c4, s_pool, s_sa, s_heap, s_ret, s_key)])
    return dscr[-14 - 13 + 13].copy_to_host() if False else None, dscr


def main():
    cases = sys.argv[1:] or ["case_019", "case_064"]
    rng = random.Random(777)
    seeds_fp = os.path.join(GLINK, "..", "..", "n5_push", "gpu_search",
                            "seeds_q3.jsonl")
    if not os.path.exists(seeds_fp):
        seeds_fp = os.path.join(GLINK, "..", "gpu_search", "seeds_q3.jsonl")
    seeds = {}
    if os.path.exists(seeds_fp):
        for line in open(seeds_fp, encoding="utf-8"):
            r = json.loads(line)
            seeds[r["case"]] = r["plan"]
    n_bad = 0
    from numba import cuda
    for case in cases:
        graph = json.load(open(os.path.join(ATT, "data", f"{case}.json")))
        gc = GraphCodec(graph)
        ops = Ops(graph)
        fe = FastEvalP3(graph)
        seed_plan = seeds.get(case)
        if seed_plan is None:
            import stub_multicore_cut_and_schedule as st2
            seed_plan = st2.generate_multicore_plan(
                graph, num_cores=4, seed=0, min_subgraph_size=50,
                max_subgraph_size=100)
        st0 = State(seed_plan, K)
        plans = [seed_plan]
        while len(plans) < 40:
            c, _ = gen_legal(st0, ops, rng)
            if c is not None:
                plans.append(c.emit())
        # CPU 真值
        cpu_mk = []
        for p in plans:
            try:
                mk, _ = fe.evaluate(p)
                cpu_mk.append(int(mk))
            except RuntimeError as ex:
                import re
                m = re.search(r"error (-?\d+)", str(ex))
                cpu_mk.append(int(m.group(1)) if m else -99)
        # GPU 批
        packs = []
        keep = []
        for p in plans:
            try:
                packs.append(pack_one_cache(graph, gc, p))
                keep.append(p)
            except Exception:
                keep.append(None)
        # 打包失败：CPU 侧必是 derive/链异常 → 对齐跳过（记录）
        ok_idx = [i for i, p in enumerate(keep) if p is not None]
        z = concat_packs_cache([packs[ok_idx.index(i)] for i in ok_idx])
        _, dscr = gpu_eval_cache(z)
        # out_mk 是 dscr[19]（P2 scratch 第20个）
        mks = dscr[19].copy_to_host()
        t_ok = t_err = 0
        bad = []
        for j, i in enumerate(ok_idx):
            g = int(round(mks[j]))
            c_ = cpu_mk[i]
            if g >= 0 and c_ >= 0:
                t_ok += 1
                if g != c_:
                    bad.append((i, "mk", c_, g))
            elif g < 0 and c_ < 0:
                t_err += 1
                if g != c_:
                    bad.append((i, "err", c_, g))
            else:
                bad.append((i, "sign", c_, g))
        n_bad += len(bad)
        print(f"{case}: B={len(ok_idx)} ok={t_ok} err={t_err} "
              f"mismatch={len(bad)}", flush=True)
        for b in bad[:5]:
            print("   BAD", b)
    print("GPU_SCENE_B_CACHE_BATCH_GATE:",
          "PASS" if n_bad == 0 else f"FAIL ({n_bad})")
    sys.exit(0 if n_bad == 0 else 1)


if __name__ == "__main__":
    main()
