# -*- coding: utf-8 -*-
"""批内核对拍门禁：scene_b_batch_kernel[B,TPB] vs 已验证的逐 plan 结果。

基准 = gpu_link/node1_out/{case}_gpu_results.jsonl（旧单 plan 内核
[1,1] 逐个发射，800/800 已与 CPU FastEvalP2 bit-exact）。
失败码映射（旧→新）：-1→-2（无事件死锁）、-2→-8（时间不推进）、
-3→-7（迭代上限），见 gpu_scene_b_batch.py 头注。

用法: py -3.11 gpu_scene_b_batch_gate.py [case ...]
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
GL = os.path.join(HERE, "..", "n5_push", "gpu_link")
WORK = os.path.join(GL, "work")
REF = os.path.join(GL, "node1_out")

MAP = {-1: -2, -2: -8, -3: -7}
TPB = 128


def alloc_scratch(B, mng, mnc, mnl):
    s_ng = int(mng)
    s_c4 = int(mnc) * 4
    s_pool = int(mng) + 8
    s_sa = max(int(mng) + 8, int(mnl) + 8)
    s_heap = int(mnl) + 8
    s_ret = int(mnc) * 4 + 8
    z = lambda n, dt: np.zeros(n, dtype=dt)  # noqa: E731
    scr = dict(
        st_status=z(B * s_ng, np.int64), st_end=z(B * s_ng, np.float64),
        st_pred=z(B * s_ng, np.int64),
        st_pcur=z(B * s_c4, np.int64), st_rdy=z(B * s_c4, np.int64),
        st_eop=z(B * s_c4, np.int64), st_eend=z(B * s_c4, np.float64),
        st_po=z(B * s_pool, np.int64), st_pw=z(B * s_pool, np.float64),
        st_pa=z(B * s_pool, np.bool_),
        st_pn=z(B, np.int64), st_pslot=z(B * s_ng, np.int64),
        st_lu=z(B, np.float64),
        st_sa=z(B * s_sa, np.int64), st_sw=z(B * s_pool, np.float64),
        rel_sched=z(B * s_ng, np.int64),
        heap=z(B * s_heap, np.int64), heap_n=z(B, np.int64),
        ret_buf=z(B * s_ret, np.int64), out_mk=z(B, np.float64),
    )
    return scr, (s_ng, s_c4, s_pool, s_sa, s_heap, s_ret)


def main():
    from numba import cuda
    from gpu_scene_b_batch import scene_b_batch_kernel

    cases = sys.argv[1:] or ["case_019", "case_064", "case_001",
                             "case_005"]
    n_bad = 0
    for case in cases:
        manifest = [json.loads(l) for l in
                    open(os.path.join(WORK, f"{case}_candidates.jsonl"),
                         encoding="utf-8")]
        ref = [json.loads(l) for l in
               open(os.path.join(REF, f"{case}_gpu_results.jsonl"),
                    encoding="utf-8")]
        ref_by_hash = {r["plan_hash"]: r for r in ref}
        z = np.load(os.path.join(WORK, f"{case}_scene.npz"))
        pm = z["plan_meta"].astype(np.int64)
        B = len(pm)
        # npz 里 stage-ok 的 plan 顺序 = manifest 中 pack_idx 顺序
        packs = [m for m in manifest if m.get("stage_status") == "ok"]
        packs.sort(key=lambda m: m["pack_idx"])
        assert len(packs) == B, (len(packs), B)

        mng = (pm[:, 1]).max()
        mnc = (pm[:, 0]).max()
        mnl = (pm[:, 2]).max()
        scr, st = alloc_scratch(B, mng, mnc, mnl)

        d = cuda.to_device
        din = tuple(d(z[k]) for k in
                    ["gop_pipe", "gop_dur", "gop_is_ddr", "gop_core",
                     "task_seq_ptr", "task_seq_arr",
                     "pipe_ptr", "pipe_seq_arr", "pipe_ptr_base",
                     "pipe_seq_base", "task_op_base",
                     "succ_ptr", "succ_arr", "pred_cnt",
                     "ext_pred_arr", "ext_succ_ptr", "ext_succ_arr"])
        doff = tuple(d(z[k].astype(np.int64)) for k in
                     ["gop_off", "ptrN_off", "core_off", "core1_off",
                      "core5_off", "ts_off", "succ_off", "link_off"])
        dpm = d(pm)
        dscr = tuple(d(v) for v in scr.values())
        out = dscr[-1]  # out_mk 是最后一个

        t0 = time.perf_counter()
        grid = (B + TPB - 1) // TPB
        scene_b_batch_kernel[grid, TPB](
            np.int64(B), np.int64(0), dpm, *doff, *din, *dscr,
            *[np.int64(s) for s in st])
        cuda.synchronize()
        t_gpu = time.perf_counter() - t0
        mks = out.copy_to_host()

        n_ok = n_err = 0
        bad = []
        for i, m in enumerate(packs):
            r = ref_by_hash[m["plan_hash"]]
            mk = int(round(mks[i]))
            if mk >= 0:
                n_ok += 1
                if r["status"] != "ok" or int(r["mk"]) != mk:
                    bad.append((m["plan_hash"][:8], "ok-vs",
                                r.get("mk"), mk))
            else:
                n_err += 1
                if r["status"] == "ok":
                    bad.append((m["plan_hash"][:8], "err-vs-ok",
                                r.get("mk"), mk))
                else:
                    old = int(r["err"].split()[-1])
                    if MAP.get(old, old) != mk:
                        bad.append((m["plan_hash"][:8], "errcode",
                                    MAP.get(old, old), mk))
        n_bad += len(bad)
        print(f"{case}: B={B} ok={n_ok} err={n_err} "
              f"mismatch={len(bad)} gpu={t_gpu*1000:.0f}ms "
              f"({B/t_gpu:.1f} plan/s)", flush=True)
        for b in bad[:5]:
            print("   BAD", b)
    print("GPU_SCENE_B_BATCH_GATE:",
          "PASS" if n_bad == 0 else f"FAIL ({n_bad})")
    sys.exit(0 if n_bad == 0 else 1)


if __name__ == "__main__":
    main()
