# -*- coding: utf-8 -*-
"""arm_g5tf 批量收割:87 例 × 官方评估入池。"""
import subprocess, sys, json, time, os
from pathlib import Path
HERE = Path(__file__).resolve().parent
LAB = Path(r"C:/shumo_live/02_求解/A题_2026/a_lab")
PY = sys.executable
OUT = HERE / "arm_g5tf_out"
OUT.mkdir(exist_ok=True)

def eval_plan(case, plan):
    import sys as s
    s.path.insert(0, str(LAB / "solvers" / "exp_strand"))
    s.path.insert(0, r"C:/shumo_live/a_data/code")
    from common import load_case, ev_p3
    g = load_case(case)
    sc = json.load(open(rf"C:/shumo_live/02_求解/A题_2026/results/singlecore/{case}_sc.json"))["makespan"]
    r, _ = ev_p3(g, plan)
    return sc / r["makespan"], r["makespan"]

def main():
    cases = open(HERE / "cases_arm87.txt").read().strip().split(",")
    pool = json.load(open(HERE / "POOL_q3_N5.json"))
    log = open(HERE / "arm_drive.log", "a", encoding="utf-8")
    for case in cases:
        fp = OUT / case
        if (fp / "plan.json").exists():
            continue
        t0 = time.perf_counter()
        try:
            subprocess.run(
                [PY, str(LAB / "lab_solvers" / "arm_g5tf.py"),
                 "--graph", rf"C:/shumo_live/a_data/data/{case}.json",
                 "--question", "3", "--cores", "5", "--seed", "11",
                 "--budget-seconds", "300", "--workdir", str(fp)],
                capture_output=True, text=True, timeout=420, cwd=str(LAB))
            if (fp / "plan.json").exists():
                plan = json.load(open(fp / "plan.json"))
                sp, mk = eval_plan(case, plan)
                json.dump({"plan": plan, "mk": mk, "sp": sp},
                          open(OUT / f"{case}_q3_N5.json", "w"))
                pv = pool.get(case, 0)
                log.write(f"{case}: arm sp={sp:.4f} (池 {pv:.3f}, "
                          f"{(sp/pv-1)*100 if pv else 0:+.1f}%) "
                          f"[{time.perf_counter()-t0:.0f}s]\n")
            else:
                log.write(f"{case}: NO-PLAN [{time.perf_counter()-t0:.0f}s]\n")
        except subprocess.TimeoutExpired:
            log.write(f"{case}: TIMEOUT\n")
        except Exception as exc:
            log.write(f"{case}: ERR {str(exc)[:60]}\n")
        log.flush()
    log.write("== arm done ==\n")
    log.close()

if __name__ == "__main__":
    main()
