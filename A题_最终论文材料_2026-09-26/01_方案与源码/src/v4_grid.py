# -*- coding: utf-8 -*-
"""v4 全网格:100 例 × N∈{2,3,4,5} × q∈{2,3},韧性子进程驱动。"""
import subprocess
import sys
import time
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
OUT = HERE / "v4_out"
OUT.mkdir(exist_ok=True)


def main():
    import os
    qs = tuple(int(x) for x in os.environ.get('N5_GRID_Q', '2,3').split(','))
    tasks = []
    for q in qs:
        for N in (2, 3, 4, 5):
            for i in range(1, 101):
                case = f"case_{i:03d}"
                fp = OUT / f"{case}_q{q}_N{N}_s11.json"
                if fp.exists():
                    continue
                tasks.append((case, q, N))
    log = open(HERE / "v4_grid.log", "a", encoding="utf-8")
    print(f"[grid] 待跑 {len(tasks)} 任务", flush=True)
    log.write(f"[start] {len(tasks)} tasks\n")
    log.flush()
    import os
    parallel = int(os.environ.get('N5_GRID_PAR', '1'))
    from concurrent.futures import ThreadPoolExecutor
    if parallel > 1:
        with ThreadPoolExecutor(parallel) as ex:
            list(ex.map(lambda t: run_one(t), tasks))
        return
    for t in tasks:
        run_one(t)

def run_one(t):
    case, q, N = t
    t0 = time.perf_counter()
    try:
        r = subprocess.run(
            [PY, str(HERE / "v4_solver.py"), "--case", case,
             "--q", str(q), "--N", str(N), "--budget", "300",
             "--out", str(OUT)],
            capture_output=True, text=True, timeout=420, cwd=str(HERE))
        line = [ln for ln in r.stdout.splitlines() if "sp=" in ln]
        if line:
            log.write(line[-1] + "\n")
        else:
            log.write(f"{case} q{q} N{N}: NO-RESULT "
                      f"{r.stderr[-120:] if r.stderr else ''}\n")
    except subprocess.TimeoutExpired:
        log.write(f"{case} q{q} N{N}: TIMEOUT\n")
    except Exception as exc:
        log.write(f"{case} q{q} N{N}: ERR {str(exc)[:60]}\n")
    log.flush()
    log.write("== grid done ==\n")
    log.close()


if __name__ == "__main__":
    main()
    pass

