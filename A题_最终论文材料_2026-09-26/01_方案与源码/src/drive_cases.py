# -*- coding: utf-8 -*-
"""逐例独立进程驱动:单例超时自动跳过,队列永不堵塞。"""
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable


def drive(cases, q, out_dir, iters=120, batch=12, workers=10,
          timeout=420, log_name="drive.log"):
    out = HERE / out_dir
    out.mkdir(exist_ok=True)
    log = open(HERE / log_name, "a", encoding="utf-8")
    done = 0
    skip = 0
    for case in cases:
        fp = out / f"{case}_q{q}_N5.json"
        if fp.exists():
            done += 1
            continue
        t0 = time.perf_counter()
        try:
            engine = os.environ.get("N5_ENGINE", "refine2")
            r = subprocess.run(
                [PY, str(HERE / f"{engine}.py"), "--cases", case,
                 "--q", str(q), "--iters", str(iters), "--batch", str(batch),
                 "--workers", str(workers),
                 "--seeds", "2" if iters >= 1000 else "1",
                 "--out", str(out)],
                capture_output=True, text=True, timeout=timeout,
                cwd=str(HERE))
            line = [ln for ln in r.stdout.splitlines() if "sp=" in ln]
            if line:
                log.write(line[-1] + "\n")
                log.flush()
                done += 1
            else:
                log.write(f"{case}: NO-RESULT "
                          f"{r.stdout[-100:] if r.stdout else ''} "
                          f"{r.stderr[-150:] if r.stderr else ''}\n")
                log.flush()
                skip += 1
        except subprocess.TimeoutExpired:
            log.write(f"{case}: TIMEOUT({timeout}s) skipped\n")
            log.flush()
            skip += 1
    log.write(f"== done {done} skip {skip} ==\n")
    log.close()


if __name__ == "__main__":
    q = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    which = sys.argv[2] if len(sys.argv) > 2 else "prio"
    out_override = sys.argv[3] if len(sys.argv) > 3 else None
    if which == "prio":
        cases = open(HERE / "cases_prio94.txt").read().strip().split(",")
    elif which == "prio_q2":
        cases = open(HERE / "cases_prio94_q2.txt").read().strip().split(",")
    elif which == "light":
        cases = ["case_003", "case_040", "case_043", "case_053",
                 "case_054", "case_063", "case_068", "case_085",
                 "case_098", "case_002"]
    elif which == "mega":
        cases = ["case_016", "case_028", "case_067", "case_091",
                 "case_079", "case_014"]
    elif which == "mega2":
        cases = ["case_016", "case_028", "case_067", "case_091",
                 "case_079", "case_014"]
    else:
        cases = which.split(",")
    if which == "prio_q2":
        drive(cases, q, out_dir="refined2_q2", iters=120, batch=12,
              workers=12, timeout=420,
              log_name=f"drive_q{q}_{which}.log")
    elif which == "light":
        drive(cases, q, out_dir="refined2_light", iters=25, batch=6,
              workers=6, timeout=900,
              log_name=f"drive_q{q}_{which}.log")
    elif which == "mega2":
        drive(cases, q, out_dir="refined2_mega", iters=20, batch=6,
              workers=8, timeout=1500,
              log_name=f"drive_q{q}_{which}.log")
    else:
        if out_override:
            mara = os.environ.get("N5_MARATHON", "0") == "1"
            drive(cases, q, out_dir=out_override,
                  iters=1500 if mara else 150, batch=12,
                  workers=12, timeout=2100 if mara else 420,
                  log_name=f"drive_q{q}_{out_override}.log")
        else:
            drive(cases, q, out_dir="refined2_mega" if which == "mega" else "refined2",
                  iters=100 if which == "mega" else 120,
                  workers=6 if which == "mega" else 12,
                  timeout=900 if which == "mega" else 420,
                  log_name=f"drive_q{q}_{which}.log")
