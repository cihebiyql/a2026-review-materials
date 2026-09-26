# 拉回方案官方评估器终验(绕开 FastEval 原生崩溃)
import sys, json
sys.path.insert(0, r'C:/shumo_live/02_求解/A题_2026/v3_solver')
sys.path.insert(0, r'C:/shumo_live/a_data/code')
from common import load_case, ev_p3

ROOT = r'C:/shumo_live/02_求解/A题_2026'
pool = json.load(open(ROOT + '/audit_20260924_latest/POOL_q3_N5.json'))
PULL_MK = {'case_080': None, 'case_063': None, 'case_090': None,
           'case_012': None, 'case_056': None}

for case in ['case_080', 'case_063', 'case_090', 'case_012', 'case_056']:
    d = json.load(open(ROOT + f'/n5_push/bxcpu_pull/{case}_q3_pull.json'))
    plan = d['plan'] if 'plan' in d else d
    g = load_case(case)
    r, wt = ev_p3(g, plan)
    mk = r['makespan']
    sc = json.load(open(ROOT + f'/results/singlecore/{case}_sc.json'))['makespan']
    print(f'{case}: official mk={mk} sp={sc/mk:.4f} | pool={pool[case]:.4f} '
          f'| slots={len(plan["core_schedules"])} ({wt:.0f}s)', flush=True)
