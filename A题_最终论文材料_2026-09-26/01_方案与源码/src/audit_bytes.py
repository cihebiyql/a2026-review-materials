import sys, json, os, glob
sys.path.insert(0, r'C:/shumo_live/02_求解/A题_2026/fast_eval')
sys.path.insert(0, r'C:/shumo_live/02_求解/A题_2026/v3_solver')
sys.path.insert(0, r'C:/shumo_live/a_data/code')
from common import load_case
from fast_eval_p2 import FastEvalP2, FastEvalP3


def champion(case, q):
    best = (0.0, None)
    for fp in (glob.glob(rf'C:/shumo_live/02_求解/A题_2026/n5_push/superlinear_analysis/{case}_q{q}_mechA.json') +
              glob.glob(rf'C:/shumo_live/02_求解/A题_2026/n5_push/node1_mechA_out/{case}_q{q}_mechA.json')):
        d = json.load(open(fp))
        if d.get('sp', 0) > best[0]:
            best = (d['sp'], d['plan'])
    for sub in ('refined2', 'refined', 'strand_n5'):
        for fp in glob.glob(rf'C:/shumo_live/02_求解/A题_2026/n5_push/{sub}/{case}_q{q}_N5.json'):
            d = json.load(open(fp))
            if d.get('sp', 0) > best[0]:
                best = (d['sp'], d['plan'])
    return best[1]


for case in ['case_084', 'case_008', 'case_095', 'case_041', 'case_046']:
    for q in (2, 3):
        try:
            plan = champion(case, q)
            if plan is None:
                continue
            g = load_case(case)
            fe = FastEvalP2(g) if q == 2 else FastEvalP3(g)
            sc = json.load(open(rf'C:/shumo_live/02_求解/A题_2026/results/singlecore/{case}_sc.json'))['makespan']
            mk, info = fe.evaluate(plan)
            nsg = len(set(plan['node_to_subgraph'].values()))
            ratio = info['added_copy_bytes'] / max(1, info['original_copy_bytes'])
            print(f"{case} q{q}: sp={sc/mk:.3f} orig={info['original_copy_bytes']/1e6:.2f}MB "
                  f"added={info['added_copy_bytes']/1e6:.2f}MB ({ratio:.2f}x) "
                  f"spill={info['spill_added_copy_bytes']/1e6:.2f}MB n_sg={nsg}", flush=True)
        except Exception as e:
            print(case, q, 'ERR', str(e)[:50], flush=True)
