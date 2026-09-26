import sys, json, csv, glob, os
sys.path.insert(0, r'C:/shumo_live/02_求解/A题_2026/fast_eval')
sys.path.insert(0, r'C:/shumo_live/02_求解/A题_2026/v3_solver')
sys.path.insert(0, r'C:/shumo_live/02_求解/A题_2026/n5_push/superlinear_analysis')
sys.path.insert(0, r'C:/shumo_live/02_求解/A题_2026/n5_push')
sys.path.insert(0, r'C:/shumo_live/a_data/code')
from common import load_case
from fast_eval_p2 import FastEvalP2
from phase3_mechA2 import comp_depth
from mechA_v3 import relabel_v3
case = sys.argv[2]
plan0 = None; best_sp = 0
for fp in glob.glob(rf'superlinear_analysis/{case}_q2_mechA.json'):
    d = json.load(open(fp)); plan0 = d['plan']
for sub in ('refined2', 'refined', 'strand_n5'):
    for fp in glob.glob(rf'{sub}/{case}_q2_N5.json'):
        d = json.load(open(fp))
        if d.get('sp', 0) > best_sp and 'plan' in d and (plan0 is None or d['sp'] > best_sp):
            plan0 = d['plan']; best_sp = d['sp']
if plan0 is None:
    print(case, 'no seed'); sys.exit()
g = load_case(case)
fe = FastEvalP2(g)
sc = json.load(open(rf'../results/singlecore/{case}_sc.json'))['makespan']
mk0, _ = fe.evaluate(plan0)
comp_of, depth = comp_depth(g)
core_of_sg = {sg: c for c, sgl in enumerate(plan0['core_schedules']) for sg in sgl}
op_core = {op: core_of_sg[sg] for op, sg in {int(k): v for k, v in plan0['node_to_subgraph'].items()}.items()}
out = [{'case': case, 'eval_idx': 0, 'best_sp': round(sc/mk0, 4)}]
best = mk0; i = 0
for B in range(8, 240, 4):
    for L in (3, 4):
        try:
            mk1 = fe.evaluate(relabel_v3(plan0, op_core, comp_of, depth, B, L, 'min_id'))[0]
            i += 1
            if mk1 < best: best = mk1
            out.append({'case': case, 'eval_idx': i, 'best_sp': round(sc/best, 4)})
        except Exception: pass
with open('../audit_20260924_latest/deliverable/convergence_curve.csv', 'a', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['case', 'eval_idx', 'best_sp'])
    w.writerows(out)
print(case, 'points:', len(out))
