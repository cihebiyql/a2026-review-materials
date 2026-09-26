# 全库双指标审计: 逐例冠军方案的 added_copy_bytes 分布(题面次要指标)
import sys, json, os, glob
sys.path.insert(0, r'C:/shumo_live/02_求解/A题_2026/fast_eval')
sys.path.insert(0, r'C:/shumo_live/02_求解/A题_2026/v3_solver')
sys.path.insert(0, r'C:/shumo_live/a_data/code')
from common import load_case
from fast_eval_p2 import FastEvalP2, FastEvalP3
from fast_eval_p1 import FastEvalP1

AUD = r'C:/shumo_live/02_求解/A题_2026/audit_20260924_latest'
N5 = r'C:/shumo_live/02_求解/A题_2026/n5_push'

def champion(case, q):
    pats = []
    if q == 1:
        pats += [f'{N5}/p1_refine_out/{case}_q1_refined.json', f'{N5}/p1_seeds/{case}_seed.json']
    else:
        pats += [f'{N5}/superlinear_analysis/{case}_q{q}_mechA.json',
                 f'{N5}/node1_mechA_out/{case}_q{q}_mechA.json',
                 f'{N5}/split_probe_out/{case}_q{q}_best.json']
        for sub in ('refined2', 'refined', 'strand_n5'):
            pats.append(f'{N5}/{sub}/{case}_q{q}_N5.json')
        pats.append(f'{N5}/pull_plans/{case}_q{q}_pull.json')
    best = (0.0, None)
    for pat in pats:
        for fp in glob.glob(pat):
            try: d = json.load(open(fp))
            except Exception: continue
            if d.get('sp', 0) > best[0] and 'plan' in d:
                best = (d['sp'], d['plan'])
    return best

out_rows = []
if os.environ.get('AUD_ONE'):
    q = int(os.environ['AUD_ONE']); case = os.environ['AUD_CASE']
    import json as _j
    pool = {}
    try:
        sp_pool, plan = champion(case, q)
        g = load_case(case)
        fe = FastEvalP1(g) if q == 1 else (FastEvalP2(g) if q == 2 else FastEvalP3(g))
        mk, info = fe.evaluate(plan)
        ratio = info['added_copy_bytes'] / max(1, info['original_copy_bytes'])
        print(_j.dumps({'case': case, 'q': q, 'mk': mk,
                        'orig_MB': round(info['original_copy_bytes']/1e6, 3),
                        'added_MB': round(info['added_copy_bytes']/1e6, 3),
                        'ratio': round(ratio, 3),
                        'spill_MB': round(info['spill_added_copy_bytes']/1e6, 3)}), flush=True)
    except Exception as e:
        print(_j.dumps({'case': case, 'q': q, 'status': 'ERR:' + str(e)[:40]}), flush=True)
    sys.exit(0)
for q in (1, 2, 3):
    pool = json.load(open(f'{AUD}/verified_merge_q1.json')) if q == 1 else json.load(open(f'{AUD}/final_preview_q{q}.json'))
    risky = []
    for i in range(1, 101):
        case = f'case_{i:03d}'
        if case in ('case_014', 'case_040') and q == 3:
            continue  # FastEval 崩图另行处理
        try:
            sp_pool, plan = champion(case, q)
            g = load_case(case)
            if q == 1: fe = FastEvalP1(g)
            elif q == 2: fe = FastEvalP2(g)
            else: fe = FastEvalP3(g)
            mk, info = fe.evaluate(plan)
            ratio = info['added_copy_bytes'] / max(1, info['original_copy_bytes'])
            out_rows.append({'case': case, 'q': q, 'sp': round(sp_pool, 4),
                             'mk': mk, 'orig_MB': round(info['original_copy_bytes']/1e6, 3),
                             'added_MB': round(info['added_copy_bytes']/1e6, 3),
                             'ratio': round(ratio, 3),
                             'spill_MB': round(info['spill_added_copy_bytes']/1e6, 3)})
            if ratio > 1.0:
                risky.append((case, round(ratio, 2)))
        except Exception:
            out_rows.append({'case': case, 'q': q, 'status': 'ERR'})
    print(f'q{q}: 高搬运风险例(ratio>1): {len(risky)} -> {risky[:12]}', flush=True)
json.dump(out_rows, open(f'{N5}/bytes_audit.json', 'w'), ensure_ascii=False, indent=1)
print('AUDIT_DONE')
