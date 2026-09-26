# 对策1: 高搬运例的帕累托备选——在全部历史方案中找 (sp, added) 非支配方案
# 输出: 每例的冠军方案 vs 低搬运备选(added<=1xorig 中 sp 最高)
import sys, json, os, glob
sys.path.insert(0, r'C:/shumo_live/02_求解/A题_2026/fast_eval')
sys.path.insert(0, r'C:/shumo_live/02_求解/A题_2026/v3_solver')
sys.path.insert(0, r'C:/shumo_live/a_data/code')
from common import load_case
from fast_eval_p2 import FastEvalP2, FastEvalP3

AUD = r'C:/shumo_live/02_求解/A题_2026/audit_20260924_latest'
N5 = r'C:/shumo_live/02_求解/A题_2026/n5_push'

audit = [json.loads(l) for l in open(f'{N5}/bytes_audit_one.log', encoding='utf-8') if l.strip().startswith('{') and '"mk"' in l]
high = [r for r in audit if r['ratio'] > 1 and r['q'] == 2]
high.sort(key=lambda r: -r['added_MB'])
targets = [r['case'] for r in high[:20]]
print(f'q2 高搬运 top20: {targets}')


def all_plans(case, q):
    out = []
    pats = [f'{N5}/superlinear_analysis/{case}_q{q}_mechA.json',
            f'{N5}/node1_mechA_out/{case}_q{q}_mechA.json']
    for sub in ('refined2', 'refined', 'strand_n5'):
        pats.append(f'{N5}/{sub}/{case}_q{q}_N5.json')
    pats.append(f'{N5}/pull_plans/{case}_q{q}_pull.json')
    for pat in pats:
        for fp in glob.glob(pat):
            try:
                d = json.load(open(fp))
            except Exception:
                continue
            if 'plan' in d and d.get('sp'):
                out.append((os.path.basename(fp), d['sp'], d['plan']))
    return out


results = []
for case in targets:
    try:
        g = load_case(case)
        fe = FastEvalP2(g)
        sc = json.load(open(rf'C:/shumo_live/02_求解/A题_2026/results/singlecore/{case}_sc.json'))['makespan']
        cands = all_plans(case, 2)
        champ = max(cands, key=lambda x: x[1])
        mk_c, info_c = fe.evaluate(champ[2])
        orig = info_c['original_copy_bytes']
        clean = [c for c in cands]
        best_clean = None
        for name, sp, plan in cands:
            mk, info = fe.evaluate(plan)
            if info['added_copy_bytes'] <= orig:
                if best_clean is None or sp > best_clean[1]:
                    best_clean = (name, sp, plan, info['added_copy_bytes'])
        row = {'case': case, 'champ_sp': round(champ[1], 4), 'champ_added_MB': round(info_c['added_copy_bytes']/1e6, 2),
               'champ_src': champ[0]}
        if best_clean:
            row.update(clean_sp=round(best_clean[1], 4), clean_added_MB=round(best_clean[3]/1e6, 2), clean_src=best_clean[0],
                       sp_loss=round(champ[1]-best_clean[1], 4))
        else:
            row['clean'] = '无满足 added<=1xorig 的方案'
        results.append(row)
        print(row, flush=True)
    except Exception as e:
        print(case, 'ERR', str(e)[:60], flush=True)
json.dump(results, open(f'{N5}/pareto_backup_q2.json', 'w'), ensure_ascii=False, indent=1)
