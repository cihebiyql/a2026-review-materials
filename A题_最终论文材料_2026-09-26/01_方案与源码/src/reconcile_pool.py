# reconcile_pool.py — 冠军池全源对账: 逐(case,Q)取全源最优方案, 落回 superlinear_analysis
# 本程序及代码是在人工智能工具辅助下完成的。
# 人工智能工具名称：GLM，版本/型号：zai-api/GLM-5.3，开发机构/公司：智谱AI(Z.ai)，版本颁布日期：2026-09-26
import json, glob, os, re
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
SA = BASE + '/superlinear_analysis'

sc = {}
for i in range(1, 101):
    c = f'case_{i:03d}'
    sc[c] = json.load(open(rf'C:/shumo_live/02_求解/A题_2026/results/singlecore/{c}_sc.json'))['makespan']

NAME = re.compile(r'^(case_\d{3})_q([123])')
best = {}


def consider(fp, tag):
    m = NAME.search(os.path.basename(fp))
    if not m:
        return
    case, q = m.group(1), int(m.group(2))
    try:
        d = json.load(open(fp))
    except Exception:
        return
    if not (isinstance(d, dict) and d.get('plan') is not None):
        return
    mk = d.get('mk')
    if not mk and d.get('sp'):
        mk = sc[case] / d['sp']
    if not mk:
        return
    if (case, q) not in best or mk < best[(case, q)][0]:
        best[(case, q)] = (mk, d['plan'], tag)


for fp in glob.glob(f'{SA}/case_*_q[123]_mechA.json'):
    consider(fp, 'mechA')
for sub in ('refined8', 'refined7', 'refined6', 'refined5', 'refined4', 'refined3',
            'refined2', 'refined', 'refined_mara', 'strand_n5', 'pull_plans'):
    for fp in glob.glob(f'{BASE}/{sub}/case_*_q[123]_*.json'):
        consider(fp, sub)
for fp in glob.glob(f'{BASE}/reinforce_v5/improve_json/*_reinforce.json'):
    consider(fp, 'v5')
# final_seeds_q{1,2,3}.jsonl: 每行 {case, mk, sp, plan} —— GPU 战役官方终验产物
for q in (1, 2, 3):
    fp = f'{BASE}/gpu_search/final_seeds_q{q}.jsonl'
    if not os.path.exists(fp):
        continue
    for line in open(fp, encoding='utf-8'):
        try:
            d = json.loads(line)
        except Exception:
            continue
        case = d.get('case')
        if case in sc and d.get('mk') and d.get('plan'):
            if (case, q) not in best or d['mk'] < best[(case, q)][0]:
                best[(case, q)] = (d['mk'], d['plan'], 'final_seeds')
for fp in glob.glob(BASE + '/../audit_20260924_latest/verified_merge_q*.json'):
    q = int(re.search(r'_q(\d)', os.path.basename(fp)).group(1))
    try:
        d = json.load(open(fp))
    except Exception:
        continue
    items = d.values() if isinstance(d, dict) else d
    for v in items:
        if isinstance(v, dict) and v.get('plan') and v.get('case') and v.get('mk'):
            c = v['case']
            if (c, q) not in best or v['mk'] < best[(c, q)][0]:
                best[(c, q)] = (v['mk'], v['plan'], 'audit_v1')
for fp in glob.glob(BASE + '/../audit_20260924_latest/verified_merge_v2_q*.json'):
    q = int(re.search(r'_q(\d)', os.path.basename(fp)).group(1))
    try:
        d = json.load(open(fp))
    except Exception:
        continue
    if not isinstance(d, dict):
        continue
    for case, v in d.items():
        if isinstance(v, dict) and v.get('plan') is not None and v.get('mk'):
            if (case, q) not in best or v['mk'] < best[(case, q)][0]:
                best[(case, q)] = (v['mk'], v['plan'], 'audit_v2')

avg = defaultdict(list)
merged = 0
for (c, q), (mk, plan, src) in best.items():
    avg[q].append(sc[c] / mk)
    dst = f'{SA}/{c}_q{q}_mechA.json'
    old = json.load(open(dst)) if os.path.exists(dst) else None
    if not old or not old.get('mk') or int(mk) < old['mk']:
        json.dump({'plan': plan, 'mk': int(mk), 'sp': sc[c] / mk,
                   'source': f'reconcile {src}'}, open(dst, 'w'))
        merged += 1

for q in (1, 2, 3):
    print(f'q{q}: {len(avg[q])}例 平均sp={sum(avg[q]) / len(avg[q]):.4f}')
print('reconcile 更新文件数:', merged)
