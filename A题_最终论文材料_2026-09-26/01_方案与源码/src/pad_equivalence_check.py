# P0-A: 补空槽等价性实测 F(ι5(P)) = F(P)
# 对 verified_merge 中 slots<5 的胜出方案,官方评测原槽数 vs 末尾补空槽
import sys, json, csv
sys.path.insert(0, r'C:/shumo_live/02_求解/A题_2026/v3_solver')
sys.path.insert(0, r'C:/shumo_live/a_data/code')
from common import load_case, ev_p2, ev_p3
from pathlib import Path

ROOT = Path(r'C:/shumo_live/02_求解/A题_2026')
PP = ROOT / 'n5_push/pull_plans'
PAD = ROOT / 'n5_push/plans_padded'; PAD.mkdir(exist_ok=True)

for q in (3, 2):
    ev = ev_p3 if q == 3 else ev_p2
    pool = json.load(open(ROOT / f'audit_20260924_latest/POOL_q{q}_N5.json'))
    for row in csv.DictReader(open(ROOT / f'audit_20260924_latest/verified_merge_q{q}.csv', encoding='utf-8')):
        if row['status'] != 'verified_gain':
            continue
        c = row['case']
        fp = PP / f'{c}_q{q}_pull.json'
        d = json.load(open(fp))
        plan = d['plan'] if 'plan' in d else d
        k = len(plan['core_schedules'])
        if k >= 5:
            continue
        g = load_case(c)
        mk_orig = ev(g, plan)[0]['makespan']
        padded = {'node_to_subgraph': plan['node_to_subgraph'],
                  'core_schedules': list(plan['core_schedules']) + [[]] * (5 - k)}
        mk_pad = ev(g, padded)[0]['makespan']
        eq = (mk_orig == mk_pad)
        print(f'{c} q{q}: slots={k} mk_orig={mk_orig} mk_padded={mk_pad} equal={eq}', flush=True)
        if eq:
            json.dump({'plan': padded, 'mk': mk_pad, 'orig_slots': k,
                       'mk_orig': mk_orig, 'source': str(fp)},
                      open(PAD / f'{c}_q{q}_padded5.json', 'w'))
