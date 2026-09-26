# P0-C: 胜者完整绑定的 verified_merge_v2
# 修正审阅指出的配对缺陷:胜者是 pool 时 official_mk 不再沿用 pull 方案的 mk;
# 胜者是 pull 时登记补槽后方案哈希(等价性已由 pad_equivalence_check 实测)。
import sys, json, csv, hashlib
sys.path.insert(0, r'C:/shumo_live/02_求解/A题_2026/v3_solver')
from pathlib import Path

ROOT = Path(r'C:/shumo_live/02_求解/A题_2026')
AUD = ROOT / 'audit_20260924_latest'
PP = ROOT / 'n5_push/pull_plans'
PAD = ROOT / 'n5_push/plans_padded'


def plan_hash(plan):
    s = json.dumps({'node_to_subgraph': plan['node_to_subgraph'],
                    'core_schedules': plan['core_schedules']}, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


for q in (2, 3):
    pool = json.load(open(AUD / f'POOL_q{q}_N5.json'))
    rows, verified = [], {}
    tot_old = tot_new = 0.0
    for c in sorted(pool):
        psp = pool[c]; tot_old += psp
        fp = PP / f'{c}_q{q}_pull.json'
        rec = {'case': c, 'q': q, 'pool_sp': round(psp, 6),
               'winner': 'pool', 'winner_official_mk': '',
               'winner_plan_hash': '', 'winner_slots': 5,
               'padded_equiv': '', 'verified_sp': round(psp, 6),
               'gain': 0.0}
        if fp.exists():
            d = json.load(open(fp))
            plan = d['plan'] if 'plan' in d else d
            # v1 csv 里的官方终验 mk(原槽数评测)
            v1mk = None
            for row in csv.DictReader(open(AUD / f'verified_merge_q{q}.csv', encoding='utf-8')):
                if row['case'] == c and row['official_mk']:
                    v1mk = int(row['official_mk']); break
            pf = PAD / f'{c}_q{q}_padded5.json'
            if v1mk is not None:
                sc = json.load(open(ROOT / f'results/singlecore/{c}_sc.json'))['makespan']
                vsp = sc / v1mk
                if vsp > psp + 1e-9:
                    padded_plan = json.load(open(pf))['plan'] if pf.exists() else \
                        {'node_to_subgraph': plan['node_to_subgraph'],
                         'core_schedules': list(plan['core_schedules']) +
                         [[]] * (5 - len(plan['core_schedules']))}
                    rec.update(winner='pull(官方实测mk)', winner_official_mk=v1mk,
                               winner_plan_hash=plan_hash(padded_plan),
                               winner_slots=5,
                               padded_equiv='实测相等' if pf.exists() else '',
                               verified_sp=round(vsp, 6), gain=round(vsp - psp, 6))
                    tot_new += vsp
                    verified[c] = vsp
                    rows.append(rec); continue
        verified[c] = psp; tot_new += psp
        rows.append(rec)
    n = len(pool)
    json.dump(verified, open(AUD / f'verified_merge_v2_q{q}.json', 'w'), indent=1)
    with open(AUD / f'verified_merge_v2_q{q}.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f'q{q}: pool {tot_old/n:.4f} -> verified_v2 {tot_new/n:.4f} '
          f'(+{tot_new-tot_old:.2f} 均值sp), pull-winners={sum(1 for r in rows if r["winner"]!="pool")}')
