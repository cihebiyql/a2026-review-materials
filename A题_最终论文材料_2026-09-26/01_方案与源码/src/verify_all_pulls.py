# 全量官方终验 pull_plans,生成终验后安全合并池 verified_merge_q{2,3}.json
import sys, json, glob, csv, time
sys.path.insert(0, r'C:/shumo_live/02_求解/A题_2026/v3_solver')
sys.path.insert(0, r'C:/shumo_live/a_data/code')
from common import load_case, ev_p2, ev_p3
from pathlib import Path

ROOT = Path(r'C:/shumo_live/02_求解/A题_2026')
AUD = ROOT / 'audit_20260924_latest'
PP = ROOT / 'n5_push/pull_plans'

for q in (3, 2):
    pool = json.load(open(AUD / f'POOL_q{q}_N5.json'))
    ev = ev_p3 if q == 3 else ev_p2
    verified, rows = {}, []
    tot_old = tot_new = 0.0
    for c in sorted(pool):
        fp = PP / f'{c}_q{q}_pull.json'
        psp = pool[c]; tot_old += psp
        if not fp.exists():
            verified[c] = psp; tot_new += psp
            rows.append({'case': c, 'pool_sp': round(psp, 6), 'pull_sp': '',
                         'official_mk': '', 'verified_sp': round(psp, 6),
                         'gain': 0.0, 'slots': '', 'status': 'pool_only'})
            continue
        d = json.load(open(fp))
        plan = d['plan'] if 'plan' in d else d
        g = load_case(c)
        try:
            r, wt = ev(g, plan)
            mk = r['makespan']
        except Exception as ex:
            print(f'{c} q{q} OFFICIAL ERR {str(ex)[:60]}', flush=True)
            verified[c] = psp; tot_new += psp
            rows.append({'case': c, 'pool_sp': round(psp, 6), 'pull_sp': '',
                         'official_mk': '', 'verified_sp': round(psp, 6),
                         'gain': 0.0, 'slots': '', 'status': 'official_err'})
            continue
        sc = json.load(open(ROOT / f'results/singlecore/{c}_sc.json'))['makespan']
        vsp = sc / mk
        final = max(vsp, psp)
        verified[c] = final; tot_new += final
        rows.append({'case': c, 'pool_sp': round(psp, 6),
                     'pull_sp': round(vsp, 6), 'official_mk': mk,
                     'verified_sp': round(final, 6),
                     'gain': round(final - psp, 6),
                     'slots': len(plan['core_schedules']),
                     'status': 'verified_gain' if vsp > psp + 1e-9 else 'verified_no_gain'})
        print(f'{c} q{q}: official sp={vsp:.4f} pool={psp:.4f} '
              f'{"GAIN" if vsp > psp + 1e-9 else "no-gain"} slots={len(plan["core_schedules"])}',
              flush=True)
    n = len(pool)
    json.dump(verified, open(AUD / f'verified_merge_q{q}.json', 'w'), indent=1)
    with open(AUD / f'verified_merge_q{q}.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f'== q{q}: pool {tot_old/n:.4f} -> verified-merged {tot_new/n:.4f} '
          f'(+{(tot_new-tot_old)*100:.1f} 粗点 = +{tot_new-tot_old:.2f} 均值sp) ==',
          flush=True)
