# 重合池:pool(现口径) vs 全部官方 pull 记录(任意核数)按例取最优
# 产出:POOL_q{2,3}_merged.json + merged_source_map_q{2,3}.csv
import json, glob, csv
from pathlib import Path

ROOT = Path(r'C:/shumo_live/02_求解/A题_2026')
AUD = ROOT / 'audit_20260924_latest'


def best_pulls(q):
    best = {}
    for fp in glob.glob(str(ROOT / 'a_lab/records/**/*.jsonl'), recursive=True):
        for line in open(fp, encoding='utf-8'):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get('question') != q or r.get('status') != 'ok':
                continue
            if r.get('cores') and r['cores'] > 5:
                continue  # 防超五核方案混入(审阅意见 P0-B)
            leg = r.get('legality') or {}
            if leg.get('legal') is False:
                continue
            c = r.get('case'); sp = r.get('speedup') or 0
            if not c:
                continue
            if c not in best or sp > best[c]['sp']:
                best[c] = {'sp': sp, 'mk': r.get('makespan'),
                           'sc': r.get('sc_makespan'), 'cores': r.get('cores'),
                           'plan_path': r.get('plan_path'),
                           'exp': r.get('experiment_id'),
                           'ts': r.get('timestamp_utc'),
                           'eval_hash': (r.get('binding') or {}).get('evaluator_code_hash')}
    return best


for q in (2, 3):
    pool = json.load(open(AUD / f'POOL_q{q}_N5.json'))
    bp = best_pulls(q)
    merged, rows, tot_old, tot_new, gains = {}, [], 0.0, 0.0, 0
    for c in sorted(pool):
        psp = pool[c]; tot_old += psp
        b = bp.get(c)
        if b and b['sp'] > psp + 1e-9:
            merged[c] = b['sp']; tot_new += b['sp']; gains += 1
            rows.append({'case': c, 'q': q, 'pool_sp': round(psp, 6),
                         'pull_sp': round(b['sp'], 6), 'mk': b['mk'],
                         'cores': b['cores'], 'exp': b['exp'],
                         'plan_path': b['plan_path'], 'ts': b['ts'],
                         'source': 'pull', 'verified_local': ''})
        else:
            merged[c] = psp; tot_new += psp
            rows.append({'case': c, 'q': q, 'pool_sp': round(psp, 6),
                         'pull_sp': round(b['sp'], 6) if b else '',
                         'mk': b['mk'] if b else '', 'cores': b['cores'] if b else '',
                         'exp': b['exp'] if b else '', 'plan_path': b['plan_path'] if b else '',
                         'ts': b['ts'] if b else '', 'source': 'pool', 'verified_local': ''})
    n = len(pool)
    json.dump(merged, open(AUD / f'POOL_q{q}_merged.json', 'w'), indent=1)
    with open(AUD / f'merged_source_map_q{q}.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f'q{q}: pool mean {tot_old/n:.4f} -> merged {tot_new/n:.4f} '
          f'(+{(tot_new-tot_old)*100:.1f} 点), gains {gains}/{n}')
