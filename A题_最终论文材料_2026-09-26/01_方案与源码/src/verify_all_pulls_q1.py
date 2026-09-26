# P1 补扫终验:槽位过滤(题面 2~5 核,不能信 pull 的 cores 字段)+ 逐例官方评测
# 输入: pull_plans_q1/{case}_q1_pull.json + posthoc N5 基线
# 输出: verified_merge_q1.{json,csv}, 全量日志 verify_q1_full.log
import json, os, sys, csv
sys.path.insert(0, '<WORKDIR>/v3_solver')
sys.path.insert(0, '<EVALDIR>')
from common import load_case, ev_p1

ROOT = '<WORKDIR>'
p = json.load(open(ROOT + '/a_lab/records/POSTHOC_BEST.json'))
post = {k.split('|')[0]: v for k, v in p.items() if '|q1|N5' in k}
verified, rows = {}, {}
tot_old = tot_new = 0.0
for c in sorted(post):
    psp = post[c]; tot_old += psp
    fp = f'{ROOT}/n5_push/pull_plans_q1/{c}_q1_pull.json'
    if not os.path.exists(fp):
        verified[c] = psp; tot_new += psp
        rows[c] = {'status': 'posthoc_only', 'slots': ''}
        continue
    d = json.load(open(fp)); plan = d['plan'] if 'plan' in d else d
    k = len(plan['core_schedules'])
    if k > 5:  # 题面: 核数 2~5。实测发现 MKV2-N235 等实验产物为真实 6 核划分
        verified[c] = psp; tot_new += psp
        rows[c] = {'status': 'REJECT_over5slots', 'slots': k}
        print(f'{c}: REJECT slots={k} (>5)', flush=True)
        continue
    g = load_case(c)
    try:
        mk = ev_p1(g, plan)[0]['makespan']
    except Exception as ex:
        verified[c] = psp; tot_new += psp
        rows[c] = {'status': 'official_err:' + str(ex)[:40], 'slots': k}
        continue
    sc = json.load(open(ROOT + f'/results/singlecore/{c}_sc.json'))['makespan']
    vsp = sc / mk
    verified[c] = max(vsp, psp); tot_new += max(vsp, psp)
    rows[c] = {'status': 'gain' if vsp > psp + 1e-9 else 'no_gain',
               'slots': k, 'official_mk': mk,
               'posthoc_sp': round(psp, 6), 'verified_sp': round(max(vsp, psp), 6),
               'gain': round(max(vsp, psp) - psp, 6)}
    if vsp > psp + 1e-9:
        print(f'{c}: sp={vsp:.4f} posthoc={psp:.4f} GAIN slots={k}', flush=True)
n = len(post)
json.dump(verified, open(ROOT + '/audit_20260924_latest/verified_merge_q1.json', 'w'), indent=1)
with open(ROOT + '/audit_20260924_latest/verified_merge_q1.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['case', 'posthoc_sp', 'official_mk', 'slots',
                                      'verified_sp', 'gain', 'status'])
    w.writeheader()
    for c, r in rows.items():
        w.writerow({'case': c, 'posthoc_sp': r.get('posthoc_sp', round(post[c], 6)),
                    'official_mk': r.get('official_mk', ''), 'slots': r.get('slots', ''),
                    'verified_sp': r.get('verified_sp', round(post[c], 6)),
                    'gain': r.get('gain', 0.0), 'status': r['status']})
print(f'== q1 N5: posthoc {tot_old/n:.4f} -> verified {tot_new/n:.4f} '
      f'(+{(tot_new-tot_old):.2f} 均值sp) ==')
