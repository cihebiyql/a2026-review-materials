# 最终交付表生成器: 三问逐例官方重评(题面提交格式)
# q1/q2: case, makespan, added_copy_bytes
# q3: 冠军方案 × (无L2=ev_p2, 有Cache=ev_p3) 两配置 + cache hit_rate
import sys, json, os, glob, csv
sys.path.insert(0, r'C:/shumo_live/02_求解/A题_2026/fast_eval')
sys.path.insert(0, r'C:/shumo_live/02_求解/A题_2026/v3_solver')
sys.path.insert(0, r'C:/shumo_live/a_data/code')
from common import load_case, ev_p1, ev_p2, ev_p3

ROOT = r'C:/shumo_live/02_求解/A题_2026'
N5 = ROOT + '/n5_push'
AUD = ROOT + '/audit_20260924_latest'
OUT = AUD + '/deliverable'
os.makedirs(OUT, exist_ok=True)


def champion(case, q):
    pats = []
    if q == 1:
        pats += [f'{N5}/p1_refine_out/{case}_q1_refined.json', f'{N5}/p1_seeds/{case}_seed.json',
                 f'{N5}/pull_plans_q1/{case}_q1_r2.json', f'{N5}/pull_plans_q1/{case}_q1_pull.json']
    else:
        pats += [f'{N5}/superlinear_analysis/{case}_q{q}_mechA.json',
                 f'{N5}/node1_mechA_out/{case}_q{q}_mechA.json',
                 f'{N5}/split_probe_out/{case}_q{q}_best.json']
        for sub in ('refined2', 'refined', 'strand_n5'):
            pats.append(f'{N5}/{sub}/{case}_q{q}_N5.json')
        pats.append(f'{N5}/pull_plans/{case}_q{q}_pull.json')
    best = (0.0, None)
    pool = json.load(open(f'{AUD}/verified_merge_q1.json')) if q == 1 else            json.load(open(f'{AUD}/final_preview_q{q}.json'))
    for pat in pats:
        for fp in glob.glob(pat):
            try:
                d = json.load(open(fp))
            except Exception:
                continue
            sp = d.get('sp') or pool.get(case, 0)   # 无 sp 的种子用池值兜底
            plan = d['plan'] if 'plan' in d else (d if 'core_schedules' in d else None)
            if sp > best[0] and plan is not None:
                best = (sp, plan)
    return best[1]


def one(case):
    rows = {}
    for q, ev in ((1, ev_p1), (2, ev_p2), (3, ev_p3)):
        plan = champion(case, q)
        if plan is None:
            rows[q] = None
            continue
        g = load_case(case)
        r, _ = ev(g, plan)
        dm = r.get('data_movement_bytes', {})
        rows[q] = {'mk': r['makespan'], 'added': dm.get('added_copy_bytes'),
                   'hit_rate': (r.get('cache_stats') or {}).get('hit_rate')}
        if q == 3:
            # 无 L2 配置: 同方案在 ev_p2 评
            r2, _ = ev_p2(g, plan)
            dm2 = r2.get('data_movement_bytes', {})
            rows[3]['mk_nol2'] = r2['makespan']
            rows[3]['added_nol2'] = dm2.get('added_copy_bytes')
    return rows


if __name__ == '__main__':
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    out_jsonl = f'{OUT}/rows.jsonl'
    done = set()
    if os.path.exists(out_jsonl):
        for line in open(out_jsonl, encoding='utf-8'):
            try:
                done.add(json.loads(line)['case'])
            except Exception:
                pass
    for i in range(start, 101):
        case = f'case_{i:03d}'
        if case in done:
            continue
        try:
            rows = one(case)
        except Exception as e:
            rows = {'status': 'ERR:' + str(e)[:60]}
        with open(out_jsonl, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'case': case, 'rows': rows}, ensure_ascii=False, default=str) + '\n')
        print(case, 'ok' if rows.get(1) or rows.get(2) or rows.get(3) else rows.get('status'), flush=True)
    print('TABLE_DONE', flush=True)
