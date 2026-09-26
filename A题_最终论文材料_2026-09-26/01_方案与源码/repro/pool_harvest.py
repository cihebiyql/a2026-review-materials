# pool_harvest.py — 冠军池(A题 300 个 N5 最终方案)次指标收割: 额外搬运量 + P3 Cache命中率
# 本程序及代码是在人工智能工具辅助下完成的。
# 人工智能工具名称：GLM，版本/型号：zai-api/GLM-5.3，开发机构/公司：智谱AI(Z.ai)，版本颁布日期：2026-09-26
#
# 冠军池定义: 每 (case,Q) 取 池目录(远程 seedsync 全量) 与 本地上传目录(local_pool) 中
#             记录 mk 更小者 —— best-of 合并,共 100用例 × 3问 = 300 方案(全部 N=5)。
# 评估: 逐 (case,Q) 子进程 pool_worker.py(同合规收割协议);
#       原生崩溃组(retcode!=0)自动改用 官方评估器(official=true) 兜底重评。
# 输出(OUT 目录, 列结构与 harvest_out/appendix_problem*.csv 完全一致):
#   pool_appendix_problem1.csv  用例,N,Makespan(cycles),额外搬运量(bytes)
#   pool_appendix_problem2.csv  同上
#   pool_appendix_problem3.csv  用例,N,Makespan_无L2,额外搬运_无L2,Makespan_只读Cache,
#                               额外搬运_只读Cache,Cache命中率(次数),Cache命中率(字节),
#                               相对加速比(无L2/Cache)     [无L2列=q2池方案, Cache列=q3池方案, 按用例join]
#   pool_summary.json           三问 N=5 平均 sp + 崩溃兜底明细 + 池来源统计 + 复核差异
# 用法: python pool_harvest.py   (读 A2026_ROOT/A2026_SC/A2026_POOL_OUT 环境变量)
import sys, os, json, csv, glob, subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get('A2026_ROOT', r'C:/shumo_live/02_求解/A题_2026')
SC_DIR = os.environ.get('A2026_SC', ROOT + '/results/singlecore')
POOL_DIR = os.environ.get('A2026_POOL_DIR', HERE + '/superlinear_analysis')
LOCAL_DIR = os.environ.get('A2026_POOL_LOCAL', POOL_DIR + '/local_pool')
OUT = os.environ.get('A2026_POOL_OUT', HERE + '/pool_out')
PY = os.environ.get('V5PY', sys.executable)
N = 5
os.makedirs(OUT, exist_ok=True)

CASES = [f'case_{i:03d}' for i in range(1, 101)]


def pick_pool(case, Q):
    """收集 (case,Q) 的全部候选(远程池目录 + 本地上传目录), 按记录 mk 升序。
    远程 seedsync 记录 mk 存在少量与 plan 不一致的组(重评对不上),
    由 eval_group 按"先验证先取"消解。"""
    cands = []
    for src, d in (('remote', POOL_DIR), ('local', LOCAL_DIR)):
        fp = f'{d}/{case}_q{Q}_mechA.json'
        if os.path.exists(fp):
            try:
                dj = json.load(open(fp))
                if dj.get('mk'):
                    cands.append({'mk_rec': int(dj['mk']), 'src': src,
                                  'plan': dj['plan']})
            except Exception as e:
                print(f'[warn] bad json {fp}: {e}', flush=True)
    cands.sort(key=lambda t: t['mk_rec'])
    return cands


pool = {}          # (case,Q) -> [cand, ...] (按记录 mk 升序)
missing = []
for case in CASES:
    for Q in (1, 2, 3):
        r = pick_pool(case, Q)
        if not r:
            missing.append((case, Q))
        else:
            pool[(case, Q)] = r
print(f'pool selected: {len(pool)} groups, missing={missing}', flush=True)

sc = {}
for case in CASES:
    sc[case] = json.load(open(f'{SC_DIR}/{case}_sc.json'))['makespan']


def run_worker(case, Q, plans, official, timeout):
    req = json.dumps({'case': case, 'Q': Q, 'plans': plans,
                      'official': bool(official)})
    pr = subprocess.run([PY, f'{HERE}/pool_worker.py'], input=req,
                        capture_output=True, text=True, timeout=timeout,
                        env=dict(os.environ))
    if pr.returncode != 0:
        raise RuntimeError(f'retcode={pr.returncode} stderr={pr.stderr[-400:]}')
    return json.loads(pr.stdout.strip().splitlines()[-1])


def eval_group(key):
    """逐候选取"记录mk最小且重评可复现"者为该组最终方案;
    全部不可复现时取重评 mk 最小者(诚实值, 计入 chosen_unverified)。"""
    case, Q = key
    cands = pool[key]
    rec = {'case': case, 'Q': Q, 'N': N}
    try:
        res = None
        try:
            res = run_worker(case, Q, [c['plan'] for c in cands],
                             official=False, timeout=1800)
            rec['eval'] = 'fast'
        except Exception as e:                      # 原生崩溃 → 官方评估器兜底
            rec['fast_err'] = str(e)[:300]
            try:
                res = run_worker(case, Q, [c['plan'] for c in cands],
                                 official=True, timeout=3000)
                rec['eval'] = 'official'
            except Exception as e2:
                rec['official_err'] = str(e2)[:300]
                rec['eval'] = 'failed'
        if not res:
            raise RuntimeError(rec.get('official_err', rec.get('fast_err', '')))
        # 先验证先取; 全不匹配则取重评最小
        chosen = None
        for i, c in enumerate(cands):
            r0 = res[i] if i < len(res) else None
            if r0 and r0['mk'] == c['mk_rec']:
                chosen = (c, r0)
                break
        if chosen is None:
            ok = [(c, res[i]) for i, c in enumerate(cands)
                  if i < len(res) and res[i].get('mk')]
            if ok:
                chosen = min(ok, key=lambda t: t[1]['mk'])
                rec['unverified'] = True
        if chosen:
            c, r0 = chosen
            rec.update({'src': c['src'], 'mk_rec': c['mk_rec'],
                        'mk': r0['mk'], 'info': r0.get('info') or {},
                        'eval': r0.get('eval', rec['eval'])})
        else:
            rec.update({'src': cands[0]['src'], 'mk_rec': cands[0]['mk_rec'],
                        'mk': None, 'info': {}, 'eval': 'failed'})
    except Exception as e:                          # 理论兜底(run_worker超时等)
        rec['eval'] = 'failed'
        rec['official_err'] = str(e)[:300]
        rec.update({'src': cands[0]['src'], 'mk_rec': cands[0]['mk_rec'],
                    'mk': None, 'info': {}})
    return rec


rows = []
with ThreadPoolExecutor(14) as ex:
    for rec in ex.map(eval_group, sorted(pool)):
        rows.append(rec)
        tag = '' if rec['mk'] == rec['mk_rec'] else ' MK-DIFF'
        print(f"{rec['case']} q{rec['Q']} src={rec['src']} eval={rec['eval']} "
              f"mk={rec['mk']}/{rec['mk_rec']}{tag}", flush=True)

byk = {(r['case'], r['Q']): r for r in rows}
verify_bad = [f"{r['case']}_q{r['Q']}" for r in rows
              if r['mk'] is not None and r['mk'] != r['mk_rec']]
discarded_top = [f"{r['case']}_q{r['Q']}" for r in rows
                 if r.get('unverified') or r.get('fast_err')]
crash_fast = sorted(f"{r['case']}_q{r['Q']}" for r in rows if r.get('fast_err'))
official_ok = sorted(f"{r['case']}_q{r['Q']}" for r in rows if r['eval'] == 'official')
failed = sorted(f"{r['case']}_q{r['Q']}" for r in rows if r['eval'] == 'failed')
src_cnt = defaultdict(int)
for r in rows:
    src_cnt[r['src']] += 1

# ---- 附录 P1/P2(列结构与 harvest_out/appendix_problem{1,2}.csv 一致) ----
for Q in (1, 2):
    with open(f'{OUT}/pool_appendix_problem{Q}.csv', 'w', newline='',
              encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['用例', 'N', 'Makespan(cycles)', '额外搬运量(bytes)'])
        for case in CASES:
            r = byk.get((case, Q))
            if not r:
                continue
            w.writerow([case, N, r['mk'] if r['mk'] is not None else '',
                        r['info'].get('added_copy_bytes',
                                      r['info'].get('spill_added_copy_bytes', ''))])

# ---- 附录 P3(无L2列=q2池方案, Cache列=q3池方案, 同用例join) ----
with open(f'{OUT}/pool_appendix_problem3.csv', 'w', newline='',
          encoding='utf-8-sig') as f:
    w = csv.writer(f)
    w.writerow(['用例', 'N', 'Makespan_无L2(cycles)', '额外搬运_无L2(bytes)',
                'Makespan_只读Cache(cycles)', '额外搬运_只读Cache(bytes)',
                'Cache命中率(次数)', 'Cache命中率(字节)', '相对加速比(无L2/Cache)'])
    for case in CASES:
        r2, r3 = byk.get((case, 2)), byk.get((case, 3))
        if not r3:
            continue
        cs = r3['info'].get('cache_stats', {}) or {}
        hit_cnt = cs.get('hit_cnt_rate', '')
        hit_byte = cs.get('hit_rate', cs.get('hit_byte_rate', ''))
        rel = (r2['mk'] / r3['mk']) if (r2 and r2['mk'] and r3['mk']) else ''
        w.writerow([case, N,
                    r2['mk'] if r2 and r2['mk'] is not None else '',
                    r2['info'].get('added_copy_bytes', '') if r2 else '',
                    r3['mk'] if r3['mk'] is not None else '',
                    r3['info'].get('added_copy_bytes', ''),
                    hit_cnt, hit_byte, rel])

# ---- 汇总(三问 N=5 平均 sp, 口径=单核基准/池方案mk 的算术平均) ----
avg_sp = {}
for Q in (1, 2, 3):
    sps = [sc[r['case']] / r['mk'] for r in rows
           if r['Q'] == Q and r['mk']]
    avg_sp[f'q{Q}_N5'] = round(sum(sps) / len(sps), 4) if sps else None
summary = {
    'rows': len(rows), 'N': N, 'avg_sp': avg_sp,
    'pool_source_counts': dict(src_cnt),
    'verify_bad': verify_bad,
    'crash_fast_fallback': {'fast_crash_groups': crash_fast,
                            'official_ok': official_ok, 'failed': failed},
    'candidate_policy': ('每(case,Q)评估远程池与本地上传两候选, 取"记录mk最小且重评'
                         '可复现"者; 若均不可复现取重评mk最小者(计入verify_bad); '
                         '原生崩溃组由官方评估器兜底重评'),
    'groups_with_discarded_or_fallback_eval': sorted(set(discarded_top)),
    'missing_groups': missing,
    'mk_note': 'mk列为重评值(fast/official双评估器口径一致, 见pool_worker.py)',
}
json.dump(summary, open(f'{OUT}/pool_summary.json', 'w'),
          ensure_ascii=False, indent=1)
print(json.dumps(summary, ensure_ascii=False), flush=True)
