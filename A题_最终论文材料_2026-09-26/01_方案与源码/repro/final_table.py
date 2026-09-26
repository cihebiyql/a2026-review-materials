# final_table.py — 终版论文总表(18列用户指定格式)
# 本程序及代码是在人工智能工具辅助下完成的。
# 人工智能工具名称：GLM，版本/型号：zai-api/GLM-5.3，开发机构/公司：智谱AI(Z.ai)，版本颁布日期：2026-09-26
# 行来源 = 两版本并集:
#   version=final  : 冠军池逐用例最终方案(N=5, algorithm=offline_calibrated_v5)
#   version=online : 在线求解器全解(N=2..5, algorithm=v5_two_stage_online)
# 列: case,problem,cores,version,makespan,singlecore_makespan,speedup,
#     added_copy_bytes,partition_added_copy_bytes,spill_added_copy_bytes,
#     cache_hit_rate,mk_nol2,added_nol2_bytes,cache_speedup,
#     official_seconds,solve_seconds,algorithm
# 用法(建议 SLURM): python final_table.py  (读环境变量 A2026_*)
import sys, os, json, glob, csv, subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get('A2026_ROOT', r'C:/shumo_live/02_求解/A题_2026')
SA = ROOT + '/n5_push/superlinear_analysis'
CO = os.environ.get('A2026_COUT', HERE + '/compliant_out')
SC_DIR = os.environ.get('A2026_SC', ROOT + '/results/singlecore')
OUT = os.environ.get('A2026_FINALTAB', HERE + '/final_table_out')
PY = os.environ.get('V5PY', sys.executable)
os.makedirs(OUT, exist_ok=True)

sc = {}
for i in range(1, 101):
    c = f'case_{i:03d}'
    sc[c] = json.load(open(f'{SC_DIR}/{c}_sc.json'))['makespan']

groups = defaultdict(list)      # (case,Q) -> [(version,cores,solve_s,plan)]
for q in (1, 2, 3):
    for fp in sorted(glob.glob(f'{SA}/case_*_q{q}_mechA.json')):
        case = os.path.basename(fp).split('_q')[0]
        groups[(case, q)].append(('final', 5, None, json.load(open(fp))['plan']))
for fp in sorted(glob.glob(f'{CO}/*_q*_N*_compliant.json')):
    b = os.path.basename(fp)[:-5]
    parts = b.split('_')
    case, q, n = '_'.join(parts[:2]), int(parts[2][1:]), int(parts[3][1:])
    d = json.load(open(fp))
    if d['rec'].get('tag'):
        continue
    groups[(case, q)].append(('online', n, d['rec'].get('wall_s'), d['plan']))


def eval_group(key):
    case, q = key
    req = {'case': case, 'Q': q, 'plans': [p for *_, p in groups[key]],
           'time_official': False}
    rows = None
    for mode in (False, True):          # fast 优先, 崩溃组官方兜底(带计时)
        req['time_official'] = mode
        try:
            pr = subprocess.run([PY, f'{HERE}/final_worker.py'],
                                input=json.dumps(req), capture_output=True,
                                text=True, timeout=3600)
            if pr.returncode == 0:
                rows = json.loads(pr.stdout.strip().splitlines()[-1])
                break
        except Exception:
            pass
    return (key, rows or [None] * len(groups[key]))


rows_out = []
with ThreadPoolExecutor(int(os.environ.get('FT_WORKERS', '12'))) as ex:
    for (case, q), res in ex.map(eval_group, sorted(groups)):
        for (ver, cores, solve_s, _p), r in zip(groups[(case, q)], res):
            r = r or {}
            mk = r.get('mk')
            if not mk:
                continue
            sp = sc[case] / mk
            rows_out.append({
                'case': case, 'problem': q, 'cores': cores, 'version': ver,
                'makespan': mk, 'singlecore_makespan': sc[case],
                'speedup': round(sp, 6),
                'added_copy_bytes': r.get('added'),
                'partition_added_copy_bytes': r.get('partition'),
                'spill_added_copy_bytes': r.get('spill'),
                'cache_hit_rate': (round(r['cache_hit_rate'], 6)
                                   if isinstance(r.get('cache_hit_rate'), (int, float))
                                   else r.get('cache_hit_rate')),
                'mk_nol2': r.get('mk_nol2'),
                'added_nol2_bytes': r.get('added_nol2'),
                'cache_speedup': (round(r['mk_nol2'] / mk, 6)
                                  if r.get('mk_nol2') else ''),
                'official_seconds': r.get('official_seconds'),
                'solve_seconds': solve_s,
                'algorithm': ('offline_calibrated_v5' if ver == 'final'
                              else 'v5_two_stage_online'),
            })

COLS = ['case', 'problem', 'cores', 'version', 'makespan', 'singlecore_makespan',
        'speedup', 'added_copy_bytes', 'partition_added_copy_bytes',
        'spill_added_copy_bytes', 'cache_hit_rate', 'mk_nol2',
        'added_nol2_bytes', 'cache_speedup', 'official_seconds',
        'solve_seconds', 'algorithm']
fp = f'{OUT}/final_results_table.csv'
with open(fp, 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=COLS)
    w.writeheader()
    w.writerows(sorted(rows_out, key=lambda r: (r['case'], r['problem'],
                                                r['version'], r['cores'])))
print(f'rows={len(rows_out)} -> {fp}')
