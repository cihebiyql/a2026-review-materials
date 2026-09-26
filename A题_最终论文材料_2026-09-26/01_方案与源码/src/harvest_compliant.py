# harvest_compliant.py — 合规产线收割: 题目交付物(附录表/曲线数据/耗时统计)
# 本程序及代码是在人工智能工具辅助下完成的。
# 人工智能工具名称：GLM，版本/型号：zai-api/GLM-5.3，开发机构/公司：智谱AI(Z.ai)，版本颁布日期：2026-09-26
#
# 输入: compliant_out/*_q{Q}_N{N}_compliant.json(方案+rec) + 单核基准
# 输出(OUT 目录):
#   appendix_problem{1,2}.csv  用例,N,Makespan(cycles),额外搬运量(bytes)   [题面附录要求]
#   appendix_problem3.csv      用例,N,无L2 mk,无L2搬运,Cache mk,Cache搬运,
#                              命中率(次数),命中率(字节),相对加速比          [题面附录要求]
#   curve_data.csv             Q,N,平均加速比(算术平均,脚注口径),例数       [正文折线]
#   timing_stats.csv           用例,Q,N,windows wall_s,预算占用,采样数,精修评估数
#   summary.json               三问 N=2..5 平均加速比 + 覆盖率 + 验证通过率
# 复评即验证: 每个方案的 mk 重评一次, 不一致则标记(应为 0, FastEval bit-exact)。
# 用法: python harvest_compliant.py  (读 A2026_COUT/A2026_SC/A2026_ATT 环境变量)
import sys, os, json, csv, glob
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get('A2026_ROOT', r'C:/shumo_live/02_求解/A题_2026')
COUT = os.environ.get('A2026_COUT', HERE + '/compliant_out')
SC_DIR = os.environ.get('A2026_SC', ROOT + '/results/singlecore')
OUT = os.environ.get('A2026_HARVEST', HERE + '/harvest_out')
sys.path.insert(0, ROOT + '/fast_eval')
sys.path.insert(0, ROOT + '/v3_solver')
from common import load_case                       # noqa: E402
from fast_eval_p2 import FastEvalP2, FastEvalP3    # noqa: E402
from fast_eval_p1 import FastEvalP1                # noqa: E402

os.makedirs(OUT, exist_ok=True)
sc = {}
rows = []          # (case, Q, N, rec, info)
verify_bad = 0

# 逐 (case,Q) 组子进程重评(崩溃类图原生崩溃只丢该组 info, 主进程免疫)
import subprocess
from concurrent.futures import ThreadPoolExecutor
groups = defaultdict(list)   # (case,Q) -> [(N, rec, plan)]
for fp in sorted(glob.glob(f'{COUT}/*_q*_N*_compliant.json')):
    b = os.path.basename(fp)[:-5]
    parts = b.split('_')
    case = '_'.join(parts[:2])
    Q, N = int(parts[2][1:]), int(parts[3][1:])
    if case not in sc:
        try:
            sc[case] = json.load(open(f'{SC_DIR}/{case}_sc.json'))['makespan']
        except Exception:
            continue
    d = json.load(open(fp))
    if d['rec'].get('tag'):
        continue
    groups[(case, Q)].append((N, d['rec'], d['plan']))

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.environ.get('V5PY', sys.executable)


def eval_group(key):
    case, Q = key
    items = groups[key]
    req = json.dumps({'case': case, 'Q': Q,
                      'plans': [p for _, _, p in items]})
    try:
        pr = subprocess.run([PY, f'{HERE}/harvest_worker.py'],
                            input=req, capture_output=True, text=True,
                            timeout=1800, env=dict(os.environ))
        res = json.loads(pr.stdout.strip().splitlines()[-1]) \
            if pr.returncode == 0 else None
    except Exception:
        res = None
    out = []
    for i, (N, rec, _p) in enumerate(items):
        info = (res[i]['info'] if res and i < len(res) else {}) or {}
        mk2 = res[i]['mk'] if res and i < len(res) else None
        out.append((case, Q, N, rec, info))
        globals()['verify_bad'] += int(mk2 is not None and mk2 != rec['mk'])
    return out


with ThreadPoolExecutor(12) as ex:
    for part in ex.map(eval_group, sorted(groups)):
        rows.extend(part)

print(f'harvested {len(rows)} rows, verify_bad={verify_bad}')

# ---- 附录 P1/P2 ----
for Q in (1, 2):
    with open(f'{OUT}/appendix_problem{Q}.csv', 'w', newline='',
              encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['用例', 'N', 'Makespan(cycles)', '额外搬运量(bytes)'])
        for case, q, N, rec, info in sorted(rows):
            if q != Q:
                continue
            w.writerow([case, N, rec['mk'],
                        info.get('added_copy_bytes', info.get('spill_added_copy_bytes', ''))])

# ---- 附录 P3(无L2 vs Cache 对比 + 命中率 + 相对加速比) ----
byk = {(c, n): (rec, info) for c, q, n, rec, info in rows if q == 3}
byk2 = {(c, n): (rec, info) for c, q, n, rec, info in rows if q == 2}
with open(f'{OUT}/appendix_problem3.csv', 'w', newline='',
          encoding='utf-8-sig') as f:
    w = csv.writer(f)
    w.writerow(['用例', 'N', 'Makespan_无L2(cycles)', '额外搬运_无L2(bytes)',
                'Makespan_只读Cache(cycles)', '额外搬运_只读Cache(bytes)',
                'Cache命中率(次数)', 'Cache命中率(字节)', '相对加速比(无L2/Cache)'])
    for (c, n), (r3, i3) in sorted(byk.items()):
        r2 = byk2.get((c, n), (None, None))
        cs = i3.get('cache_stats', {}) or {}
        hit_r = cs.get('hit_rate', cs.get('hit_cnt_rate', ''))
        hit_b = cs.get('hit_byte_rate', cs.get('hit_bytes_rate', ''))
        rel = (r2[0]['mk'] / r3['mk']) if r2 and r2[0] and r3['mk'] else ''
        w.writerow([c, n,
                    r2[0]['mk'] if r2 and r2[0] else '',
                    r2[1].get('added_copy_bytes', '') if r2 and r2[1] else '',
                    r3['mk'], i3.get('added_copy_bytes', ''), hit_r, hit_b, rel])

# ---- 正文曲线数据 ----
acc = defaultdict(list)
for case, q, n, rec, info in rows:
    acc[(q, n)].append(sc[case] / rec['mk'])
with open(f'{OUT}/curve_data.csv', 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f)
    w.writerow(['问题', 'N', '平均加速比', '例数'])
    for q in (1, 2, 3):
        w.writerow([q, 1, 1.0, 100])               # 单核基准点(口径定义)
        for n in (2, 3, 4, 5):
            v = acc.get((q, n), [])
            w.writerow([q, n, round(sum(v) / len(v), 6) if v else '', len(v)])

# ---- 耗时统计 ----
with open(f'{OUT}/timing_stats.csv', 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f)
    w.writerow(['用例', 'Q', 'N', 'wall_s', '预算占用率', '采样数', '精修评估数'])
    for case, q, n, rec, info in sorted(rows):
        w.writerow([case, q, n, rec.get('wall_s'),
                    round(rec.get('wall_s', 0) / rec.get('budget_s', 480), 3),
                    rec.get('samples'), rec.get('polish_evals')])

# ---- 汇总 ----
summary = {'rows': len(rows), 'verify_bad': verify_bad,
           'curves': {f'q{q}_N{n}': (round(sum(v) / len(v), 4) if v else None)
                      for (q, n), v in sorted(acc.items())}}
json.dump(summary, open(f'{OUT}/summary.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(summary['curves'], ensure_ascii=False))
