# v5_ablation.py — 有效性评价与消融实验批(论文实验章数据源)
# 本程序及代码是在人工智能工具辅助下完成的。
# 人工智能工具名称：GLM，版本/型号：zai-api/GLM-5.3，开发机构/公司：智谱AI(Z.ai)，版本颁布日期：2026-09-26
#
# 实验矩阵(全部为在线求解端消融, 复用已训练 logits, 不改训练):
#   A 采样数消融   : greedy(0) / S16 / S64 / S256 (关精修, 隔离采样作用)
#   B 学习先验消融 : trained / uniform(无先验) / rank(min_id序) (S64, 关精修)
#   C 局部精修消融 : polish on/off (S256)
#   D 解码器项消融 : 默认 / w_traffic=0 / w_spread=0 (S64+精修, 短预算)
#   E 多种子方差   : seed 1/2/3 (完整配置)
# 用法: python v5_ablation.py <Q> [--cases ...] [--workers 16]
# 输出: ABL_OUT/ablation_q{Q}.jsonl (每行含 tag 便于汇总)
import sys, os, json, time, argparse, subprocess
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.environ.get('V5PY', sys.executable)
ABL_OUT = os.environ.get('A2026_ABL', HERE + '/ablation_out')
os.makedirs(ABL_OUT, exist_ok=True)

REP_CASES = ['case_084', 'case_095', 'case_041', 'case_038', 'case_087',
             'case_034', 'case_072', 'case_076', 'case_029', 'case_064',
             'case_002', 'case_031']


def configs():
    out = []
    for s in (0, 16, 64, 256):                       # A 采样数
        out.append((f'A_samples{s}', ['--samples', str(s), '--no-polish',
                                      '--budget_s', '600']))
    for m in ('trained', 'uniform', 'rank'):         # B 学习先验
        if m != 'trained':
            out.append((f'B_{m}', ['--logits-mode', m, '--samples', '64',
                                   '--no-polish', '--budget_s', '600']))
    out.append(('C_polish', ['--samples', '256', '--budget_s', '480']))  # C(vs 主产线 no-polish 对照)
    for d in ('w_traffic', 'w_spread'):              # D 解码器项
        out.append((f'D_{d}0', ['--samples', '64', '--budget_s', '300'],
                    {d: 0.0}))
    for sd in (1, 2, 3):                             # E 多种子
        out.append((f'E_seed{sd}', ['--samples', '256', '--budget_s', '480',
                                    '--seed', str(sd)]))
    return out


def run_one(job):
    case, Q, tag, args, deco = job
    outl = f'{ABL_OUT}/ablation_q{Q}.jsonl'
    env = dict(os.environ)
    if deco:
        env['A2026_DECO_PARAMS'] = json.dumps(deco)
    cmd = [PY, f'{HERE}/solve_compliant.py', case, str(Q), '5',
           '--tag', tag] + args
    try:
        p = subprocess.run(cmd, cwd=HERE, env=env, timeout=1200,
                           capture_output=True, text=True)
        for line in p.stdout.splitlines():
            if line.startswith('COMPLIANT '):
                rec = json.loads(line[len('COMPLIANT '):])
                rec['tag'] = tag
                with open(outl, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                return True
    except Exception:
        pass
    with open(outl, 'a', encoding='utf-8') as f:
        f.write(json.dumps({'case': case, 'Q': Q, 'tag': tag,
                            'status': 'FAIL'}, ensure_ascii=False) + '\n')
    return False


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('Q', type=int)
    ap.add_argument('--cases', default='')
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    cases = a.cases.split(',') if a.cases else REP_CASES
    jobs = []
    for tag_spec in configs():
        tag, args = tag_spec[0], tag_spec[1]
        deco = tag_spec[2] if len(tag_spec) > 2 else None
        for c in cases:
            lg = os.environ.get('A2026_LOGITS', HERE + '/reinforce_v5') \
                 + f'/{c}_q{a.Q}_logits.json'
            if os.path.exists(lg):
                jobs.append((c, a.Q, tag, args, deco))
    print(f'ablation jobs: {len(jobs)} (Q={a.Q}, {len(cases)} cases)', flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        ok = sum(ex.map(run_one, jobs))
    print(f'ABLATION_DONE ok={ok}/{len(jobs)} '
          f'wall={(time.time()-t0)/60:.1f}min', flush=True)
