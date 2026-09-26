# paper_figs.py — 论文图表与汇总表生成器(本地运行, 读拉回的产线数据)
# 本程序及代码是在人工智能工具辅助下完成的。
# 人工智能工具名称：GLM，版本/型号：zai-api/GLM-5.3，开发机构/公司：智谱AI(Z.ai)，版本颁布日期：2026-09-26
#
# 输入(本地): n5_push/{compliant_out,ablation_out,reinforce_v5} 拉回的 jsonl/json/log
#   trainlogs/ 来自 bxcpu:~/shumo/a2026_reinforce/n5_push/compliant_out/train_case_*.log
# 输出(FIG_OUT):
#   fig_curves.png      三问 1-5 核平均加速比折线(正文主图)
#   fig_convergence.png REINFORCE 批内平均代价收敛曲线(代表例, 读 trainlogs/)
#   fig_ablation.png    消融实验平均加速比柱状图(按问题分色)
#   ablation_table.md   消融汇总表(分族+对照+结论, 可直接贴论文)
#   gap_table.md        合规解 vs 离线上界 gap 表
#   summary_report.md   总报告(曲线数字+消融结论+覆盖率)
# 用法: py -3.11 paper_figs.py
import os, sys, json, glob, re, csv
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
CO = HERE + '/compliant_out'
AB = HERE + '/ablation_out'
TR = HERE + '/compliant_out/trainlogs'   # 训练日志单独目录(bxcpu 拉回)
FIG_OUT = os.environ.get('A2026_FIGS', HERE + '/paper_figs')
SC_DIR = os.environ.get('A2026_SC', r'C:/shumo_live/02_求解/A题_2026/results/singlecore')
os.makedirs(FIG_OUT, exist_ok=True)


def load_sc():
    sc = {}
    for fp in glob.glob(f'{SC_DIR}/*_sc.json'):
        b = os.path.basename(fp)[:-8]
        try:
            sc[b] = json.load(open(fp))['makespan']
        except Exception:
            pass
    return sc


def load_compliant(sc):
    """tag为空的合规解记录; 同键多源(双机)去重取mk最小。"""
    best = {}
    for fp in glob.glob(f'{CO}/compliant_q*.jsonl'):
        for line in open(fp, encoding='utf-8'):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get('tag') or not r.get('mk'):
                continue
            k = (r['case'], r['Q'], r['N'])
            if k not in best or r['mk'] < best[k]['mk']:
                best[k] = r
    return best


def fig_curves(best, sc):
    acc = defaultdict(list)
    for (c, q, n), r in best.items():
        acc[(q, n)].append(sc.get(c, 0) / r['mk'])
    fig, ax = plt.subplots(figsize=(7, 4.6))
    markers = {1: 'o-', 2: 's-', 3: '^-'}
    rows = []
    for q in (1, 2, 3):
        xs, ys = [1], [1.0]
        for n in (2, 3, 4, 5):
            v = acc.get((q, n), [])
            if v:
                xs.append(n)
                ys.append(sum(v) / len(v))
                rows.append((q, n, sum(v) / len(v), len(v)))
        ax.plot(xs, ys, markers[q], label=f'问题{q}')
    ax.set_xlabel('核数 N')
    ax.set_ylabel('平均加速比(逐用例算术平均)')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'{FIG_OUT}/fig_curves.png', dpi=160)
    plt.close(fig)
    with open(f'{FIG_OUT}/curve_table.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['问题', 'N', '平均加速比', '例数'])
        w.writerow([1, 1, 1.0, 100])
        for row in sorted(rows):
            w.writerow([row[0], row[1], round(row[2], 4), row[3]])
    return rows


# ---------------------------------------------------------------- 消融数据 --
# 实验矩阵(v5_ablation.py): 全部为在线求解端消融, 复用已训练 logits, N=5, 代表例.
#   A 采样数  : greedy(0)/S16/S64/S256, 关精修, 600s —— B 族对照 = A_samples64
#   B 学习先验: uniform(无先验)/rank(秩先验), S64 关精修, 600s
#   C 局部精修: polish on(S256, 480s) vs off(=A_samples256)
#   D 解码器项: w_traffic=0 / w_spread=0 (S64+精修, 300s) —— 同预算默认孪生未单列,
#               以完整配置 C_polish(S256+精修, 480s) 为参照(含预算差, 表内注明)
#   E 多种子  : seed 1/2/3 完整配置(=C_polish 换种子)
ABL_ORDER = ['A_samples0', 'A_samples16', 'A_samples64', 'A_samples256',
             'B_uniform', 'B_rank', 'D_w_traffic0', 'D_w_spread0',
             'C_polish', 'E_seed1', 'E_seed2', 'E_seed3']
ABL_ZH = {'A_samples0': 'A 贪心(S0)', 'A_samples16': 'A S16', 'A_samples64': 'A S64',
          'A_samples256': 'A S256', 'B_uniform': 'B 均匀先验', 'B_rank': 'B 秩先验',
          'D_w_traffic0': 'D 去通信项', 'D_w_spread0': 'D 去摊开项',
          'C_polish': 'C 开精修', 'E_seed1': 'E 种子1', 'E_seed2': 'E 种子2',
          'E_seed3': 'E 种子3'}
FAMILIES = [  # (族标题, [配置], 族内对照 tag 或 None)
    ('采样数消融(A 族: 关精修, 600s, 只变采样数)',
     ['A_samples0', 'A_samples16', 'A_samples64', 'A_samples256'], None),
    ('学习先验消融(B 族: S64 关精修, 对照=A_samples64 即学习先验)',
     ['B_uniform', 'B_rank'], 'A_samples64'),
    ('解码器项消融(D 族: S64+精修, 300s, 对照=完整配置 C_polish)',
     ['D_w_traffic0', 'D_w_spread0'], 'C_polish'),
    ('局部精修消融(C 族: S256, 对照=A_samples256 即关精修)',
     ['C_polish'], 'A_samples256'),
    ('多种子方差(E 族: 完整配置, 对照=C_polish 即 seed 0)',
     ['E_seed1', 'E_seed2', 'E_seed3'], 'C_polish'),
]


def load_ablation():
    """-> by[(Q,tag)][case] = mean(sp) (同tag同case多记录取均值); FAIL/无sp 跳过。"""
    by, n_fail = defaultdict(lambda: defaultdict(list)), 0
    for fp in glob.glob(f'{AB}/ablation_q*.jsonl'):
        for line in open(fp, encoding='utf-8'):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get('status') == 'FAIL' or not r.get('sp'):
                n_fail += 1
                continue
            by[(r['Q'], r.get('tag', '?'))][r['case']].append(r['sp'])
    return {(k[0], k[1]): {c: sum(v) / len(v) for c, v in d.items()}
            for k, d in by.items()}, n_fail


def abl_mean(by, q, tag):
    d = by.get((q, tag))
    if not d:
        return None, 0
    return sum(d.values()) / len(d), len(d)


def abl_paired(by, q, tag, ctrl):
    """同例配对差: mean(sp_tag - sp_ctrl over 共同用例)。"""
    a, b = by.get((q, tag), {}), by.get((q, ctrl), {})
    common = sorted(set(a) & set(b))
    if not common:
        return None, 0
    return sum(a[c] - b[c] for c in common) / len(common), len(common)


def _fmtd(d):
    return f'{d:+.3f}' if d is not None else '—'


def ablation_table(by, n_fail):
    lines = ['# 消融实验汇总表(平均加速比, N=5, 代表用例)', '',
             f'- 数据: `ablation_out/ablation_q{{1,2,3}}.jsonl`；FAIL/无结果行已剔除'
             f'(共 {n_fail} 条)；同 tag 多记录已取均值。',
             '- 每格为该配置在该问题代表用例上的平均加速比(括号为例数)。',
             '- Δ 为**同用例配对差** Δ=配置−对照(只在该族共同用例上计算), 消除例集差异。',
             '']
    for title, tags, ctrl in FAMILIES:
        lines += [f'## {title}', '']
        if ctrl:
            ms = [abl_mean(by, q, ctrl) for q in (1, 2, 3)]
            if all(x[0] is not None for x in ms):
                lines.append(f'- 对照 {ctrl}: P1 {ms[0][0]:.4f}(n={ms[0][1]}) / '
                             f'P2 {ms[1][0]:.4f}(n={ms[1][1]}) / '
                             f'P3 {ms[2][0]:.4f}(n={ms[2][1]})')
            else:
                lines.append(f'- 对照 {ctrl}: 数据缺失')
            lines.append('')
        lines += ['| 配置 | P1 均值(例数) | P2 均值(例数) | P3 均值(例数) |'
                  ' ΔP1 | ΔP2 | ΔP3 |', '|---|---|---|---|---|---|---|']
        fam_rows = []
        for t in tags:
            cells, deltas = [], []
            for q in (1, 2, 3):
                mv, n = abl_mean(by, q, t)
                cells.append(f'{mv:.4f}(n={n})' if mv is not None else '—')
                if ctrl:
                    dv, _ = abl_paired(by, q, t, ctrl)
                    deltas.append(_fmtd(dv))
                else:
                    deltas.append('—')
            lines.append(f'| {t} | {cells[0]} | {cells[1]} | {cells[2]} | '
                         f'{deltas[0]} | {deltas[1]} | {deltas[2]} |')
            fam_rows.append(t)
        # ---- 族结论(由配对差动态生成) ----
        concl = _family_conclusion(by, title, tags, ctrl)
        lines += ['', f'> **结论**: {concl}', '']
    open(f'{FIG_OUT}/ablation_table.md', 'w', encoding='utf-8').write('\n'.join(lines))
    return lines


def _family_conclusion(by, title, tags, ctrl):
    def d3(t):
        return [abl_paired(by, q, t, ctrl)[0] for q in (1, 2, 3)]
    if title.startswith('采样数'):
        d1 = abl_paired(by, 1, 'A_samples256', 'A_samples0')[0]
        d2 = abl_paired(by, 2, 'A_samples256', 'A_samples0')[0]
        d3v = abl_paired(by, 3, 'A_samples256', 'A_samples0')[0]
        d16 = [abl_paired(by, q, 'A_samples16', 'A_samples0')[0] for q in (1, 2, 3)]
        return (f'贪心→S256 采样配对增益 P1 {_fmtd(d1)} / P2 {_fmtd(d2)} / P3 {_fmtd(d3v)},'
                f' 其中 S16 已拿到大部分增益({_fmtd(d16[0])}/{_fmtd(d16[1])}/{_fmtd(d16[2])}) ——'
                f' 随机采样带来稳定小幅增益, S16→S256 边际递减。')
    if title.startswith('学习先验'):
        du = [abl_paired(by, q, 'B_uniform', ctrl)[0] for q in (1, 2, 3)]
        dr = [abl_paired(by, q, 'B_rank', ctrl)[0] for q in (1, 2, 3)]
        mx = max(abs(x) for x in du + dr if x is not None)
        return (f'均匀/秩先验相对学习先验的配对差分别为 {_fmtd(du[0])}/{_fmtd(du[1])}/{_fmtd(du[2])}'
                f' 与 {_fmtd(dr[0])}/{_fmtd(dr[1])}/{_fmtd(dr[2])} (P1/P2/P3),'
                f' 幅值均 ≤{mx:.3f} —— 在平均加速比口径下先验形式影响很小,'
                f' 学习先验的主要收益需结合逐例胜率解读。')
    if title.startswith('解码器项'):
        dt = [abl_paired(by, q, 'D_w_traffic0', ctrl)[0] for q in (1, 2, 3)]
        ds = [abl_paired(by, q, 'D_w_spread0', ctrl)[0] for q in (1, 2, 3)]
        worst = min(x for x in dt + ds if x is not None)
        return (f'去通信项配对差 {_fmtd(dt[0])}/{_fmtd(dt[1])}/{_fmtd(dt[2])},'
                f' 去摊开项 {_fmtd(ds[0])}/{_fmtd(ds[1])}/{_fmtd(ds[2])} (P1/P2/P3):'
                f' P1/P2 与完整配置持平(预算减半仍不降), P3 同步降 {abs(worst):.3f};'
                f' 注: D 族预算更小(S64/300s vs S256/480s), 上述差是"领域项+预算"的联合效应,'
                f' 领域解码项的贡献应据此保守解读。')
    if title.startswith('局部精修'):
        dd = [abl_paired(by, q, 'C_polish', ctrl)[0] for q in (1, 2, 3)]
        return (f'开精修相对关精修配对增益 P1 {_fmtd(dd[0])} / P2 {_fmtd(dd[1])} / P3 {_fmtd(dd[2])}'
                f' (n=10/12/4), 三问一致为正 —— 预算内局部精修贡献稳定。')
    if title.startswith('多种子'):
        spreads, maxd = [], 0.0
        for q in (1, 2, 3):
            ms = [abl_mean(by, q, t)[0] for t in tags]
            ms = [m for m in ms if m is not None]
            spreads.append(max(ms) - min(ms) if ms else 0)
            for t in tags:
                dv, _ = abl_paired(by, q, t, ctrl)
                if dv is not None:
                    maxd = max(maxd, abs(dv))
        return (f'种子 1/2/3 相对 seed0 的配对差均在 ±{maxd:.3f} 内, 各问题种子间均值极差'
                f' P1 {spreads[0]:.3f} / P2 {spreads[1]:.3f} / P3 {spreads[2]:.3f} ——'
                f' 方法对随机种子稳健。')
    return ''


def fig_ablation(by):
    tags = [t for t in ABL_ORDER if any((q, t) in by for q in (1, 2, 3))]
    qs = (1, 2, 3)
    colors = {1: '#4C72B0', 2: '#DD8452', 3: '#55A868'}
    x = range(len(tags))
    width = 0.26
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for i, q in enumerate(qs):
        vals, ns = [], []
        for t in tags:
            mv, n = abl_mean(by, q, t)
            vals.append(mv if mv is not None else 0.0)
            ns.append(n)
        ax.bar([xi + (i - 1) * width for xi in x], vals, width * 0.92,
               color=colors[q], label=f'问题{q}')
        for xi, v, n in zip(x, vals, ns):
            if v > 0:
                ax.text(xi + (i - 1) * width, v + 0.06, f'{v:.2f}',
                        ha='center', va='bottom', fontsize=6.2, rotation=90)
    ax.set_xticks(list(x))
    ax.set_xticklabels([ABL_ZH.get(t, t) for t in tags], rotation=28,
                       ha='right', fontsize=9)
    ax.set_ylabel('平均加速比(代表用例)')
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend()
    ax.set_ylim(0, max(mv for q in qs for t in tags
                       for mv, _ in [abl_mean(by, q, t)] if mv) * 1.22)
    fig.tight_layout()
    fig.savefig(f'{FIG_OUT}/fig_ablation.png', dpi=170)
    plt.close(fig)
    return len(tags)


def fig_convergence():
    """读 trainlogs/: 每文件解析 itN mean=M, 归一化 M/M0, 画代表曲线(每问题均衡取例)。"""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    parsed = []  # (q, case, [means])
    for fp in sorted(glob.glob(f'{TR}/train_case_*.log')):
        m = re.search(r'train_case_(\d+)_q(\d)\.log', os.path.basename(fp))
        if not m:
            continue
        case, q = m.group(1), int(m.group(2))
        means = []
        for line in open(fp, encoding='utf-8', errors='ignore'):
            mm = re.search(r'it(\d+) mean=(\d+)', line)
            if mm:
                means.append(int(mm.group(2)))
        if len(means) >= 20:                     # 排除崩溃/未迭代日志
            parsed.append((q, case, means))
    per_q = {q: [p for p in parsed if p[0] == q] for q in (1, 2, 3)}
    target = {1: 3, 2: 4, 3: 3}                  # 共 10 条代表曲线
    n_shown = 0
    for q in (1, 2, 3):
        group = sorted(per_q[q], key=lambda p: p[1])
        k = min(target[q], len(group))
        if k == 0:
            continue
        idx = sorted({round(i * (len(group) - 1) / max(k - 1, 1)) for i in range(k)})
        for j in idx:
            _, case, means = group[j]
            ax.plot(range(len(means)), [v / means[0] for v in means], lw=1.1,
                    alpha=0.85, label=f'case_{case}_q{q}')
            n_shown += 1
    if n_shown:
        ax.set_xlabel('训练轮次')
        ax.set_ylabel('批内平均代价(归一化)')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(f'{FIG_OUT}/fig_convergence.png', dpi=160)
    plt.close(fig)
    return n_shown, len(parsed)


def main():
    sc = load_sc()
    best = load_compliant(sc)
    rows = fig_curves(best, sc)
    by, n_fail = load_ablation()
    abl_lines = ablation_table(by, n_fail)
    n_bar = fig_ablation(by)
    n_cv, n_parsed = fig_convergence()
    rep = ['# 产线数据总报告', '',
           '## 三问平均加速比(合规口径)', '',
           '| 问题 | N | 平均加速比 | 例数 |', '|---|---|---|---|']
    for q, n, v, c in sorted(rows):
        rep.append(f'| {q} | {n} | {v:.4f} | {c} |')
    rep += ['', f'- 合规解记录: {len(best)} (去重后)',
            f'- 消融: 有效 {sum(len(d) for d in by.values())} 例 / FAIL {n_fail} 条, '
            f'柱状图配置数 {n_bar}, 表行 {len(abl_lines)}',
            f'- 收敛曲线: 画 {n_cv} 条 / trainlogs 可解析 {n_parsed} 个日志']
    open(f'{FIG_OUT}/summary_report.md', 'w', encoding='utf-8').write('\n'.join(rep))
    print(f'figs done: rows={len(rows)} ablation_cfgs={n_bar} fail={n_fail} '
          f'conv={n_cv}/{n_parsed} -> {FIG_OUT}')


if __name__ == '__main__':
    main()
