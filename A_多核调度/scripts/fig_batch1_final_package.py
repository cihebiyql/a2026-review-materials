# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成的。
# 工具名称：ZCode（会话代理）；模型：GLM-5.3；开发机构：智谱（Z.ai）；版本发布日期：待补
# 用途：生成论文批次1数据图 F04/F19/F14/F08/F11（成图待审核；图题进 LaTeX caption，图内不设总标题）
# 数据源：../../A题_交付数据包_0925终版/MANIFEST_终版.csv（RES-012，CONFIRMED）与 speedup_stats_v4.csv（P25/P75 分位）
# 输出：../figs/F04_*.png/pdf、F19_*、F14_*、F08_*、F11_* 及 ../figs/figdata/figdata_batch1_0926.csv（可重绘数据）
import csv, os, statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.normpath(os.path.join(HERE, '..', '..', '..', '..', '..', 'A题_交付数据包_0925终版'))
FIG = os.path.normpath(os.path.join(HERE, '..', 'figs'))
DAT = os.path.join(FIG, 'figdata')
os.makedirs(DAT, exist_ok=True)

man = list(csv.DictReader(open(os.path.join(PKG, 'MANIFEST_终版.csv'), encoding='utf-8-sig')))
v4 = {(r['问题'].replace('问题', ''), r['N']): r
      for r in csv.DictReader(open(os.path.join(PKG, 'speedup_stats_v4.csv'), encoding='utf-8-sig'))}

groups = {}
for r in man:
    groups.setdefault((r['q'], int(r['N'])), []).append(float(r['sp']))
mean = {k: statistics.mean(v) for k, v in groups.items()}
Ns = [1, 2, 3, 4, 5]


def savefig(fig, name):
    fig.savefig(os.path.join(FIG, name + '.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(FIG, name + '.pdf'), bbox_inches='tight')
    plt.close(fig)


def core_curve(qkey, color, name, band_label):
    ys = [1.0] + [mean[(qkey, n)] for n in Ns[1:]]
    p25 = [1.0] + [float(v4[(qkey, str(n))]['p25']) for n in Ns[1:]]
    p75 = [1.0] + [float(v4[(qkey, str(n))]['p75']) for n in Ns[1:]]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.fill_between(Ns, p25, p75, alpha=0.18, color=color, label='P25–P75 分位区间')
    ax.plot(Ns, Ns, 'k--', lw=1.0, label='理想线性加速')
    ax.plot(Ns, ys, 'o-', color=color, label='100 例平均加速比')
    for n, y in zip(Ns[1:], ys[1:]):
        ax.annotate(f'{y:.2f}', (n, y), textcoords='offset points', xytext=(0, 8),
                    ha='center', fontsize=9)
    ax.set_xticks(Ns)
    ax.set_xlabel('核数 N')
    ax.set_ylabel('相对单核基准的平均加速比')
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 5.4)
    savefig(fig, name)
    return ys


y_p1 = core_curve('1', '#1f77b4', 'F19_p1_core_scaling', 'P1')
y_p2 = core_curve('2', '#d62728', 'F04_p2_core_scaling', 'P2')
y_p3 = [1.0] + [mean[('3', n)] for n in Ns[1:]]

# F14：(a) 两配置核数扩展；(b) L2 相对增益（均值口径 q3/q2）
gain = [1.0] + [y_p3[i] / y_p2[i] for i in range(1, 5)]
fig, (a, b) = plt.subplots(1, 2, figsize=(9.8, 4.0))
a.plot(Ns, Ns, 'k--', lw=1.0, label='理想线性')
a.plot(Ns, y_p2, 'o-', color='#7f7f7f', label='问题二（无 L2）')
a.plot(Ns, y_p3, 's-', color='#2ca02c', label='问题三（只读 L2）')
a.set_xticks(Ns)
a.set_xlabel('核数 N')
a.set_ylabel('100 例平均加速比')
a.legend(frameon=False, fontsize=9, loc='upper left')
a.grid(alpha=0.3)
a.set_ylim(0, 5.4)
a.set_title('(a) 两配置核数扩展', fontsize=10)
dev = [g - 1.0 for g in gain[1:]]
bars = b.bar([str(n) for n in Ns[1:]], dev, bottom=1.0, color='#2ca02c', alpha=0.8, width=0.55)
b.axhline(1.0, color='k', lw=0.8)
for i, g in enumerate(gain[1:]):
    b.annotate(f'×{g:.3f}', (i, g), textcoords='offset points', xytext=(0, 5),
               ha='center', fontsize=9)
b.set_ylim(0.0, 1.25)
b.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
b.set_xlabel('核数 N')
b.set_ylabel('L2 相对无 L2 的平均增益')
b.grid(alpha=0.3, axis='y')
b.set_title('(b) 相对增益（均值比口径）', fontsize=10)
savefig(fig, 'F14_cache_gain_vs_cores')


def dist(qkey, color, hi_color, name):
    cases = {r['case']: float(r['sp']) for r in man if r['q'] == qkey and r['N'] == '5'}
    srt = sorted(cases.items(), key=lambda kv: kv[1])
    vals = [v for _, v in srt]
    m = statistics.mean(vals)
    colors = [hi_color if v > 5 else color for v in vals]
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.bar(range(1, 101), vals, color=colors, width=1.0)
    ax.axhline(m, color='k', ls='--', lw=1.0)
    ax.annotate(f'均值 {m:.3f}', (58, m), textcoords='offset points', xytext=(0, 5), fontsize=9)
    for idx, dy in [(0, 10), (1, 4), (97, 4), (98, 10), (99, 16)]:
        c, v = srt[idx]
        ax.annotate(c.replace("case_", "") + " (" + format(v, ".2f") + ")", (idx + 1, v), textcoords="offset points", xytext=(0, dy), ha="center", fontsize=7.5)
    n_hi = sum(1 for v in vals if v > 5)
    ax.set_xlabel('用例（按加速比排序）')
    ax.set_ylabel('N=5 相对单核基准加速比')
    ax.grid(alpha=0.3, axis='y')
    import matplotlib.patches as mpatches
    ax.legend(handles=[mpatches.Patch(color=color, label=f'加速比 ≤ 5（{100 - n_hi} 例）'),
                       mpatches.Patch(color=hi_color, label=f'加速比 > 5（{n_hi} 例）')],
              frameon=False, fontsize=9, loc='upper left')
    savefig(fig, name)
    return srt[0], srt[-1], n_hi


d1 = dist('1', '#9ecae1', '#1f77b4', 'F08_p1_speedup_dist')
d2 = dist('2', '#fcbba1', '#d62728', 'F11_p2_speedup_dist')

# 可重绘数据落盘
with open(os.path.join(DAT, 'figdata_batch1_0926.csv'), 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.writer(f)
    w.writerow(['series', 'N', 'value'])
    for q, ys in [('P1', y_p1), ('P2', y_p2), ('P3', y_p3), ('gain_q3_over_q2', gain)]:
        for n, v in zip(Ns, ys):
            w.writerow([q, n, f'{v:.4f}'])
    w.writerow(['note', '', 'P25/P75 见 speedup_stats_v4.csv；逐例 sp 见 MANIFEST_终版.csv'])
print('F04/F19/F14/F08/F11 done')
print('P1N5 extremes:', d1, '| P2N5 extremes:', d2)
