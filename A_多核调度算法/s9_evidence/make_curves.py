# -*- coding: utf-8 -*-
"""s9_evidence.make_curves — 赛题正文图：1~5 核平均加速比折线图。

  figs/core_scaling_q{Q}.png：V4 最终版 1~5 核曲线（N=1 定义为 1），
      叠加 V1 基线曲线（改进幅度可视化）。
  figs/q3_cache_vs_nol2.png（仅 Q3）：无 L2 vs 只读 Cache 两配置曲线
      + 逐核数 Cache 加速比标注（题三正文要求）。
  ablation.png：N=5 版本阶梯柱状图（V1→V4 各问平均加速比）。
"""
import csv
import os

import paths


def _read_summary():
    fp = os.path.join(paths.RESULTS, 'tables', 'summary.csv')
    data = {}
    for r in csv.DictReader(open(fp)):
        data[(int(r['Q']), int(r['N']), r['version'])] = \
            float(r['mean_sp']) if r['mean_sp'] else None
    return data


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    S = _read_summary()
    figs = os.path.join(paths.RESULTS, 'figs')
    os.makedirs(figs, exist_ok=True)
    qs = sorted({k[0] for k in S})

    # 1) 1~5 核折线（每问一图：V4 主线 + V1 基线）
    for Q in qs:
        fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150)
        xs_all = sorted({k[1] for k in S if k[0] == Q})
        for ver, color, label in (('v1', '#7f7f7f', 'V1 baseline'),
                                  ('v4', '#1f77b4', 'V4 final')):
            ys = [1.0] + [S.get((Q, n, ver)) for n in xs_all if n > 1]
            xs = [1] + [n for n in xs_all if n > 1]
            ax.plot(xs, ys, 'o-' if ver == 'v4' else 'o--',
                    color=color, lw=2, ms=6, label=label)
            for x, y in zip(xs, ys):
                if y is not None:
                    ax.annotate(f'{y:.4f}', (x, y),
                                textcoords='offset points',
                                xytext=(0, 8 if ver == 'v4' else -14),
                                ha='center', fontsize=8, color=color)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlabel('Number of cores N')
        ax.set_ylabel('Mean speedup (100 cases)')
        ax.set_title(f'P{Q}: core-scaling curve (official evaluator)')
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(figs, f'core_scaling_q{Q}.png'))
        plt.close(fig)

    # 2) Q3 两配置对比 + Cache 加速比
    if (3, 2, 'v4') in S:
        # 逐例 cache_speedup 从 percase 表读
        tab = os.path.join(paths.RESULTS, 'tables')
        cache_sp = {}
        for N in (2, 3, 4, 5):
            fp = os.path.join(tab, f'percase_q3_N{N}_v4.csv')
            if not os.path.exists(fp):
                continue
            vals = [float(r['cache_speedup']) for r in
                    csv.DictReader(open(fp)) if r.get('cache_speedup')]
            if vals:
                cache_sp[N] = sum(vals) / len(vals)
        if cache_sp:
            fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150)
            xs = [1] + sorted(cache_sp)
            noL2 = [1.0] + [S.get((3, n, 'v4')) for n in sorted(cache_sp)]
            cache = [1.0] + [S.get((3, n, 'v4')) * cache_sp[n]
                             for n in sorted(cache_sp)]
            ax.plot(xs, noL2, 's--', color='#d62728', lw=2,
                    label='P3 without L2 (=P2 curve)')
            ax.plot(xs, cache, 'o-', color='#1f77b4', lw=2,
                    label='P3 with read-only Cache')
            for x, y, cs in zip(xs[1:], cache[1:],
                                [cache_sp[n] for n in sorted(cache_sp)]):
                ax.annotate(f'{y:.4f}\n(cache x{cs:.4f})', (x, y),
                            textcoords='offset points', xytext=(0, 10),
                            ha='center', fontsize=8)
            ax.set_xticks([1, 2, 3, 4, 5])
            ax.set_xlabel('Number of cores N')
            ax.set_ylabel('Mean speedup (100 cases)')
            ax.set_title('P3: without-L2 vs read-only Cache (V4)')
            ax.grid(alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(figs, 'q3_cache_vs_nol2.png'))
            plt.close(fig)

    # 3) N=5 版本阶梯消融
    if any((Q, 5, 'v4') in S for Q in qs):
        fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150)
        width = 0.2
        for i, Q in enumerate(qs):
            vals = [S.get((Q, 5, v)) for v in ('v1', 'v2', 'v3', 'v4')]
            xs = [j + (i - len(qs) / 2) * width
                  for j in range(len(vals))]
            bars = ax.bar(xs, [v or 0 for v in vals], width,
                          label=f'P{Q}')
            for x, v in zip(xs, vals):
                if v:
                    ax.annotate(f'{v:.3f}', (x, v),
                                textcoords='offset points', xytext=(0, 3),
                                ha='center', fontsize=7)
        ax.set_xticks(range(4))
        ax.set_xticklabels(['V1 baseline', 'V2 +B×L relabel',
                            'V3 +structures', 'V4 +learned assign'])
        ax.set_ylabel('Mean speedup @ N=5')
        ax.set_title('Version ladder (ablation, N=5)')
        ax.grid(alpha=0.3, axis='y')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(figs, 'ablation_ladder.png'))
        plt.close(fig)
    print('CURVES_DONE ->', figs)


if __name__ == '__main__':
    main()
