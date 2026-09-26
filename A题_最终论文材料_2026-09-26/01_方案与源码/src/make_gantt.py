# make_gantt.py — 最终方案轨三例甘特时间线(官方评估器 per_core_timeline 直出, 子图级)
# 本程序及代码是在人工智能工具辅助下完成的。
# 人工智能工具名称：GLM，版本/型号：zai-api/GLM-5.3，开发机构/公司：智谱AI(Z.ai)，版本颁布日期：2026-09-26
# 用法: py -3.11 make_gantt.py case_008 case_064 case_084
import sys, os, json
sys.path.insert(0, r'C:/shumo_live/02_求解/A题_2026/v3_solver')
sys.path.insert(0, r'C:/shumo_live/a_data/code')
os.environ['A2026_ATT'] = r'C:/shumo_live/a_data'
from common import load_case, ev_p2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

OUT = r'C:/shumo_live/02_求解/A题_2026/n5_push/paper_figs'
SA = r'C:/shumo_live/02_求解/A题_2026/n5_push/superlinear_analysis'

for case in sys.argv[1:]:
    plan = json.load(open(f'{SA}/{case}_q2_mechA.json'))['plan']
    res = ev_p2(load_case(case), plan)[0]
    tl = res['per_core_timeline']
    fig, ax = plt.subplots(figsize=(10, 0.9 * len(tl) + 1.6))
    cmap = plt.get_cmap('tab20')
    for row, core in enumerate(sorted(tl, key=lambda c: c['core_id'])):
        c = core['core_id']
        for seg in core['subgraphs']:
            ax.barh(row, seg['duration'], left=seg['start'], height=0.62,
                    color=cmap(seg['subgraph_id'] % 20), edgecolor='none')
    ax.set_yticks(range(len(tl)))
    ax.set_yticklabels([f"核{core['core_id']}" for core in sorted(tl, key=lambda x: x['core_id'])])
    ax.set_xlabel('时钟周期 (cycles)')
    ax.set_xlim(0, res['makespan'])
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    fp = f'{OUT}/fig_gantt_{case}_q2_N5.png'
    fig.savefig(fp, dpi=160)
    plt.close(fig)
    n_seg = sum(len(core['subgraphs']) for core in tl)
    print(f'{case}: mk={res["makespan"]} 核数={len(tl)} 子图段={n_seg} -> {fp}')
