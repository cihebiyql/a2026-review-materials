# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成的。
# 人工智能工具名称：ZCode；版本/型号：GLM-5.3（会话标识 AI-07）；开发机构：智谱（Z.ai）；版本发布日期：待补
# 用途：生成第一章两张示意图草图——F20《计算图最小示例》（1.2.2 节图位，题面附录 B.7 最小例）
#   与 F06《计算图与 Task 组织对照（场景 A/B）》（1.2.3 节图位，同一子 DAG 在两场景下的 Task 组织差异）
# 数据来源：2026A 题面附录 B.7（S01）；纯示意结构图，无实验数据、不伪造执行时刻
# 设计约束：F20——图管关系、表管字段，节点只标代表性属性（Tensor: id/pos/size；Op: id/op/pipe，
#   cycles 仅标在核内计算操作上，DDR 搬运不标以防"由 cycles 定耗"误读）；张量按 pos 上色（DDR 玫瑰/核内绿），
#   操作按管道族上色（MTE 橙/计算蓝）；底部分区仅表示 pos 逻辑位置，非执行时数据流。
#   F06——黑箭头=有向边；玫红折线=经 DDR 中转（右缘出、左缘入，竖段不穿任何子图框）；绿箭头=Task 内驻留复用；
#   实线圆角框=子图；虚线框=Task；DDR 带内通道分层、标签置于带上方避开竖段；图内措辞与正文 V006/V007
#   逐字一致；图内不放图题。输出为 _draft 草图（figs/），用户审定后另定终版名（D034 防覆盖惯例）。
# 版本：v3.1（2026-09-26 GPT 画法建议仲裁轮 A034：F20 补底部图例（形状语义+管道色，形状示意框用白底
#   防"节点类型=某颜色"误读）；图注拟稿落 registry caption 列。v3 三轮修正：F20 节点间距防重叠（像素核验
#   44px 间隙）；F06 面板标题上移避开核心标签、DDR 通道分层且标签避竖段、子图收入 Task 框内；文字包围盒
#   重叠自检，打印 overlap-check）。

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

C_CALC = "#D6E4F0"
C_MTE = "#FDEBD0"
C_MEM = "#DFF0D8"
C_DDR = "#FADBD8"
C_CORE = "#FFFFFF"
C_TASKFILL = "#FCFCFC"
EDGE = "#202020"
ROSE = "#C0392B"
GREEN = "#1E8449"


def rbox(ax, x, y, w, h, fc, lw=0.9, ls="-", ec=EDGE, rounding=None, z=2):
    if rounding is None:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=lw,
                               ls=ls, zorder=z))
    else:
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0,rounding_size=%s" % rounding,
                     facecolor=fc, edgecolor=ec, lw=lw, ls=ls, zorder=z))


def arrow(ax, p0, p1, color=EDGE, lw=1.0, z=3):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8,
                 color=color, lw=lw, zorder=z))


def overlap_check(fig, ax, name):
    """文字包围盒两两求交，打印重叠对；草图自检用，不修改图形。"""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    items = [(t.get_text(), t.get_window_extent(r)) for t in ax.texts]
    bad = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i][1], items[j][1]
            if a.overlaps(b):
                bad.append((items[i][0], items[j][0]))
    if bad:
        print("[%s] overlap-check: %d 处文字重叠 -> %s" % (name, len(bad), bad))
    else:
        print("[%s] overlap-check: OK" % name)


# ============================================================ F20
fig, ax = plt.subplots(figsize=(14 / 2.54, 6.4 / 2.54), dpi=300)
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

rbox(ax, 1, 24, 13, 64, "#FDF3F2", lw=0.7, ls=(0, (4, 3)), ec="#D98880", z=1)
rbox(ax, 15.5, 24, 69, 64, "#F4FAF3", lw=0.7, ls=(0, (4, 3)), ec="#82A87C", z=1)
rbox(ax, 86, 24, 13, 64, "#FDF3F2", lw=0.7, ls=(0, (4, 3)), ec="#D98880", z=1)
ax.text(7.5, 20, "DDR", ha="center", fontsize=6.5, color="#943126")
ax.text(50, 20, "核内缓存（L1 / UB）", ha="center", fontsize=6.5, color="#4D6B47")
ax.text(92.5, 20, "DDR", ha="center", fontsize=6.5, color="#943126")

NW_T, NW_O = 11, 12
Y = 52
nodes = [
    ("t", 6.5,  ["Tensor 1", "pos: DDR", "size: 16"], C_DDR),
    ("o", 21,   ["Op 10", "COPY_IN", "PIPE_MTE2"], C_MTE),
    ("t", 35.5, ["Tensor 2", "pos: UB", "size: 16"], C_MEM),
    ("o", 50,   ["Op 11", "ADD", "PIPE_V", "cycles: 4"], C_CALC),
    ("t", 64.5, ["Tensor 3", "pos: UB", "size: 16"], C_MEM),
    ("o", 79,   ["Op 12", "COPY_OUT", "PIPE_MTE3"], C_MTE),
    ("t", 93.5, ["Tensor 4", "pos: DDR", "size: 16"], C_DDR),
]
for kind, x, lines, fc in nodes:
    w = NW_T if kind == "t" else NW_O
    rbox(ax, x - w / 2, Y - 12, w, 24, fc, lw=0.9,
         rounding=2.0 if kind == "t" else 0.0)
    n = len(lines)
    step = 6.2
    y0 = Y + (n - 1) * step / 2
    for i, s in enumerate(lines):
        ax.text(x, y0 - i * step, s, ha="center", va="center",
                fontsize=6.4 if i == 0 else 5.9,
                weight="bold" if i == 0 else "normal")

for a, b in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]:
    wa = NW_T if nodes[a][0] == "t" else NW_O
    wb = NW_T if nodes[b][0] == "t" else NW_O
    arrow(ax, (nodes[a][1] + wa / 2 + 0.3, Y), (nodes[b][1] - wb / 2 - 0.3, Y))

ax.text(50, 95.5, "节点标注 id 与代表性属性；箭头为有向边（只允许 Tensor→Op 或 Op→Tensor）",
        ha="center", fontsize=6.0, color="#404040")

# 图例（A034 采纳项）：形状语义＋管道色；张量 pos 色由底部分区带标注，形状示意框用中性白底防"类型=颜色"误读
ly = 10
rbox(ax, 6, ly - 1.6, 6, 3.2, "#FFFFFF", lw=0.9, rounding=1.2)
ax.text(13, ly, "Tensor（圆角框）", fontsize=5.8, va="center", ha="left")
rbox(ax, 30, ly - 1.6, 6, 3.2, "#FFFFFF", lw=0.9)
ax.text(37, ly, "Op（直角框）", fontsize=5.8, va="center", ha="left")
rbox(ax, 50, ly - 1.6, 6, 3.2, C_MTE, lw=0.9)
ax.text(57, ly, "搬运管道（MTE）", fontsize=5.8, va="center", ha="left")
rbox(ax, 72, ly - 1.6, 6, 3.2, C_CALC, lw=0.9)
ax.text(79, ly, "计算管道（V）", fontsize=5.8, va="center", ha="left")

overlap_check(fig, ax, "F20")
fig.savefig("figs/F20_input_graph_example_draft.png", bbox_inches="tight",
            facecolor="white")
plt.close(fig)

# ============================================================ F06
# 纵向布局（ylim 0-115，单位均为数据坐标）：
#   每面板：DDR 带 y0+1.5..y0+9.5（带内通道层 y0+5 / y0+8）；核 y0+16.5..y0+44.5；
#   核心标签（框外上方）y0+46.1 起；面板标题 y0+51.5；折线标注带 y0+11.5 / y0+12.5。
SG_W, SG_H = 16, 10.5
CHIP_W, CHIP_H = 8, 4.6

fig, ax = plt.subplots(figsize=(14 / 2.54, 16.1 / 2.54), dpi=300)
ax.set_xlim(0, 100); ax.set_ylim(0, 115); ax.axis("off")


def subgraph(cx, by, idx):
    rbox(ax, cx - SG_W / 2, by, SG_W, SG_H, "#F7F9FB", lw=1.0, rounding=1.2)
    ax.text(cx, by + SG_H - 2.2, "子图 %d" % idx, ha="center", va="center",
            fontsize=6.2, weight="bold")
    rbox(ax, cx - CHIP_W / 2, by + 1.2, CHIP_W, CHIP_H, C_CALC, lw=0.8)
    ax.text(cx, by + 1.2 + CHIP_H / 2, "o%d" % idx, ha="center", va="center",
            fontsize=6.2)
    return dict(cx=cx, y=by, my=by + SG_H / 2)


def taskbox(x, y, w, h, lab):
    rbox(ax, x, y, w, h, C_TASKFILL, lw=1.0, ls=(0, (4, 3)))
    ax.text(x + 1.8, y + h - 2.2, lab, fontsize=6.2, color="#555555",
            ha="left", va="center")


def dip(src, dst, lane, label, lab_x, lab_y):
    """玫红折线：src 右缘中点出 → 竖降至带内通道 → 横移 → 竖升至 dst 左缘中点 → 入。
    竖段取 cx±(SG_W/2+1.5)，按构造不穿任何子图框；标签放带上方，x 避开两条竖段。"""
    x0 = src["cx"] + SG_W / 2 + 1.5
    x1 = dst["cx"] - SG_W / 2 - 1.5
    ax.plot([src["cx"] + SG_W / 2, x0], [src["my"], src["my"]], color=ROSE,
            lw=1.1, zorder=3)
    ax.plot([x0, x0], [src["my"], lane], color=ROSE, lw=1.1, zorder=3)
    ax.plot([x0, x1], [lane, lane], color=ROSE, lw=1.1, zorder=3)
    ax.plot([x1, x1], [lane, dst["my"]], color=ROSE, lw=1.1, zorder=3)
    arrow(ax, (x1, dst["my"]), (dst["cx"] - SG_W / 2, dst["my"]), color=ROSE,
          lw=1.1)
    ax.text(lab_x, lab_y, label, fontsize=5.7, color=ROSE, ha="center",
            va="center")


def panel(y0, title, mode):
    dy0 = y0 + 1.5                    # DDR 带
    cy = y0 + 16.5                    # 核心底
    ch = 28
    rbox(ax, 6, dy0, 88, 8, C_DDR, lw=0.9)
    ax.text(50, y0 + 2.8, "DDR（全核共享）", ha="center", va="center",
            fontsize=6.5, color="#943126")
    for cx, cl in ((6, "核心 0"), (53, "核心 1")):
        rbox(ax, cx, cy, 41, ch, C_CORE, lw=1.1)
        ax.text(cx + 1.0, cy + ch + 1.6, cl, fontsize=7.0, weight="bold",
                ha="left", va="bottom")

    if mode == "A":
        taskbox(8.5, cy + 13.5, 36, 13, "Task 1")
        s0 = subgraph(26.5, cy + 15.2, 0)
        taskbox(8.5, cy + 0.5, 36, 12.3, "Task 2")
        s1 = subgraph(26.5, cy + 1.7, 1)
        taskbox(55.5, cy + 7.5, 36, 13, "Task 3")
        s2 = subgraph(73.5, cy + 9.2, 2)
        dip(s0, s1, y0 + 8, "经 DDR 中转", 26.5, y0 + 11.5)   # 同核：上通道
        dip(s1, s2, y0 + 5, "经 DDR 中转", 55, y0 + 11.5)     # 跨核：下通道
    else:
        taskbox(8.5, cy + 0.6, 36, 26, "Task 1")
        s0 = subgraph(26.5, cy + 15.5, 0)
        s1 = subgraph(26.5, cy + 1.5, 1)
        taskbox(55.5, cy + 7.5, 36, 13, "Task 2")
        s2 = subgraph(73.5, cy + 9.2, 2)
        arrow(ax, (s0["cx"], s0["y"] - 0.6),
              (s1["cx"], s1["y"] + SG_H + 0.6), color=GREEN, lw=1.5)
        ax.text(s0["cx"] + 2.0, (s0["y"] + s1["y"] + SG_H) / 2, "驻留 L1/UB",
                fontsize=5.7, color=GREEN, ha="left", va="center")
        dip(s1, s2, y0 + 5.5, "写入 DDR ＋ 同步信号 → 读取", 50, y0 + 12.5)
    ax.text(50, y0 + 51.5, title, ha="center", fontsize=7.0, weight="bold")


panel(59, "(a) 场景 A：每个子图独立构成一个 Task，Task 间不保留核内缓存状态，跨子图数据经 DDR 中转",
      "A")
panel(6, "(b) 场景 B：同一核心上的全部子图合并为一个 Task，Task 内的子图边界不再插入数据搬运；跨核依赖经 DDR 与同步信号完成",
      "B")

ly = 1.2
ax.plot([12, 17], [ly, ly], color=EDGE, lw=1.1)
ax.text(18, ly, "有向边", fontsize=5.8, va="center")
ax.plot([28, 33], [ly, ly], color=ROSE, lw=1.1)
ax.text(34, ly, "经 DDR 中转", fontsize=5.8, va="center")
arrow(ax, (48, ly + 1.2), (48, ly - 1.2), color=GREEN, lw=1.5)
ax.text(50, ly, "Task 内驻留复用", fontsize=5.8, va="center")
rbox(ax, 64, ly - 1.5, 5, 3, "#F7F9FB", lw=1.0)
ax.text(70, ly, "子图", fontsize=5.8, va="center")
rbox(ax, 76, ly - 1.5, 5, 3, C_TASKFILL, lw=1.0, ls=(0, (4, 3)))
ax.text(82, ly, "Task", fontsize=5.8, va="center")

overlap_check(fig, ax, "F06")
fig.savefig("figs/F06_task_org_draft.png", bbox_inches="tight", facecolor="white")
plt.close(fig)
print("done v3.1")
