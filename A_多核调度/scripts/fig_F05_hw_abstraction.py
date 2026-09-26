# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成的。
# 人工智能工具名称：ZCode；版本/型号：GLM-5.3（会话标识 AI-04）；开发机构：智谱（Z.ai）；版本发布日期：待补
# 用途：生成论文图 F05《多核 NPU 硬件抽象与存储层级示意图》（1.2.1 节图位）
# 数据来源：2026A 题面附录 A、B.4、D.5 固定配置（S01）；纯示意结构图，无实验数据
# 设计约束（用户确认，D024/05-F05/V2 修订）：本文建模采用的题设硬件抽象，不复刻芯片结构；
#   核内为三个功能层（计算/搬运/私有存储），MTE 块居中错位、不与 L1/UB 垂直对齐（避免固定配对暗示）；
#   写路径旁路 L2 直连 DDR（L2 只读，非必经层）；读路径经 L2，标注命中/未命中数据流向；
#   L2 虚线框表示仅问题三引入；图内不放图题（caption 归 LaTeX）；
#   DDR 措辞与正文 V004 完全一致（"由 DDR 服务的核外数据搬运共同竞争"）。

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 配色：低饱和学术彩色（各色相亮度拉开，转灰后仍可辨）；GRAYSCALE=True 切回灰度版。
GRAYSCALE = False
if GRAYSCALE:
    C_CORE, C_CALC, C_MTE, C_MEM, C_DDR, C_L2 = \
        "#F2F2F2", "#D9D9D9", "#E8E8E8", "#C8C8C8", "#BFBFBF", "#FAFAFA"
else:
    C_CORE = "#FFFFFF"      # 核外框：白
    C_CALC = "#D6E4F0"      # 计算单元：浅蓝
    C_MTE = "#FDEBD0"       # 搬运单元：浅橙
    C_MEM = "#DFF0D8"       # 私有缓存：浅绿
    C_DDR = "#FADBD8"       # DDR：浅玫瑰
    C_L2 = "#E8E1F2"        # L2：浅紫
EDGE = "#202020"

fig, ax = plt.subplots(figsize=(14 / 2.54, 10.5 / 2.54), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def box(x, y, w, h, fc, lw=0.9, ls="-"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.6",
                       fc=fc, ec=EDGE, lw=lw, linestyle=ls)
    ax.add_patch(p)
    return p


def txt(x, y, s, fs=7.5, bold=False, color="#202020", ha="center", va="center", lsp=1.25):
    ax.text(x, y, s, fontsize=fs, fontweight="bold" if bold else "normal",
            color=color, ha=ha, va=va, linespacing=lsp)


def arrow(x1, y1, x2, y2, lw=1.2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=10, lw=lw, color=EDGE))


def line(x1, y1, x2, y2, lw=1.2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-",
                                 lw=lw, color=EDGE))


# ---------- 顶层：N 个 AI 核心（画 Core 0 与 Core N-1，中间省略） ----------
def draw_core(x0):
    box(x0, 62, 34, 34, C_CORE, lw=1.3)
    # 计算层：两块分居左右
    box(x0 + 2, 84.5, 13.5, 6.5, C_CALC)
    txt(x0 + 8.75, 87.75, "Cube\n(PIPE_M)", fs=7)
    box(x0 + 18.5, 84.5, 13.5, 6.5, C_CALC)
    txt(x0 + 25.25, 87.75, "Vector\n(PIPE_V)", fs=7)
    # 搬运层：两块居中并拢，不与上下层块对齐（避免 MTE—存储固定配对暗示）
    box(x0 + 7.5, 76.5, 8.5, 6, C_MTE)
    txt(x0 + 11.75, 79.5, "MTE2", fs=7)
    box(x0 + 18, 76.5, 8.5, 6, C_MTE)
    txt(x0 + 22.25, 79.5, "MTE3", fs=7)
    txt(x0 + 17, 74.6, "数据搬运单元", fs=6.2, color="#505050")
    # 私有存储层：整宽子框 + 内含两块
    box(x0 + 1.5, 63.5, 31, 9.5, C_MEM)
    txt(x0 + 17, 71.2, "私有核内存储", fs=6.2, color="#505050")
    box(x0 + 3, 64.2, 12, 6, "#FFFFFF", lw=0.7)
    txt(x0 + 9, 67.2, "L1\n512 KiB", fs=7)
    box(x0 + 19, 64.2, 12, 6, "#FFFFFF", lw=0.7)
    txt(x0 + 25, 67.2, "UB\n128 KiB", fs=7)


draw_core(5)
txt(22, 94, "Core 0", fs=9, bold=True)
draw_core(61)
txt(78, 94, "Core N-1", fs=9, bold=True)
txt(50, 87, "……", fs=13)
txt(50, 78, "同构核心", fs=7.5, color="#505050")

# ---------- 核底向下到共享访问路径 ----------
arrow(22, 62, 22, 54)
arrow(78, 62, 78, 54)
line(22, 54, 78, 54, lw=1.4)
txt(50, 55.8, "共享访问路径", fs=6.8, color="#505050")

# ---------- 分叉：写路径（旁路 L2）与可缓存读路径（经 L2） ----------
# 写路径：左侧直下 DDR（末端箭头标示访问方向；数据流向由标注文字说明）
arrow(30, 54, 30, 34)
txt(28.5, 45, "写路径（旁路 L2）", fs=6.8, ha="right")
# 读路径：右侧经 L2 虚线框到 DDR
arrow(70, 54, 70, 47.5)
txt(71.5, 51, "可缓存读路径", fs=6.8, ha="left")

# L2：虚线框，仅覆盖读路径（挂在读取路径上的可选模块，仅问题三）
box(52, 37, 40, 10.5, C_L2, lw=1.4, ls=(0, (5, 3)))
txt(72, 44.6, "只读 L2（仅问题三）", fs=7.8, bold=True)
txt(72, 40.2, "1048576 bytes（题面记为 1 MB）；读带宽 250 bytes/cycle，独立于 DDR",
    fs=6.6)
line(70, 37, 70, 35.5)
arrow(70, 35.5, 70, 34)
txt(71.5, 35.5, "命中：L2→Core；未命中：DDR→L2→Core", fs=6.2, ha="left", color="#303030")

# ---------- 底层：共享 DDR ----------
box(5, 22, 90, 12, C_DDR, lw=1.4)
txt(50, 30.6, "共享核外主存 DDR", fs=9, bold=True)
txt(50, 25.2, "总物理带宽 60 bytes/cycle，由 DDR 服务的核外数据搬运共同竞争", fs=7.2)

# ---------- 底注 ----------
txt(50, 14.5,
    "注：L1、UB 为每核私有，容量分别为 524288、131072 bytes；核心数量仅作示意。\n"
    "MTE2、MTE3 的完整搬运方向见正文，其中包含 L1 与 UB 之间的核内搬运。",
    fs=6.8, color="#303030", lsp=1.5)

# 输出为 _scriptref 参考版：F05_hw_abstraction.png 已是用户定稿的 AI 生成终图，
# 此脚本不得覆盖它（终图无重绘脚本，题面参数变更时需人工重新制作，见 08/D026）。
fig.savefig("../figs/F05_hw_abstraction_scriptref.png", bbox_inches="tight", facecolor="white")
fig.savefig("../figs/F05_hw_abstraction_scriptref.pdf", bbox_inches="tight", facecolor="white")
print("saved: ../figs/F05_hw_abstraction_scriptref.png / .pdf")
