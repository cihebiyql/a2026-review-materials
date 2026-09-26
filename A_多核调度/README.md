# 华为杯数学建模 A 题写作工作区

这里是团队仓库本地副本中的可编辑工作区，创建于 2026-09-24，遵循仓库 `competition/<年份>/<题号>_<短名>/` 规范。正文在 `论文.tex`，人与 AI 的协作记录集中在 `notes/`。

## 每次工作的入口

上下文按 [写作规则](notes/00_writing_rules.md) 第 4 节"按任务加载"取用（共同基线 = 00 + [待办与交接](notes/07_todo.md)），本 README 只做导航、不记录当前状态（D027）。完成一轮工作后，说明修改内容、使用资料、数据来源和待人工确认问题，更新受影响的记录与待办，重要选择写入决策日志。

了解“每问有什么方法、哪个代码版本、哪套结果、还缺什么”，直接读[三问方法—版本—结果一页表](notes/01_context.md#method-result-map)。表内仅链接结果ID；详细数字和核验仍由02维护。仓库已有方法、新候选与论文最终采用分别记录，最终选择未定时不写成已采用。

结果使用 `CONFIRMED / PRELIMINARY / DEPRECATED / UNKNOWN` 四种状态。文稿的“草稿 / 待人工审核 / 已审核”另行记录；已确认结果不等于文稿已定稿。目录以 [paper_outline.md](paper_outline.md) 的用户确认冻结版为准（结构冻结不等于方法或结果已确认）。

| 文件 | 用途 |
|---|---|
| [00_writing_rules.md](notes/00_writing_rules.md) | 人与 AI 的职责、写作规范、证据要求和每轮工作流程 |
| [01_context.md](notes/01_context.md) | 官方机制与决策边界、指标及交付要求、带版本的进展快照、来源冲突和候选主线 |
| [02_confirmed_results.md](notes/02_confirmed_results.md) | v0.2：三问结果索引与详细核验记录，独立管理状态、结果类型、用途及历史变更 |
| [paper_outline.md](paper_outline.md) | 冻结目录、章节说明与图表/公式/结果 TODO |
| [03_outline.md](notes/03_outline.md) | 分章文件映射、正文标签与章节审核状态 |
| [04_claim_evidence.md](notes/04_claim_evidence.md) | 每项论文论断与证据的对应关系 |
| [05_figure_plan.md](notes/05_figure_plan.md) | 图表用途、证据需求和拟放位置 |
| [06_glossary.md](notes/06_glossary.md) | 术语、符号和单位的统一约定 |
| [07_todo.md](notes/07_todo.md) | 当前任务、状态、依赖和下一轮交接 |
| [08_decision_log.md](notes/08_decision_log.md) | 已作决定、依据、影响范围和待决策项 |
| [09_ai_usage_log.md](notes/09_ai_usage_log.md) | AI 使用逐条留痕、官方附件4三场景标注规则与交稿使用说明入口 |
| [10_delivery_checklist.md](notes/10_delivery_checklist.md) | 交稿终检：数字一致性、摘要篇幅、图表、语言门、引用、AI 合规、附件2参数表与双通道验收 |
| [11_version_log.md](notes/11_version_log.md) | 草稿历史留档：git 本地提交版本索引与回滚方式（D019） |

图表文件路径、标签、版本和制作状态继续登记在 `figs/figure_registry.csv`，与图表计划共用编号。

## 文件放在哪里

```text
A_多核调度/
├── 论文.tex                 论文唯一主入口
├── paper_outline.md         冻结目录与写作 TODO
├── paper/
│   ├── sections/            七章正文及参考文献的 LaTeX 骨架
│   └── appendix/            三问逐用例结果及补充算法/实验骨架
├── notes/
│   ├── 00_writing_rules.md
│   ├── 01_context.md
│   ├── 02_confirmed_results.md
│   ├── 03_outline.md
│   ├── 04_claim_evidence.md
│   ├── 05_figure_plan.md
│   ├── 06_glossary.md
│   ├── 07_todo.md
│   ├── 08_decision_log.md
│   ├── 09_ai_usage_log.md
│   ├── 10_delivery_checklist.md
│   └── 11_version_log.md
├── references.bib           已核验的参考文献
├── figs/
│   ├── README.md            图表交付规范
│   └── figure_registry.csv  图表登记表
│   （v2_reference/ 保存已有 v2 图表的参考副本）
├── data/
│   ├── raw/                接收的原始实验结果副本
│   └── processed/          用于图表的整理数据
├── scripts/                后续绘图与数据整理脚本
├── tables/                 可复用的 LaTeX 表格片段
├── 题面/                   官方题目位置索引
├── code/                   已有求解代码位置索引
├── results/                实验结果索引与待核验摘要
├── build/                  编译产物（首次编译时生成）
└── build.ps1               Windows 编译入口
```

## 开始写作

1. 按 `notes/01_context.md` 的来源索引阅读题目原文、仿真器说明和各版本报告；路径已经登记。
2. 按 `paper_outline.md` 的冻结目录，在 `paper/sections/` 和 `paper/appendix/` 中逐章补充待补内容。摘要最后写。
3. 实验数值与核验记录维护在 `notes/02_confirmed_results.md`，论文论断与依据对应关系维护在 `notes/04_claim_evidence.md`。
4. 图表先登记，再添加数据、脚本和成图；将核验完成的图表引用到正文。
5. 每次修改正文后编译查看 PDF。提交前清除全部“待补”标记，按 [交付终检清单](notes/10_delivery_checklist.md) 和官方模板检查格式；AI 参与的工作随做随登入 [AI 使用台账](notes/09_ai_usage_log.md)。

## 编译和预览

需要含中文支持的 TeX Live 或 MiKTeX，以及可直接调用的 `xelatex`。在本目录的 PowerShell 中运行：

```powershell
.\build.ps1
```

成功后打开 `build/论文.pdf`。脚本连续编译两遍以更新交叉引用，随后自动 lint（待补标记、未定义引用/交叉引用、Overfull、重复标签），Overfull 清零后再提交（D031）；编译失败会停止并提示日志位置。本机 XeLaTeX 环境与首次编译结果见 [07_todo.md](notes/07_todo.md) T009 及本轮交接，不在此维护状态。

`references.bib` 用于积累已核验条目。团队模板使用手工 `thebibliography`；1.1已引用两条文献并同步参考文献节，其他所需文献待补。若后续改用 BibTeX/Biber，应同步调整编译流程。

## 与团队仓库的关系

- 团队地址：<https://github.com/cihebiyql/shumo-toolkit>。
- 方法证据与固定远端快照的对应关系在 [01_context.md](notes/01_context.md) 一页表与 [02_confirmed_results.md](notes/02_confirmed_results.md) 维护；正文本地留档见 [11_version_log.md](notes/11_version_log.md)，推送时机按 [00_writing_rules.md](notes/00_writing_rules.md) 第 3.3 节执行。各源文件的入库提交与实验实际运行版本分别核对。
- `论文.tex` 已复用 `../../../templates/论文模板_官方风格.tex` 的页面设置、字体、摘要页和封面入口；删除了模板中其他题目的示例数字、公式和图片引用，换为 A 题章节占位。
- 官方格式及封面原件位于 `../../../templates/官方模板_2026/`。最终提交前填写封面并编译检查；此次尚未验证最终排版。
- 已有求解代码仍在 `../A题_代码包/`，报告在 `../迭代报告_自动同步/`。本工作区用索引引用，避免复制出第二套求解代码。
- 本次没有修改自动同步目录、既有求解代码、报告或团队模板。
- 上级项目 `sources/` 属于只读同步资料区，不在其中保存写作成果。

## 后续协作

后续可以直接指定“写问题一的模型部分”“依据某份核验报告更新结果”“检查图表编号与引用”。正式绘图前确定使用 Python 还是 R；当前只整理已有图表，没有运行模型或生成新实验图。
