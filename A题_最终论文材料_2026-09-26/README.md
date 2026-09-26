# A题 最终论文材料总包（2026-09-26）

> 2026 华为杯 A题《通用神经网络处理器下的多核调度》全量交付材料。
> 两条轨道：**最终方案轨**（离线充分校准的逐用例最优方案，N=5 平均加速比
> P1 **3.6508** / P2 **4.3132** / P3 **4.3970**，300 条全部方案可复现，
> 收割对账 verify_bad=0）与**合规在线轨**（两段式求解器 8 分钟内独立产出，
> 1–5 核全曲线）。全部数字官方评估器口径。

## 论文写作取材速查

| 论文位置 | 取材 | 说明 |
|---|---|---|
| 摘要/主表 N=5 | `02_结果数据/pool_track/pool_summary.json` + `gap/gap_table.csv` | 最终方案轨三问均值 |
| 正文 1–5 核折线 | `03_图表/paper_figs/fig_curves.png` + `compliant_track/curve_data.csv` | 在线求解器全曲线 |
| 附录逐用例表 P1/P2 | `pool_track/pool_appendix_problem{1,2}.csv`（最终方案）/ `compliant_track/appendix_problem{1,2}.csv`（在线轨） | Makespan+额外搬运量 |
| 附录逐用例表 P3 | `pool_track/pool_appendix_problem3.csv` | 无L2/Cache 对比+命中率(次数/字节)+相对加速比 |
| 消融表/图 | `03_图表/paper_figs/fig_ablation.png` + `ablation_table.md` | 配对差口径,五族配置 |
| 收敛曲线 | `fig_convergence.png` | REINFORCE 训练 10 例 |
| 求解时间合规 | `compliant_track/timing_stats.csv` | 每例 wall_s/预算占用(≤8min) |
| Gap 分析(讨论节) | `gap/gap_table.csv` | 池 vs 在线逐例差 = 离线校准价值 |
| 方法章骨架 | `04_文档/论文方案与叙事_2026-09-26.md` | 两段式叙事+章节结构+引用合规格式 |

## 目录

```
01_方案与源码/
  final_plans_appendixB/   300 个逐用例最终方案(题面附录B两字段) + manifest
  repro/                    复现包: 求解器+训练器+收割器+300 logits+README(四条命令)
  src/                      全部脚本(训练/求解/解码/收割/消融/对账/打包/发射)
02_结果数据/
  champion_pool/            权威池 300 json(方案+mk+sp, 全部可复现)
  compliant_track/          在线轨: jsonl 记录+方案json+附录表+曲线+耗时
  pool_track/               最终方案轨附录表(次指标含搬运量/命中率)
  gap/                      池vs在线逐例对照表
  ablation/                 消融实验原始记录
  logits/                   300 个离线训练参数(one-shot 优先级模型)
03_图表/paper_figs/         正文三图+数据表
04_文档/                    叙事/规划/结果分析/22例改进/战役全记录(inspect_log)
```

## 关键事实（写作直接引用）

- 方法：ICLR23 one-shot 优先级采样(Jeon et al.)的领域化移植 + 构造解码器 + REINFORCE 离线训练；在线 8 分钟求解器(题面 5–10 分钟口径)。
- 合规：源码全部带 AI 辅助声明(附件4第5条)；引用条目见 repro/README。
- 覆盖：P1/P2 在线轨 100/100×4 核数；P3 覆盖随 fill/豁免通道收口（豁免=评估器原生崩溃例走官方评估器, 见 pool_summary.json crash_fast_fallback）。
- 数据完整性：55 组历史池 plan/mk 错配已全链修复(43 runs树验真+4 verified明细+8 诚实重评), 终版 300 条 verify_bad=0。

## 复现入口

见 `01_方案与源码/repro/README.md`（四条命令：stub 自测 / 单例训练 / 在线求解 / 全库分片）。
