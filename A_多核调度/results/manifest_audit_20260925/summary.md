# 交付数据包落地核验记录（2026-09-25 夜，AI-05）

对象：`../../A题_交付数据包_0925夜/`（MANIFEST.csv＋singlecore/100份＋singlecore_baseline_100.csv＋stats/6表＋speedup_stats_v4.csv，共110文件）。
方法：anaconda Python 独立读取复算，与师兄书面答复（notes/20260925_师兄数据确认回复_原文.md）逐项比对。脚本与输出见当日对话记录。

## 通过项
1. MANIFEST 1200 行（100例×3问×4核数），列 case/q/N/mk/sp/source；12 个均值中 11 个与答复成绩单精确一致（P1N2~N4、P2N2~N5、P3N2~N4）。P2N5＝4.2697≈4.270 ✓。
2. 命中率逐例均值 N2~N5＝15.2%/22.0%/24.0%/26.8%，与答复 B1 精确一致；另算得字节加权口径＝18.1%/22.4%/27.3%/29.9%，确认答复口径为逐例算术平均。
3. S_Cache 逐例均值 N2~N5＝1.0010/1.0069/1.0207/1.0508、单例最大 2.1481，与答复 B2 精确一致。
4. 单核基准交叉验证：随机抽样 120 行（有效 mk），sp＝sc_mk/mk 全部在 ±0.1% 内；singlecore 100 份 JSON 与 baseline CSV 全量一致（不含上方待澄清行）。

## 待澄清项（回传师兄）
1. P1 N5：MANIFEST 复算＝3.7048（n=100），答复/成绩单/v4＝3.709。成绩单其余格与 MANIFEST 一致、仅此格与 v4 一致——请确认终值；若 3.709 来自更新的 LNS 轮次，以新 MANIFEST 为准。
2. v4 与 MANIFEST 三处不一致：P1N5（3.709/3.7048）、P2N4（3.459/3.5509）、P3N4（3.525/3.6367）。按 RES-010 以 MANIFEST 为准则 v4 为过期快照，建议更新或注明。
3. MANIFEST 23 行空 mk（P2 N5×7、P3 N5×16，source=audit:cross_best/audit_B_best），sp 有值、mk 缺。**已自解**：p3_hit_full/p3_scache_full/p12_eval_full 三表 mk 全量无空值（含 N5），TBL-006/F15 逐例 makespan 可从 stats 表取；仅建议师兄下次导出时补 MANIFEST 字段。另 stats/superlinear_cases.csv（24 行，含逐例 sp_p1/sp_p2/spill）可支撑 TBL-007 逐例归属。
4. N2/N3 LNS 增量明晨落地后，本记录 N2/N3 相关数值需复核。

---

# 终版包核验补记（2026-09-25/26 交接夜，AI-05）

对象：`../../A题_交付数据包_0925终版/`（13 文件：MANIFEST_终版.csv、gantt/6 份时间线、ablation_stages.csv、singlecore_baseline_100.csv、speedup_stats_v4.csv、stats/3 表）。

## 已解决（对照上一节待澄清项）
1. **P1 N5 定案＝3.7048**（答复/旧 v4 的 3.709 为笔误）：终版 MANIFEST 复算 3.7048（n=100），新 v4 与终版 12 点全部一致（0 处不符）。
2. **空 mk 清零**：终版 1200 行 mk 全有值。
3. **N2/N3 LNS 增量落地**（终值）：P1 1.8068/2.4792（Δ+0.012/+0.028）；P2 1.9704/2.7601（+0.026/+0.043）；P3 1.9948/2.7887（+0.017/+0.034）；N4/N5 与夜版逐位一致。逐例变化数：P1N2 34、P1N3 41、P2N2 25、P2N3 36、P3N2 20、P3N3 27（/100）。
4. **甘特图数据交付并通过对账**：008/064/084 × pre/post 共 6 份逐操作时间线（字段 case/ver/core/op_id/op/pipe/start/end/duration/mk）；三例 post 的 mk 与 MANIFEST P2N5 逐例一致（73515/14916/378550）。F03 解锁。
5. **消融阶梯表交付**：ablation_stages.csv 10 行（stage/exp/baseline/candidate/delta/wins/losses），含 P1/P2/P3 的 LNS 段（3.587→3.7048、4.156→4.2697、4.2441→4.3091）；注意各段来自独立 A/B 实验（N4 期与 N5 期混合），拼表时须逐段标注协议；"P2股流→K5均衡（A-P2-BAL5-001）"一行数值为空。
6. singlecore_baseline_100.csv 与夜版逐字节一致（基准未变，此前核验继续有效）。

## 新发现待澄清（回传师兄，仅剩两项）
1. **stats 三表未随终版更新**：p3_hit_full / p3_scache_full / p12_eval_full 与夜版逐字节相同，但 MANIFEST 的 N2/N3 有 20~41 例逐例变化——三表的 N2/N3 行是旧方案的，请用终版方案重新生成（N4/N5 不变可不动）。影响：命中率/S_Cache 的 N2/N3 值与 P1/P2 的 N2/N3 逐例 mk 表。
2. **ablation_stages 空行**："P2股流→K5均衡"（A-P2-BAL5-001）一行 baseline/candidate/delta/wins/losses 全空，请补数值或说明跳过。

## 结论
除上述两项外，论文所需全部数据已落地并核验。成绩单终值（复算）：P1 1.8068/2.4792/3.1437/**3.7048**；P2 1.9704/2.7601/3.5509/**4.2697**；P3 1.9948/2.7887/3.6367/**4.3091**；命中率 N5=26.8%、S_Cache N5=1.0508（N4/N5 有效，N2/N3 待重生成）。

**目录清理（0926，A028）**：夜版包（`A题_交付数据包_0925夜/`）已删除；独有文件留存至 `results/夜版包留存件/`（stats/superlinear_cases.csv、p1_migrate.csv、speedup_stats.csv（陈旧留证）、singlecore/ 100 份 JSON）；registry 六行路径重定向至终版包。本文件中所有对夜版包路径的引用按留存件或终版对应文件理解。
