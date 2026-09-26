# 终版交付包核验·分记录（2026-09-25 23:15 到包，AI-08 会话）

**定位**：本文件是 [manifest_audit_20260925/summary.md](manifest_audit_20260925/summary.md) 终版补记节（并行 AI-05 会话，权威核验记录）之外的补充材料，只保留 AI-08 会话独有的复算明细与解读：消融阶梯的组件叙事解读、甘特明细、B8 终版复核、N5 source 构成。RES 状态以 02 §1.8 终版段（AI-05/A026）为准——**RES-011/012 已升 CONFIRMED（终版数值就地更新，未另设 RES-017）**，RES-013/014 维持 PRELIMINARY（N4/N5 已核、**N2/N3 行过期**：stats 三表与夜版逐字节相同而 MANIFEST N2/N3 有 20~41 例变化，问清单 #8）。

对象：`../../../../A题_交付数据包_0925终版/`（替换已删除的 `A题_交付数据包_0925夜/`，夜版独有文件留存于 [results/夜版包留存件/](夜版包留存件/)）。MANIFEST 列：`case,q,N,mk,sp,source`。环境：Git Bash awk。AI-08 独立复算的 12 格均值、P3 口径、单核抽查与 AI-05 结果一致，不在此重复。

## 1. 包内容相对夜版的变化

`MANIFEST_终版.csv`（空 mk 清零）；更新版 `speedup_stats_v4.csv`（与 MANIFEST 12/12 一致，夜版两处不一致消除）；`singlecore_baseline_100.csv`（取代 100 份 JSON，与夜版逐字节一致）；stats 三表（**未随 N2/N3 再生，见 #8**）；新增 `ablation_stages.csv` 与 `gantt/` 六份 timeline。夜版随包的 `回复_数据确认清单.md`、陈旧 `stats/speedup_stats.csv`、`superlinear_cases.csv`、`p1_migrate.csv` 不再随包。

## 2. 消融阶梯的组件叙事解读（#4 交付；问清单 #7 据此收窄）

列：`stage,exp,baseline_mean,candidate_mean,delta,wins,losses`，十行两组：

- N4 历史阶梯：v3前缀→v3波前（2.9688→3.0433，30胜6负）；wave→STRAND（3.04→3.0664）；STRAND→MERGE全评估（3.0664→3.1148，55胜0负，＝RES-007）；wave→P2股流（3.3841→3.4504，＝RES-008）；wave→P3股流（3.4511→3.5065，4胜7负，**＝RES-009 未晋级对照，入表须标注**）；wave→P3精修基线（3.436→3.4511）；P2-BAL5 一行空值（#9）。
- N5 LNS 阶梯：P1 MERGE 3.587→**3.7048**（84胜0负）；P2 **G5TF** 4.156→**4.2697**（74胜0负）；P3 池 4.2441→**4.3091**（67胜0负）。

解读：终版主线组件阶梯首次显式给出——P1＝wave→STRAND→MERGE→LNS；P2＝wave→股流→**G5TF**→LNS；P3＝wave→股流→精修→池→LNS。**mechA 仍未以本名出现于任何 source/实验名**；#7 收窄为三小问：G5TF 与 batch×level/mechA 的对应关系、"单臂可复现管线"与 source 五类逐例来源的择优规则、P3"池"在管线中的角色。各段为独立 A/B 实验、N4/N5 期混合，拼消融表须逐段标协议。

## 3. 甘特 timeline 明细（#3 交付；F03 解锁）

列：`case,ver(pre/post),core,op_id,op,pipe,start,end,duration,mk`（逐操作时序）。

| 案例 | pre mk | post mk | post 对应 |
|---|---|---|---|
| case_008 | 100603 | 73515 | P2 N5＝73515（audit:gpu_climb）✓ |
| case_064 | 19525 | 14916 | P2 N5 对账一致（AI-05） |
| case_084 | 502466 | 378550 | P2/P3 N5＝378550 ✓（两问同方案，Cache 零贡献） |

## 4. 超线性例数终版复核（B8/#6）

`awk` 终版计数：**P1=4、P2=24、P3=21**（与夜版一致）；两版数据下"sp>5 共 14 例"均不可复现。写作按 MANIFEST 口径，待师兄晨间确认。

## 5. N5 source 构成（与夜版一致）

P1＝climb:P1-LNS(53)+session:MERGE-N5(26)+archive:v2_cpc(12)+audit:strand_n5(7)+v3_main(2)；P2＝audit:gpu_climb(53)+x2:A-P2-G5TF-X2(19)+…；P3＝audit:gpu_climb_p3(36)+x2:A-P3-G5TF-X2(17)+…。N2 出现新来源 `climb:LNS-N2`（如 case_002）。

## 6. 局限

静态复算（聚合与交叉比对），未重跑官方评估器；终版无随包书面说明，"终版＝最终提交版"与 #6~#9 待师兄晨间确认（问清单已冻结至 0926 晨）。
