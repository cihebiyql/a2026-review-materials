# A 题提交包(2026-09-25 冻结)

## 内容

- `plans/q{1,2,3}_N{2,3,4,5}/case_XXX_multicore_res.json`:3 问 × 100 例 × 4 核数 = **1200 个方案文件**(官方 `node_to_subgraph + core_schedules` 格式,与 `<计算图文件名>_multicore_res.json` 约定一致,可直接被三个官方评估器读取)
- `MANIFEST.csv`:逐例 mk / sp / 来源溯源(archive=plan_archive 档案、audit=真值精修线、rerun=重跑恢复、x2/session=预算加倍池)
- 本 README

## 成绩单(MANIFEST 全量口径 = 论文 numbers.tex 同源)

| | N2 | N3 | N4 | N5 |
|---|---|---|---|---|
| P1 | 1.795 | 2.451 | 3.123 | 3.606 |
| P2 | 1.944 | 2.718 | 3.459 | **4.157** |
| P3 | 1.978 | 2.754 | 3.525 | **4.244** |

## 验收记录

- 官方评估器逐例复核 **39 例抽样全一致**(mk 逐位相同;覆盖 3 问 × 4 核数、全部 5 类来源)
- N5 全量干跑:q2/q3 各 100 例复算通过(方案-分数一致)
- 注错验收 5/5(a_lab acceptance:漏分配/商图环/超时/配置不匹配/删行)
- 合法性:harness R1-R5 全量判定(1200 例 0 违规,plan_cores=N 逐例校验)

## 再生成

`py -3.11 a_lab/freeze_deliverables.py`(六层来源:plan_archive → 审计线 14 目录 → 重跑 10 组 → X2 池 → 会话优胜 18 例;统一 sp=T1/mk 口径取优)
