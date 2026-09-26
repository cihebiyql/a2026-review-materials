# 复现包 · v5 学习式核指派（两段式：离线 logits 训练 + 在线合规求解）

本包复现论文 A 题（2026 第 23 届）三问全部核心数字的唯一计算管线。方法骨架为
one-shot 优先级采样（Jeon et al., ICLR 2023）的领域化改造：每条链一个 logit，
采样 = logits + i.i.d. Gumbel → argsort 得链全局序（Gumbel-Top-k），解码 =
确定性贪心核指派解码器 + mechA relabel，评估 = FastEval bit-exact 快速评估器。

## 一、环境要求

- Windows / Linux 均可；Python 3.11（实测 3.11.9）
- 依赖：`numpy`、`numba`（`pip install numpy numba`）；harvest 部分另需标准库即可
- **官方附件**（赛题下发，含 `code/` 官方评估器与 `data/` 用例 json），
  通过 `A2026_ATT` 指向其解压根目录（内含 `code/`、`data/` 两个子目录）
- 本包脚本除自带 5 个 .py 外，还从 `A2026_ROOT` 工作区导入共用模块：
  `fast_eval/fast_eval_p1|p2`、`v3_solver/common.py`（官方评估封装与用例加载）、
  `n5_push/superlinear_analysis/phase3_mechA2.py`。若在完整工作区内运行本包，
  上述依赖自动就位；`A2026_ROOT` 需指向该工作区根。
- 单核基准（加速比分母）：`A2026_SC` 指向单核 makespan 目录（每例一个
  `{case}_sc.json`），完整工作区默认在 `{A2026_ROOT}/results/singlecore`。

## 二、环境变量表

| 变量 | 含义 | 默认值 |
|---|---|---|
| `A2026_ROOT` | 工作区根目录（含 fast_eval/、v3_solver/、results/singlecore 等） | `C:/shumo_live/02_求解/A题_2026` |
| `A2026_ATT` | 官方附件根目录（含 `code/`、`data/`） | `C:/shumo_live/a_data` |
| `A2026_SC` | 单核基准目录 | `{A2026_ROOT}/results/singlecore` |
| `A2026_LOGITS` | logits 目录（离线训练产物，在线段读入） | `{A2026_ROOT}/n5_push/reinforce_v5` |
| `A2026_COUT` | 在线求解输出目录 | `{A2026_ROOT}/n5_push/compliant_out` |

次要变量：`A2026_ROUT`（训练落盘目录，默认同 `A2026_LOGITS`）、
`A2026_DECO_PARAMS`（解码器参数 json，默认 `{}` 用内置值）、
`A2026_HARVEST`（收割输出目录，默认脚本旁 `harvest_out/`）、
`NSHARDS` / `V5PY`（run_shard 分片数与解释器）。

## 三、两段式架构说明

**离线段（训练，产 logits）**：`reinforce_assign.py` 对每个 (case, Q) 独立训练。
状态 = 每链一个 logit；每轮 rollout 用 Gumbel-Top-k 采链序，解码器现场生成
op→core 指派（不固定于任何方案池），REINFORCE 梯度更新（批内代价标准化 +
logit 范数正则 c=0.001）。训练产物为 `{case}_q{Q}_logits.json`（本包 `logits/`
共 264 个：q1×96、q2×100、q3×68）。

**在线段（求解，读 logits）**：`solve_compliant.py` 加载该例离线训练的 logits
（模型参数），贪心序解码 + Gumbel 采样 S(k) 取优 → 预算内局部精修（单元搬核/
换核 + B 邻域 + 序扰动爬山）→ 输出方案。全程硬预算（默认 480s，题面"单例
5–10 分钟"口径），**不读任何方案池/冠军**，论文数字唯一来源。

## 四、四条命令复现

以下均以 `py -3.11`（Windows）为例，Linux 换 `python3`；在本 repro/ 目录执行。

1. **管线自测（--stub，秒级）** —— 训练器管线 bit-exact 自测：解码器恒返回冠军
   op_core、评估直读冠军 mk，所有 rollout 应恒等于 mk0，用于验证安装：

   ```
   py -3.11 reinforce_assign.py 1 case_001 --stub --iters 2 --N 4 --workers 2
   ```

2. **单例离线训练（产 logits）**：

   ```
   py -3.11 reinforce_assign.py 3 case_019 --iters 50 --N 128 --workers 48 --seed 0
   ```

   产出 `A2026_ROUT` 下 `case_019_q3_logits.json`（多例逗号分隔可批量）。

3. **单例在线求解（论文数字来源）**：

   ```
   py -3.11 solve_compliant.py case_019 3 5 --budget_s 480 --samples 256 --seed 0
   ```

   输出 `A2026_COUT` 下 `case_019_q3_N5_compliant.json`（方案 + 指标 + rec）。
   **`--official-eval` 豁免通道**：个别 numba 崩溃类图改走纯 Python 官方评估器
   （`ev_p1/ev_p2/ev_p3`，来自官方附件 code/），评估口径与 FastEval 逐项对齐；
   仅对 FastEval 无法原生评估的用例使用，常规用例不需要。

4. **全库批量（run_shard.py，分片驱动）**：

   ```
   py -3.11 run_shard.py <shard_id>     # shard_id ∈ [0, NSHARDS)
   ```

   分片 i 处理 `tasks.txt` 的第 `i::NSHARDS` 行（每行 `Q case`；tasks.txt 需按
   此格式自备，本包未附）。每任务先训练存 logits，再对 N=2..5 合规求解，
   状态写 `{A2026_COUT}/shard_{i}_status.jsonl`。

收割/汇总（附录表、曲线数据、耗时统计）：`py -3.11 harvest_compliant.py`
（读 `A2026_COUT` 全部 `*_compliant.json` 重评验证后产 csv/summary）。

## 五、AI 辅助声明

本程序及代码是在人工智能工具辅助下完成的。
人工智能工具名称：GLM，版本/型号：zai-api/GLM-5.3，开发机构/公司：智谱AI(Z.ai)，
版本颁布日期：2026-09-25。
每个 .py 文件头部均带同一声明，请勿删除。

## 六、方法出处（引用）

- Jeon W, Gagrani M, Bartan B, Zeng W W, Teague H, Zappi P, Lott C.
  Neural DAG Scheduling via One-Shot Priority Sampling. ICLR, 2023.
  （one-shot 优先级采样思想：Gumbel-Top-k 采样 + REINFORCE 批内代价标准化；
  本文将其领域化为 DAG 算子链的核指派问题，解码层为自研贪心指派解码器。）

## 依赖部署(2026-09-26 补齐)

`deps/` 内含运行所需全部非官方依赖, 复现时目录布局:
```
工作区/
  repro/            ← 本目录全部 .py
  deps/fast_eval/   ← 复制到 工作区/fast_eval/
  deps/v3_solver/   ← 复制到 工作区/v3_solver/
  deps/superlinear_analysis/ ← 复制到 工作区/n5_push/superlinear_analysis/
```
即 `A2026_ROOT=工作区`。官方附件(评估器 code/ 与用例 data/)仍需按题面自备并设 `A2026_ATT`。
新增脚本: v5_isolated_eval(隔离评估)/final_table+final_worker(终版总表)/reconcile_pool(池对账)/package_plans(附录B打包)/make_gap_table/v5_ablation(消融)/paper_figs(三图)/pool_harvest+pool_worker(池次指标)。

## 从零一键复现(2026-09-26 终版补齐)

单核基准已随包: `deps/singlecore/*_sc.json`(100 例, 官方评估器产物, 亦可自行用附件 singlecore_evaluate.py 重算)。
部署后按序执行即可全量复现(在线轨全部数字):
```
# 0) 布局: 工作区/{fast_eval,v3_solver,n5_push/superlinear_analysis(含deps与池方案),singlecore,repro各py}
export A2026_ROOT=工作区 A2026_ATT=官方附件 A2026_SC=工作区/singlecore A2026_LOGITS=logits目录
# 1) 管线自测(bit-exact)  2) 全库训练(可选, logits已随包; 训练需池种子=champion_pool)
# 3) 全库求解(1200)       4) 消融批            5) final_table.py 总表      6) paper_figs.py 三图
```
复现层级: 在线轨/评估/表格 100% 逐位复现(同种子); 最终方案轨为"方案→成绩"可验证(方案随包)。
