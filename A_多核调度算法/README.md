# A题 多核调度求解器 —— 统一管线总包（唯一交付版本）

> 2026 研究生数模 A 题《通用神经网络处理器下的多核调度》。
> **一个包 = 一个方法 = 一条命令**：从计算图输入开始，经 版本阶梯
> V1→V4 四层算法逐层改进，对 100 个官方用例 × 三问 × N∈{2,3,4,5} 全量
> 求解，自动产出赛题要求的全部交付物（折线图 / 逐例表 / 汇总 / 运行时间）。
> 无冠军池、无逐例超参——任何数字都可由本包一键复现。

---

## 一、快速开始（一步）

官方赛题附件（评估器代码 + 100 用例 + config.txt）已内置在 `official/`，
单核基准已内置在 `sc/`——**解包即跑，无需任何配置**：

```bash
bash run_all.sh          # 一键: 全库计算 + 出全部表和图(断点续跑可中断重跑)
ls results/tables/       # 逐例表+汇总+增益+失败分析 (CSV)
ls results/figs/         # 正文图 (PNG)
```

> ⚠ official/ 为官方竞赛材料，仅供本队使用，请勿公开传播。
> 想改用外部路径时才需要 env.sh（见 env.example.sh）。

全量约需数小时（视机器核数，见 §8 时间参考）。等不及可先跑单例体验：

```bash
source env.sh
python3 pipeline.py case_001 3 --nlist 5 --versions v1,v2,v3,v4   # 单案例全链条
```

## 二、环境要求

| 依赖 | 要求 | 说明 |
|---|---|---|
| Python | ≥3.10 | 容器/服务器/本机均可 |
| numpy + numba | 建议 numba 0.60 | FastEval 加速反馈需要 |
| 官方附件 | **包内自带 `official/`** | code/(官方评估器) + data/(100用例+config.txt) + docs/，一字未改 |
| 单核基准 | **包内自带 `sc/`** | 官方 singlecore_evaluate 产出；`tools/regen_sc.py` 可再生 |

**无 numba 也能跑**：`export PIPELINE_EVAL=official` 后用官方评估器直接当
反馈（纯 Python，永不崩，慢 10~50×）——适合本机小规模调试。

## 三、命令速查

| 目的 | 命令 |
|---|---|
| 一键全量(推荐) | `source env.sh && bash run_all.sh` |
| 只跑某一问 | `python3 run_batch.py --qs 3` (Q3；`--qs 3,2,1` 全部) |
| 只跑到某版本 | `python3 run_batch.py --stop-at v2` (v1/v2/v3/v4) |
| 指定案例调试 | `python3 run_batch.py --only case_095,case_084` |
| 单案例单问全链条 | `python3 pipeline.py case_095 3 --nlist 5 --versions v1,v2,v3,v4` |
| 单案例只跑 V1 快看 | `python3 pipeline.py case_095 3 --nlist 5 --versions v1` |
| 指定核数 | `--nlist 5,4,3,2`（默认即此；N=1 由定义 sp≡1 不跑） |
| 中断后续跑 | 直接重跑同一条命令——已完成部分自动跳过 |
| 只重新出表/图 | `PYTHONPATH=. python3 s9_evidence/make_tables.py && PYTHONPATH=. python3 s9_evidence/make_curves.py && PYTHONPATH=. python3 s9_evidence/failure_report.py` |
| 纯官方评估模式(免numba) | `export PIPELINE_EVAL=official` 后照常跑 |

关键参数（统一协议，全例一致，改这里=改方法超参）：
`run_batch.py --lanes-a 32 --lanes-b 8 --lanes-rescue 32 --v4-workers 16 --v4-iters 50`；
V2/V3 搜索网格在 `s7_search/sweep_grid.py` 顶部；V1 切块密度
`PIPELINE_CPC`(默认 8)。

## 四、方法详解（一个方法 = 九环节流水线 × 四层改进阶梯）

### 4.1 九环节流水线（代码即目录）

```
① s1_preprocess   DAG 数据预处理: 官方JSON → Op-Op压缩依赖图(COPY收缩,
                  与官方 step1 同构); 周期/管道/张量视图
② s2_features     DAG 特征分析: 弱连通链分解 / 深度 / M/V双管负载 / 图统计
③ s3_units        合法运算顺序与结构单元: 链跟随Kahn拓扑序(依赖相邻→少切边);
                  链(辫状图退化时自动切拓扑段, 段序即拓扑序保证合法)
④ s4_partition    切图: 沿拓扑序切 8×N 个连续凸块(周期贪心均衡;
                  凸分区⇒商图无环+同核序天然合法)
⑤ s5_assign       分核: 轮询(V1) / 双管均衡贪心解码器(V4)
⑥ s6_order        排序: B×L重标(V2) / 三结构(V3) / 采样序重标(V4训练)
⑦ s7_search       候选搜索: 确定性网格(V2:590候选, V3:144候选) /
                  REINFORCE采样(V4: 50轮×128采样)
⑧ s8_evaluate     评估/选择: FastEval真值反馈(官方评估器numba bit-exact
                  复刻) + 终选方案官方评估器复核(official_match列)
⑨ s9_evidence     实验证据: 逐例表/折线图/版本消融/失败分析
```

### 4.2 四层改进阶梯（V_k 严格以 V(k-1) 最优方案为种子，结果单调不劣）

| 层 | 增量 | 核心机制 |
|---|---|---|
| **V1 凸块基线** | 纯构造，无搜索 | 拓扑序+连续凸块+周期均衡+轮询分核；秒级 |
| **V2 B×L 重标** | +590 候选网格深扫 | op→核不变，只重排子图标签：batch(链组,宽度B)×level(L层窗) 网格——标签即官方核内调度器的分桶键，批内跨链交错解锁 M/V 双管并发，层窗控制张量驻留宽度(控 spill) |
| **V3 三结构** | +144 候选 | ①均衡装箱：当前批缺 M 补 M、缺 V 补 V(重M链配重V链) ②跨核相位：各核 batch 序按核索引旋转，错开 DDR 争用峰值 ③层切分DP：代理代价=段内管道失衡×段长+切点惩罚λ=0.35 的 O(8n) 动态规划，替代固定 L |
| **V4 学习式核指派** | +REINFORCE 训练 | ICLR23 one-shot priority sampling 移植到指派层：每链一个 logit → logits+Gumbel → argsort=单元全局序(Gumbel-Top-k) → 确定性双管均衡解码器生成**新的 op→核指派** → B×L 重标 → FastEval 真值 → REINFORCE(批内代价标准化+logit范数正则c=0.001+Adam lr=0.05)；best-so-far+greedy 双轨追踪；logit 围绕 V3 输出诱导序热启动 |

统一协议（全例一致，无逐例调参）：V2 网格 L∈{2,3,4,6,8}×B∈{4..238 步2}；
V3 网格 12cfg×12B；V4 B∈{16,48,96,160,240}、L=4、50轮×128采样、seed=0。

### 4.3 关键合法性/合规性设计

- 凸分区+段序构造保证商图无环、同核序合法（评估器校验恒过）；
- 辫状图（链<96）自动切拓扑段，段模式用"全局排名相邻且同核"合法批
  （商图无环引理，见 `s6_order/relabel.py` docstring）；
- 搜索反馈 FastEval 与官方评估器 bit-exact（终选方案 official_match
  复核列全 True 为验收标准）；
- 逐例训练合规：100 测例为题目给定输入，对每个输入做优化是优化题本分；
  V4 离线训练+在线秒级出解符合题面"合理时间内"口径。

## 五、目录结构

```
A题_多核调度求解器/
├── README.md            本文件
├── env.example.sh       可选覆盖模板(默认零配置)
├── official/            ★官方赛题附件(内置): code/官方评估器+data/100用例+config.txt+docs/
├── run_all.sh           一键全量入口
├── paths.py             路径装配(读环境变量)
├── pipeline.py          单(case,Q)主流程: N列表 × 版本阶梯
├── run_batch.py         全库调度(独立子进程车道+崩图救援梯度)
├── s1..s9/              九环节模块(见 §4.1)
├── fasteval/            FastEval三问内核(fast_eval_p1/p2+kernels) + official_eval.py(官方评估直调封装)
├── sc/                  单核基准缓存(100例, 官方评估器产出; tools/regen_sc.py 可再生)
├── tools/regen_sc.py    单核基准再生(可选)
└── results/
    ├── runs/            每(case,Q)一个jsonl: 逐(version,N)完整记录
    ├── plans/           每(case,Q,N,version)终选方案(赛题提交格式json)
    ├── tables/          CSV: 逐例表/汇总/增益分布/失败分析
    ├── figs/            PNG: 1~5核折线/Q3两配置对比/版本消融
    └── driver_logs/     每子进程运行日志(排错用)
```

## 六、输出说明（严格对齐赛题要求）

**正文图 `results/figs/`**
- `core_scaling_q{1,2,3}.png`：1~5 核平均加速比折线（最终版 V4 实线 +
  V1 基线虚线，数值标注；题面"单核加速比定义为 1"）
- `q3_cache_vs_nol2.png`：题三要求的无 L2 vs 只读 Cache 两配置对比曲线
  + 相同核数下 Cache 加速比标注
- `ablation_ladder.png`：N=5 版本阶梯消融柱状图

**附录逐例表 `results/tables/percase_q{Q}_N{N}_{ver}.csv`**
| 列 | 含义 |
|---|---|
| makespan | 官方评估器 Makespan（cycles） |
| speedup | 单核 Makespan / N 核 Makespan |
| added_MB | 总额外数据搬运量（题面次要指标） |
| hit_rate / mk_nol2 / added_nol2_MB / cache_speedup | 仅 Q3：Cache 命中率、同方案无 L2 配置、Cache 加速比 |
| solve_time_s | **该版本该案例的算法运行时间**（题面"合理时间内"口径） |
| official_match | FastEval 与官方评估器是否逐位一致（应全 True） |
| status | ok / FAIL / v4_skipped_flaky_inherit_v3（崩图继承标注） |

**权威证据记录 `records_official.csv`**（run_all 第 5 步自动生成）：
对 `results/plans/` 全部终选方案做**官方评估器单遍重评**（与求解过程
解耦），逐行字段含 makespan、单核基准、speedup、总额外搬运及其
partition/spill 分量、q3 命中率与两配置对比、official_seconds 与
solve_seconds（算法运行时间）——论文附录逐例表的直接数据源；
`records_summary.csv` 为 problem×cores×version 分组统计。

**汇总 `summary.csv`**：Q×N×版本 → 平均加速比/平均搬运/平均与最大
运行时间/改进例数/失败数。
**增益分布 `version_gain.csv`**：每版本 Δsp 分布 + 链富集图占比（方法边界）。
**失败分析 `failures.csv`**：崩图记录与图特征。

**方案文件 `results/plans/`**：`{case}_multicore_res` 同格式
（node_to_subgraph + core_schedules），可直接喂官方评估脚本复核：
`python code/multicore_cut_evaluate_problem_3.py data/case_095.json --config data/config.txt <plan 转存路径>`。

## 七、健壮性机制

1. **单案例单进程**：FastEval 的 numba 内核存在"特定图×方案形状"的
   堆损坏（跨 numba 0.59/0.60 复现，glibc 2.39 下必现 abort）。每
   (case,Q) 一个完全独立子进程，单点崩溃不传染。
2. **崩图救援梯度**：正常(FastEval) 崩溃 → 自动改用官方评估器当反馈
   重跑（纯 Python 永不崩，慢 10~50×，32 条救援道并发）。
3. **V4 对崩图案例自动跳过**（6400 次评估训练在慢反馈下不可行），
   表格以 V3 结果继承并标注 `v4_skipped_flaky_inherit_v3`。
4. **断点续跑**：以 runs/*.jsonl 已有完整记录为准跳过；重复记录取最后一行。
5. **两阶段调度**：阶段A(v1-v3, 单核/道×32道) → 阶段B(v4, 16worker
   fork 池×8道=128核)，numba JIT 每进程只编译一次。

## 八、时间与资源参考（并行超算云 128 核容器实测）

| 阶段 | 耗时 |
|---|---|
| V1 全库 | 分钟级 |
| V2+V3 全库 | ~1-2 h（32 道；大图如 016/041 是长尾） |
| 崩图救援（约半数案例触发时） | ~2-4 h（32 道官方反馈） |
| V4 全库 | ~2-6 h（8 道×16 worker） |
| 出表出图 | 分钟级 |

单例参考：中小案例 V1<1s、V2 约 0.5-2 min、V3 约 10-30s、V4 约 0.5-5 min
（含 JIT）；makespan 求值本身毫秒-秒级，符合题面合理时间口径。

## 九、常见问题

- **`corrupted double-linked list` / 子进程 rc=-6**：预期内的 numba 内核
  崩溃，救援梯度会自动接管，无需干预；看 `results/driver_logs/` 排细节。
- **想完全避开崩溃**：`export PIPELINE_EVAL=official`（慢）。
- **`No module named 'xxx'`**：确认在包根目录运行、已 `source env.sh`、
  `PYTHONPATH=.`（run_all.sh 已内置）。
- **换机器迁移**：拷整个包 + 改 env.sh 两个路径即可；sc/ 随包走。
- **如何确认结果可信**：percase 表 official_match 列全 True；
  任取 results/plans/ 下方案用官方脚本独立复评（§6 末命令）。

## 十、与团队既有代码的关系

本包的 V2/V3/V4 算子与团队 09-25 战役版（mechA_deep / mechA_v4 /
v5 reinforce_assign）关键函数逐行同构，但种子策略不同：战役版从历史
冠军池逐例择优热启动（数字无法单法复现）；**本包从 V1 基线逐层累积，
一条命令完整复现论文主表的每一个数字**。战役池结果可作为"方法组合
可达上界"的对照参考，不进入本包主口径。
