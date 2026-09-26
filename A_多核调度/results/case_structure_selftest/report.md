# 算例结构画像报告

- 用例数：8
- 类型分布：{"deep_chain": 3, "multi_component": 1, "converge_tree": 1, "other": 3}
- 类型占比：{'deep_chain': '37.5%', 'multi_component': '12.5%', 'converge_tree': '12.5%', 'other': '37.5%'}
- 存在跨分量共享张量的用例：1
- 存在外部共享输入张量（权重型）的用例：1
- 含直接 op→op 边的用例（X01 证据）：0（共 0 条）

| case | 计算op | 分量数 | 最大分量占比 | 最长路径 | 汇聚占比 | 类型 |
|---|---|---|---|---|---|---|
| synthetic_deep_chain | 20 | 1 | 1.00 | 20 | 0.00 | deep_chain |
| synthetic_multi_comp | 50 | 10 | 0.10 | 5 | 0.00 | multi_component |
| synthetic_reduce_tree | 31 | 1 | 1.00 | 5 | 0.48 | converge_tree |
| CE-001-layer-interleave-cycle | 4 | 2 | 0.50 | 2 | 0.00 | other |
| CE-002-same-core-order | 4 | 1 | 1.00 | 4 | 0.00 | deep_chain |
| CE-003-wavefront-bytes | 12 | 6 | 0.25 | 2 | 0.00 | other |
| CE-004-deep-chain | 40 | 1 | 1.00 | 40 | 0.00 | deep_chain |
| CE-005-single-sink | 31 | 1 | 1.00 | 11 | 0.03 | other |