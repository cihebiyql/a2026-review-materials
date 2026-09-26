# -*- coding: utf-8 -*-
"""s5_assign.core_assign — 分核。

  - round_robin：V1 的轮询分核（在 convex_cut 内联实现，此处仅接口）。
  - decoder_assign：V4 学习式指派的解码层——按给定链序，用双管线负载
    均衡贪心解码器（领域知识内嵌）生成 op->核 映射。
    解码器本体在 s5_assign/decoder.py（零第三方依赖）。
"""
import paths  # noqa: F401  (装配 DEPS 到 sys.path)
from s5_assign.decoder import chain_features, assign_chain_cores


def decoder_assign(graph, comp_of, chains_order, num_cores, q, params=None):
    """按链序解码核指派。返回 op->core。"""
    feats = chain_features(graph, comp_of)
    return assign_chain_cores(list(chains_order), feats, comp_of,
                              num_cores, q, params or {})
