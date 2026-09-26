# -*- coding: utf-8 -*-
# 本程序及代码是在人工智能工具辅助下完成的。
# 工具：ZCode；版本/型号：GLM-5.3；开发机构：智谱（Z.ai）；版本发布日期：待补。
"""官方算例结构画像统计（论文 2.x「测试算例结构分析」的数据源脚本）。

输入：符合官方评估器格式的计算图 JSON（ops/tensors/edges），或包含若干
计算图 JSON 的目录。输出逐用例指标 CSV、汇总 JSON 和可读 Markdown 报告。

统计口径（与官方评估器对齐，X01 冲突下采用代码语义）：
  1. 邻接构建逐字移植 stub_multicore_cut_and_schedule._build_op_adjacency：
     tensor 中转边（op→tensor→op）与直接 op→op 边统一转为 op DAG；
     直接 op→op 边数单独输出，作为 X01（题面 vs 代码）的用例级证据。
  2. COPY_IN/COPY_OUT 节点按 _contract_excluded_copy_nodes 语义收缩旁路，
     结构统计在收缩后的计算 op 图上进行。
  3. 弱连通分量按无向连通性计算；「跨分量共享张量」指被 ≥2 个不同弱连通
     分量中的计算 op 消费的张量（不依赖任何具体切分方案）。
  4. 「外部共享输入张量」（权重型）指生产者全为 COPY_IN 且被 ≥2 个计算 op
     消费的张量；该口径服务第六章 L2 复用分析。
  5. 类型判定阈值可用 --rules 覆盖；原始指标始终全量输出，改阈值不必重算。

分类规则（默认，按优先级）：
  multi_component  n_components ≥ 8 且最大分量 op 占比 ≤ 0.5
  converge_tree    n_components ≤ 2 且 sink 数 ≤ 2 且汇聚节点占比 ≥ 0.25
                   且分支节点（fan-out ≥ 2）占比 ≤ 0.25
  deep_chain       最长路径节点数 ≥ 0.6 × 计算 op 数
  other            其余

运行环境：Python ≥ 3.9，仅标准库。
用法（相对工作区 A_多核调度/）：
  python scripts/case_structure_profile.py <官方材料包data目录> --out results/case_structure
  python scripts/case_structure_profile.py --selftest          # 合成图+反例图自测
"""

import argparse
import csv
import json
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

COPY_TYPES = {'COPY_IN', 'COPY_OUT'}
PIPES = ('PIPE_MTE2', 'PIPE_MTE3', 'PIPE_M', 'PIPE_V')

DEFAULT_RULES = {
    'multi_component_min_components': 8,
    'multi_component_max_largest_share': 0.5,
    'converge_max_components': 2,
    'converge_max_sinks': 2,
    'converge_min_merge_share': 0.25,
    'converge_max_branch_share': 0.25,
    'deep_chain_path_ratio': 0.6,
}

METRIC_FIELDS = [
    'case', 'n_ops_total', 'n_ops_compute', 'n_copy_in', 'n_copy_out',
    'n_tensors', 'n_edges', 'n_direct_op_edges', 'total_cycles',
    'cycles_pipe_m', 'cycles_pipe_v', 'cycles_pipe_mte2', 'cycles_pipe_mte3',
    'total_tensor_bytes', 'ddr_tensor_bytes',
    'n_components', 'largest_component_ops', 'largest_component_share',
    'components_ge10', 'top_component_sizes',
    'n_sources', 'n_sinks', 'max_fan_in', 'merge_ops', 'merge_share',
    'branch_ops', 'branch_share', 'longest_path_nodes', 'path_ratio',
    'n_shared_tensors', 'shared_tensor_bytes', 'shared_tensor_bytes_share',
    'n_external_shared_tensors', 'external_shared_bytes',
    'n_cross_component_shared_tensors', 'cross_component_shared_bytes',
    'max_tensor_fanout', 'sig_top_coverage', 'sig_effective_count',
    'graph_class',
]


class ProfileError(RuntimeError):
    """输入图无法完成结构统计。"""


def build_op_adjacency(graph):
    """op DAG 构建，逐字移植评估器 _build_op_adjacency（含直接 op→op 边）。"""
    op_ids = {op['id'] for op in graph.get('ops', [])}
    preds = {op_id: set() for op_id in op_ids}
    succs = {op_id: set() for op_id in op_ids}
    producers = defaultdict(set)
    consumers = defaultdict(set)
    n_direct_op_edges = 0
    for edge in graph.get('edges', []):
        src, dst = edge['source'], edge['target']
        src_is_op, dst_is_op = src in op_ids, dst in op_ids
        if src_is_op and dst_is_op and src != dst:
            if dst not in succs[src]:
                n_direct_op_edges += 1
            succs[src].add(dst)
            preds[dst].add(src)
        elif src_is_op and not dst_is_op:
            producers[dst].add(src)
        elif not src_is_op and dst_is_op:
            consumers[src].add(dst)
    for tensor_id, producer_ids in producers.items():
        for src in producer_ids:
            for dst in consumers.get(tensor_id, ()):
                if src != dst:
                    succs[src].add(dst)
                    preds[dst].add(src)
    return preds, succs, n_direct_op_edges, producers, consumers


def contract_copy_nodes(eligible_ids, succs):
    """COPY_IN/COPY_OUT 收缩，移植评估器 _contract_excluded_copy_nodes。"""
    eligible = set(eligible_ids)
    contracted_succs = {node_id: set() for node_id in eligible_ids}
    contracted_preds = {node_id: set() for node_id in eligible_ids}
    for src_id in eligible_ids:
        stack = list(sorted(succs[src_id], reverse=True))
        visited_excluded = set()
        while stack:
            dst_id = stack.pop()
            if dst_id in eligible:
                if dst_id != src_id:
                    contracted_succs[src_id].add(dst_id)
                    contracted_preds[dst_id].add(src_id)
                continue
            if dst_id in visited_excluded:
                continue
            visited_excluded.add(dst_id)
            stack.extend(sorted(succs.get(dst_id, ()), reverse=True))
    return contracted_preds, contracted_succs


def weak_components(nodes, succs):
    """无向连通分量；返回分量列表（每项为节点集合）。"""
    neighbors = {n: set() for n in nodes}
    for src, dsts in succs.items():
        if src not in neighbors:
            continue
        for dst in dsts:
            if dst in neighbors:
                neighbors[src].add(dst)
                neighbors[dst].add(src)
    seen = set()
    components = []
    for start in nodes:
        if start in seen:
            continue
        comp = {start}
        stack = [start]
        seen.add(start)
        while stack:
            cur = stack.pop()
            for nxt in neighbors[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    comp.add(nxt)
                    stack.append(nxt)
        components.append(comp)
    return components


def longest_path(nodes, preds, succs):
    """拓扑序 DP 求最长路径节点数；有环抛 ProfileError。"""
    indeg = {n: len(preds[n]) for n in nodes}
    ready = [n for n in nodes if indeg[n] == 0]
    depth = {n: 1 for n in nodes}
    order = []
    while ready:
        cur = ready.pop()
        order.append(cur)
        for nxt in succs[cur]:
            depth[nxt] = max(depth[nxt], depth[cur] + 1)
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                ready.append(nxt)
    if len(order) != len(nodes):
        raise ProfileError('计算 op 图含环，无法统计')
    return max(depth.values()) if depth else 0


def structural_signature(op_id, preds, succs, graph_ops, tensor_edges):
    """轻量结构签名：pipe、扇入扇出、周期、按大小排序的出入 tensor。"""
    op = graph_ops[op_id]
    in_sizes = sorted(tensor_edges[('in', op_id)])
    out_sizes = sorted(tensor_edges[('out', op_id)])
    return (op['pipe'], op['cycles'], len(preds[op_id]), len(succs[op_id]),
            tuple(in_sizes), tuple(out_sizes))


def classify(m, rules):
    if (m['n_components'] >= rules['multi_component_min_components']
            and m['largest_component_share'] <= rules['multi_component_max_largest_share']):
        return 'multi_component'
    if (m['n_components'] <= rules['converge_max_components']
            and m['n_sinks'] <= rules['converge_max_sinks']
            and m['merge_share'] >= rules['converge_min_merge_share']
            and m['branch_share'] <= rules['converge_max_branch_share']):
        return 'converge_tree'
    if m['path_ratio'] >= rules['deep_chain_path_ratio']:
        return 'deep_chain'
    return 'other'


def profile_case(path, rules):
    graph = json.loads(Path(path).read_text(encoding='utf-8'))
    stem = Path(path).stem
    case_name = Path(path).parent.name if stem == 'graph' else stem
    ops = graph.get('ops', [])
    tensors = graph.get('tensors', [])
    ops_by_id = {op['id']: op for op in ops}
    tensors_by_id = {t['id']: t for t in tensors}
    op_type = {op['id']: op.get('op', '') for op in ops}
    eligible_ids = [op['id'] for op in ops if op_type[op['id']] not in COPY_TYPES]

    preds, succs, n_direct, producers, consumers = build_op_adjacency(graph)
    cpreds, csuccs = contract_copy_nodes(eligible_ids, succs)

    m = {'case': case_name}
    # A. 规模
    m['n_ops_total'] = len(ops)
    m['n_ops_compute'] = len(eligible_ids)
    m['n_copy_in'] = sum(1 for i in ops_by_id if op_type[i] == 'COPY_IN')
    m['n_copy_out'] = sum(1 for i in ops_by_id if op_type[i] == 'COPY_OUT')
    m['n_tensors'] = len(tensors)
    m['n_edges'] = len(graph.get('edges', []))
    m['n_direct_op_edges'] = n_direct
    cycles_by_pipe = Counter()
    for op in ops:
        if op['id'] in set(eligible_ids):
            cycles_by_pipe[op['pipe']] += op['cycles']
    m['total_cycles'] = sum(cycles_by_pipe.values())
    m['cycles_pipe_m'] = cycles_by_pipe['PIPE_M']
    m['cycles_pipe_v'] = cycles_by_pipe['PIPE_V']
    m['cycles_pipe_mte2'] = cycles_by_pipe['PIPE_MTE2']
    m['cycles_pipe_mte3'] = cycles_by_pipe['PIPE_MTE3']
    m['total_tensor_bytes'] = sum(t['size'] for t in tensors)
    m['ddr_tensor_bytes'] = sum(t['size'] for t in tensors if t.get('pos') == 'DDR')

    # B. 弱连通分量
    components = weak_components(set(eligible_ids), csuccs)
    components.sort(key=len, reverse=True)
    m['n_components'] = len(components)
    m['largest_component_ops'] = len(components[0]) if components else 0
    m['largest_component_share'] = (
        m['largest_component_ops'] / m['n_ops_compute']) if m['n_ops_compute'] else 0.0
    m['components_ge10'] = sum(1 for c in components if len(c) >= 10)
    m['top_component_sizes'] = '+'.join(str(len(c)) for c in components[:5])

    # C. 形态
    comp_of = {}
    for idx, comp in enumerate(components):
        for n in comp:
            comp_of[n] = idx
    n = m['n_ops_compute']
    m['n_sources'] = sum(1 for i in eligible_ids if not cpreds[i])
    m['n_sinks'] = sum(1 for i in eligible_ids if not csuccs[i])
    m['max_fan_in'] = max((len(cpreds[i]) for i in eligible_ids), default=0)
    m['merge_ops'] = sum(1 for i in eligible_ids if len(cpreds[i]) >= 2)
    m['merge_share'] = m['merge_ops'] / n if n else 0.0
    m['branch_ops'] = sum(1 for i in eligible_ids if len(csuccs[i]) >= 2)
    m['branch_share'] = m['branch_ops'] / n if n else 0.0
    lp = longest_path(eligible_ids, cpreds, csuccs)
    m['longest_path_nodes'] = lp
    m['path_ratio'] = lp / n if n else 0.0

    # D. 共享张量（生产者/消费者在原始图上取，消费者只数计算 op）
    eligible_set = set(eligible_ids)
    n_shared = shared_bytes = 0
    n_ext_shared = ext_bytes = 0
    n_cc_shared = cc_bytes = 0
    max_fanout = 0
    for t in tensors:
        tid = t['id']
        prod = producers.get(tid, set())
        cons = consumers.get(tid, set()) & eligible_set
        if not cons:
            continue
        max_fanout = max(max_fanout, len(cons))
        if len(cons) >= 2:
            n_shared += 1
            shared_bytes += t['size']
            if prod and all(op_type[p] == 'COPY_IN' for p in prod):
                n_ext_shared += 1
                ext_bytes += t['size']
            spanned = {comp_of[c] for c in cons}
            if len(spanned) >= 2:
                n_cc_shared += 1
                cc_bytes += t['size']
    m['n_shared_tensors'] = n_shared
    m['shared_tensor_bytes'] = shared_bytes
    m['shared_tensor_bytes_share'] = (
        shared_bytes / m['total_tensor_bytes']) if m['total_tensor_bytes'] else 0.0
    m['n_external_shared_tensors'] = n_ext_shared
    m['external_shared_bytes'] = ext_bytes
    m['n_cross_component_shared_tensors'] = n_cc_shared
    m['cross_component_shared_bytes'] = cc_bytes
    m['max_tensor_fanout'] = max_fanout

    # E. 结构签名重复度（轻量 motif 代理）
    tensor_edges = defaultdict(list)
    for edge in graph.get('edges', []):
        src, dst = edge['source'], edge['target']
        if dst in tensors_by_id:
            tensor_edges[('out', src)].append(tensors_by_id[dst]['size'])
        if src in tensors_by_id:
            tensor_edges[('in', dst)].append(tensors_by_id[src]['size'])
    sigs = Counter(
        structural_signature(i, cpreds, csuccs, ops_by_id, tensor_edges)
        for i in eligible_ids)
    m['sig_top_coverage'] = (
        sigs.most_common(1)[0][1] / n) if n else 0.0
    m['sig_effective_count'] = len(sigs)

    m['graph_class'] = classify(m, rules)
    return m


def summarize(rows):
    n = len(rows)
    classes = Counter(r['graph_class'] for r in rows)

    def med(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return statistics.median(vals) if vals else None

    def pct(key, q):
        vals = sorted(r[key] for r in rows if r[key] is not None)
        if not vals:
            return None
        k = min(len(vals) - 1, max(0, round(q * (len(vals) - 1))))
        return vals[k]

    return {
        'n_cases': n,
        'class_counts': dict(classes),
        'class_shares': {k: v / n for k, v in classes.items()} if n else {},
        'cases_with_cross_component_shared': sum(
            1 for r in rows if r['n_cross_component_shared_tensors'] > 0),
        'cases_with_external_shared': sum(
            1 for r in rows if r['n_external_shared_tensors'] > 0),
        'cases_with_direct_op_edges': sum(
            1 for r in rows if r['n_direct_op_edges'] > 0),
        'total_direct_op_edges': sum(r['n_direct_op_edges'] for r in rows),
        'median': {
            'n_ops_compute': med('n_ops_compute'),
            'n_components': med('n_components'),
            'largest_component_share': med('largest_component_share'),
            'merge_share': med('merge_share'),
            'longest_path_nodes': med('longest_path_nodes'),
            'path_ratio': med('path_ratio'),
            'shared_tensor_bytes_share': med('shared_tensor_bytes_share'),
        },
        'p90': {
            'n_ops_compute': pct('n_ops_compute', 0.9),
            'longest_path_nodes': pct('longest_path_nodes', 0.9),
        },
    }


def write_outputs(rows, summary, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'per_case.csv'
    with csv_path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    json_path = out_dir / 'summary.json'
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 算例结构画像报告', '',
             f"- 用例数：{summary['n_cases']}",
             f"- 类型分布：{json.dumps(summary['class_counts'], ensure_ascii=False)}",
             f"- 类型占比：{ {k: f'{v:.1%}' for k, v in summary['class_shares'].items()} }",
             f"- 存在跨分量共享张量的用例：{summary['cases_with_cross_component_shared']}",
             f"- 存在外部共享输入张量（权重型）的用例：{summary['cases_with_external_shared']}",
             f"- 含直接 op→op 边的用例（X01 证据）：{summary['cases_with_direct_op_edges']}"
             f"（共 {summary['total_direct_op_edges']} 条）", '',
             '| case | 计算op | 分量数 | 最大分量占比 | 最长路径 | 汇聚占比 | 类型 |',
             '|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append(
            f"| {r['case']} | {r['n_ops_compute']} | {r['n_components']} "
            f"| {r['largest_component_share']:.2f} | {r['longest_path_nodes']} "
            f"| {r['merge_share']:.2f} | {r['graph_class']} |")
    (out_dir / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return csv_path, json_path


def make_chain_graph(n_chains, chain_len):
    """合成图：n_chains 条独立链，首节点共同消费一条 COPY_IN 产出的共享张量。"""
    wid = 10_000_000
    ops = [{'id': wid, 'op': 'COPY_IN', 'pipe': 'PIPE_MTE2', 'cycles': 10}]
    tensors = [{'id': wid + 1, 'pos': 'DDR', 'size': 4096}]
    edges = [{'source': wid, 'target': wid + 1}]
    next_id = 1
    for c in range(n_chains):
        prev_tensor = wid + 1
        for k in range(chain_len):
            oid = next_id
            next_id += 1
            ops.append({'id': oid, 'op': f'C{c}_{k}', 'pipe': 'PIPE_V', 'cycles': 5})
            mid = next_id
            next_id += 1
            tensors.append({'id': mid, 'pos': 'UB', 'size': 512})
            edges.append({'source': prev_tensor, 'target': oid})
            edges.append({'source': oid, 'target': mid})
            prev_tensor = mid
    return {'ops': ops, 'tensors': tensors, 'edges': edges}


def make_binary_reduce_tree(depth):
    """合成图：完全二叉汇聚树，2^depth 个源汇到单汇点。"""
    ops, tensors, edges = [], [], []
    next_id = 1
    layer = []
    for i in range(2 ** depth):
        oid = next_id
        next_id += 1
        ops.append({'id': oid, 'op': f'leaf{i}', 'pipe': 'PIPE_M', 'cycles': 3})
        layer.append((oid, None))
    while len(layer) > 1:
        nxt = []
        for i in range(0, len(layer), 2):
            (l, _), (r, _) = layer[i], layer[i + 1]
            oid = next_id
            next_id += 1
            ops.append({'id': oid, 'op': 'reduce', 'pipe': 'PIPE_V', 'cycles': 4})
            mid = next_id
            next_id += 1
            tensors.append({'id': mid, 'pos': 'UB', 'size': 256})
            edges.append({'source': l, 'target': mid})
            edges.append({'source': r, 'target': mid})
            edges.append({'source': mid, 'target': oid})
            nxt.append((oid, None))
        layer = nxt
    return {'ops': ops, 'tensors': tensors, 'edges': edges}


def make_single_chain(n):
    """合成图：单条 n 节点链。"""
    ops, tensors, edges = [], [], []
    src = 900_000
    tensors.append({'id': src, 'pos': 'DDR', 'size': 128})
    prev = src
    for i in range(n):
        oid = 1 + i
        ops.append({'id': oid, 'op': f's{i}', 'pipe': 'PIPE_M', 'cycles': 7})
        mid = 900_100 + i
        tensors.append({'id': mid, 'pos': 'L1', 'size': 128})
        edges.append({'source': prev, 'target': oid})
        edges.append({'source': oid, 'target': mid})
        prev = mid
    return {'ops': ops, 'tensors': tensors, 'edges': edges}


def run_selftest(out_dir):
    tmp = Path(tempfile.mkdtemp(prefix='case_profile_st_'))
    expectations = []
    graphs = {
        'synthetic_multi_comp': make_chain_graph(10, 5),
        'synthetic_reduce_tree': make_binary_reduce_tree(4),
        'synthetic_deep_chain': make_single_chain(20),
    }
    for name, g in graphs.items():
        (tmp / f'{name}.json').write_text(
            json.dumps(g), encoding='utf-8')
    expectations = [
        ('synthetic_multi_comp', 'multi_component'),
        ('synthetic_reduce_tree', 'converge_tree'),
        ('synthetic_deep_chain', 'deep_chain'),
    ]
    ce_dir = Path(__file__).resolve().parents[2] / 'A题_代码包' / 'a_lab' / \
        'knowledge' / 'counterexamples'
    ce_paths = []
    if ce_dir.is_dir():
        for sub in sorted(ce_dir.iterdir()):
            g = sub / 'graph.json'
            if g.is_file():
                ce_paths.append(g)
    rows = []
    for p in sorted(tmp.glob('*.json')) + ce_paths:
        rows.append(profile_case(p, dict(DEFAULT_RULES)))
    failures = []
    for name, expect in expectations:
        got = next(r['graph_class'] for r in rows if r['case'] == name)
        if got != expect:
            failures.append(f'{name}: 期望 {expect}，实际 {got}')
    for name, check in [
        ('synthetic_multi_comp',
         lambda r: r['n_cross_component_shared_tensors'] >= 1),
        ('synthetic_reduce_tree',
         lambda r: r['n_sinks'] == 1 and r['merge_share'] > 0.25),
        ('synthetic_deep_chain',
         lambda r: r['path_ratio'] > 0.9),
    ]:
        row = next(r for r in rows if r['case'] == name)
        if not check(row):
            failures.append(f'{name}: 指标自检未通过 {row}')
    summary = summarize(rows)
    csv_path, json_path = write_outputs(rows, summary, out_dir)
    return failures, rows, summary, csv_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('input', nargs='?', help='计算图 JSON 或其所在目录')
    parser.add_argument('--out', default='results/case_structure',
                        help='输出目录（默认 results/case_structure）')
    parser.add_argument('--rules', help='分类阈值 JSON 文件，缺省用内置默认')
    parser.add_argument('--selftest', action='store_true',
                        help='合成图 + a_lab 反例图自测')
    args = parser.parse_args(argv)

    rules = dict(DEFAULT_RULES)
    if args.rules:
        rules.update(json.loads(Path(args.rules).read_text(encoding='utf-8')))

    if args.selftest:
        default = 'results/case_structure_selftest'
        out = Path(default if args.out == 'results/case_structure' else args.out)
        failures, rows, summary, csv_path = run_selftest(out)
        print(f'自测用例 {len(rows)} 个，输出 {csv_path}')
        if failures:
            print('自测失败：', *failures, sep='\n  ')
            return 1
        print('自测通过：3 个合成图分类与指标断言全部符合预期；'
              'a_lab 反例图解析无异常。')
        return 0

    if not args.input:
        parser.error('需要输入路径，或使用 --selftest')
    in_path = Path(args.input)
    paths = sorted(in_path.glob('*.json')) if in_path.is_dir() else [in_path]
    rows = []
    errors = []
    for p in paths:
        try:
            rows.append(profile_case(p, rules))
        except (ProfileError, KeyError, ValueError) as exc:
            errors.append(f'{p.name}: {exc}')
    if not rows:
        print('没有成功统计的用例', *errors, sep='\n  ')
        return 1
    summary = summarize(rows)
    csv_path, _ = write_outputs(rows, summary, Path(args.out))
    print(f'统计 {len(rows)} 个用例，输出 {csv_path}')
    if errors:
        print(f'{len(errors)} 个失败：', *errors, sep='\n  ')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not errors else 2


if __name__ == '__main__':
    sys.exit(main())
