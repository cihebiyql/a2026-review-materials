"""V2 猜想C：模板性 / motif 重复度（块指派降维构想的数据基础）。

对收缩 COPY 后的 eligible op-DAG 做 WL 风格哈希：
  label_0(v)  = hash(op类型, pipe, cycles, 入/出 tensor 尺寸多重集)
  label_k(v)  = hash(label_{k-1}(v), 多重集(label_{k-1}(u), 边张量字节数))
统计最终标签的重复度分布。case_001 若确为 200 条同构链，应有海量高重复 motif。

输出: results/v2c_motif_hash.json
"""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / 'solver'))
import eval_lib as EL  # noqa: E402
from naive_convex import eligible_ids  # noqa: E402
from stub_multicore_cut_and_schedule import (  # noqa: E402
    _build_op_adjacency, _contract_excluded_copy_nodes)


def wl_hashes(graph, rounds=2):
    ids = eligible_ids(graph)
    op_by_id = {op['id']: op for op in graph['ops']}
    tensor_by_id = {t['id']: t for t in graph['tensors']}
    op_ids = set(op_by_id)
    # tensor -> (producers, consumers)
    prod, cons = defaultdict(set), defaultdict(set)
    direct = defaultdict(list)
    for e in graph['edges']:
        s, d = e['source'], e['target']
        if s in op_ids and d not in op_ids:
            prod[d].add(s)
        elif s not in op_ids and d in op_ids:
            cons[s].add(d)
        elif s in op_ids and d in op_ids and s != d:
            direct[s].append(d)
    _, succs = _build_op_adjacency(graph)
    cpreds, csuccs = _contract_excluded_copy_nodes(ids, succs)

    def h(obj):
        return hashlib.md5(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]

    def tensor_sig(tid):
        t = tensor_by_id.get(tid)
        return [t['pos'], t['size']] if t else None

    # 每 op 的入/出 tensor 预索引（避免 O(V*T) 扫描）
    in_tids = defaultdict(list)
    out_tids = defaultdict(list)
    for tid, cs in cons.items():
        for c in cs:
            in_tids[c].append(tid)
    for tid, ps in prod.items():
        for p in ps:
            out_tids[p].append(tid)

    label = {}
    for v in ids:
        op = op_by_id[v]
        ins = sorted(tensor_sig(t) for t in in_tids[v])
        outs = sorted(tensor_sig(t) for t in out_tids[v])
        label[v] = h([op.get('op'), op.get('pipe'), op.get('cycles'),
                      ins, outs, len(direct[v])])
    for _ in range(rounds):
        new = {}
        for v in ids:
            nbr = sorted([label[u], min(
                (tensor_by_id[t]['size'] for t in in_tids[u]), default=0)]
                for u in cpreds[v])
            nbr2 = sorted([label[u], min(
                (tensor_by_id[t]['size'] for t in out_tids[u]), default=0)]
                for u in csuccs[v])
            new[v] = h([label[v], nbr, nbr2])
        label = new
    return label, ids


def main():
    out = {}
    for case in ['case_001.json', 'case_050.json', 'case_014.json']:
        graph = EL.load_graph(case)
        label, ids = wl_hashes(graph, rounds=2)
        cnt = Counter(label[v] for v in ids)
        mult = Counter(cnt.values())  # 重复度 -> 出现次数
        n = len(ids)
        ops_in_big = sum(c * m for m, c in mult.items() if m >= 10)
        top = cnt.most_common(8)
        out[case] = {
            'n_eligible_ops': n,
            'n_distinct_motifs': len(cnt),
            'compression_ratio': round(n / len(cnt), 2),
            'ops_in_motifs_ge10': ops_in_big,
            'frac_in_motifs_ge10': round(ops_in_big / n, 3),
            'multiplicity_hist': {str(k): v for k, v in sorted(mult.items())},
            'top8': top,
        }
        print(case, out[case], flush=True)
    dest = HERE / 'results' / 'v2c_motif_hash.json'
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print('saved ->', dest)


if __name__ == '__main__':
    main()
