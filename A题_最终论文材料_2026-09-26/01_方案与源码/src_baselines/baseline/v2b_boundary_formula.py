"""V2 猜想B：切分边界搬运公式受控实验。

构造受控小图（单生产者 op A -> tensor t(size S) -> m 个消费者 op），
用不同切法对照官方评估器的 partition_added_copy_bytes：

  公式（场景A）：额外搬运 = S x (1 + #消费Task数)     [COPY_OUT 1 次 + 每消费Task COPY_IN 1 次]
  公式（场景B）：仅跨核消费计 COPY 对，同核免费（B_task 与 A 同核则免）

输出: results/v2b_boundary_formula.json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / 'solver'))
import eval_lib as EL  # noqa: E402


def build_graph(m_consumers, S, cycles=50):
    """A -(t)-> B1..Bm，B 各自经原生 COPY_OUT 输出到 DDR（贴近真实图：
    图输出自带 COPY_OUT，未切分时新增搬运应为 0）。"""
    ops, tensors, edges = [], [], []
    ops.append({'id': 1, 'op': 'MATMUL', 'pipe': 'PIPE_M', 'cycles': cycles})
    tensors.append({'id': 101, 'pos': 'UB', 'size': S})
    edges.append({'source': 1, 'target': 101})
    for j in range(m_consumers):
        b = 2 + j
        co = 200 + 2 * j          # 原生 COPY_OUT op
        ddr = 201 + 2 * j         # DDR 输出 tensor
        ops.append({'id': b, 'op': 'VECTOR', 'pipe': 'PIPE_V', 'cycles': cycles})
        ops.append({'id': co, 'op': 'COPY_OUT', 'pipe': 'PIPE_MTE3',
                    'cycles': max(1, S // 60)})
        tensors += [{'id': 102 + j, 'pos': 'UB', 'size': S},
                    {'id': ddr, 'pos': 'DDR', 'size': S}]
        edges += [{'source': 101, 'target': b},
                  {'source': b, 'target': 102 + j},
                  {'source': 102 + j, 'target': co},
                  {'source': co, 'target': ddr}]
    return {'ops': ops, 'tensors': tensors, 'edges': edges}


def plan_of(groups, core_of_group=None):
    """groups: 子图成员列表；默认子图 i 放核心 i（core_of_group 可覆盖）。"""
    n2s = {}
    for i, g in enumerate(groups):
        for v in g:
            n2s[v] = i
    core_of_group = core_of_group or {i: i for i in range(len(groups))}
    ncores = max(core_of_group.values()) + 1
    sched = [[] for _ in range(ncores)]
    for i in range(len(groups)):
        sched[core_of_group[i]].append(i)
    return {'node_to_subgraph': n2s, 'core_schedules': sched}


def main():
    out = []
    S = 12000
    for m in (1, 2, 4):
        g = build_graph(m, S)
        B = list(range(2, 2 + m))
        cases = {
            'no_cut': (plan_of([{1} | set(B)]), 0, 0),
            'A_vs_allB': (plan_of([{1}, set(B)]), S * 2, S * 2),
            'A_eachB_alone': (plan_of([{1}] + [{b} for b in B]), S * (1 + m),
                              S * 2 * m),
        }
        if m == 2:
            cases['A_vs_B1B2_samecore_with_A'] = (
                plan_of([{1}, set(B)], core_of_group={0: 0, 1: 0}), S * 2, 0)
        if m == 4:
            cases['A_B12_B34'] = (plan_of([{1}, {2, 3}, {4, 5}]), S * 3, S * 2 * 2)
            cases['A_B1B2core0_B3B4core1'] = (
                plan_of([{1}, {2, 3}, {4, 5}], core_of_group={0: 0, 1: 0, 2: 1}),
                S * 3, S * 2 * 1)
        for name, (plan, formula_A, formula_B) in cases.items():
            row = {'m': m, 'S': S, 'cut': name,
                   'formula_A': formula_A, 'formula_B': formula_B}
            for problem in (1, 2):
                result, _ = EL.evaluate(g, plan, problem)
                dm = result['data_movement_bytes']
                row['p%d_partition' % problem] = dm['partition_added_copy_bytes']
                row['p%d_spill' % problem] = dm['spill_added_copy_bytes']
                row['p%d_cross_task' % problem] = result.get('cross_task_traffic', 0)
            row['match_A'] = (row['p1_partition'] == formula_A)
            row['match_B'] = (row['p2_partition'] == formula_B)
            out.append(row)
            print(row, flush=True)
    dest = HERE / 'results' / 'v2b_boundary_formula.json'
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print('saved ->', dest)


if __name__ == '__main__':
    main()
