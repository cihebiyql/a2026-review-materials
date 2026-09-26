import sys, json, os
N5 = os.environ['A2026_ROOT'] + '/n5_push'
sys.path.insert(0, 'n5_push')
sys.argv = ['mechA_v4.py', '2']
import runpy
m = runpy.run_path('n5_push/mechA_v4.py', run_name='not_main')
from common import load_case, ev_p2
from fast_eval_p2 import FastEvalP2
from mechA_v4 import op_pipe_of, sweep
shard, nshard = int(sys.argv[0]) if False else int(os.environ['SHARD']), 4
done = set()
if os.path.exists('n5_push/mechA_v4_q2.jsonl'):
    for line in open('n5_push/mechA_v4_q2.jsonl'):
        try: done.add(json.loads(line)['case'])
        except Exception: pass
cases = [c for c in (f'case_{i:03d}' for i in range(1, 101) if i % 4 == shard and i not in (14, 40)) if c not in done]
for case in cases:
    try:
        graph = load_case(case)
        pipe_of = op_pipe_of(graph)
        fe = FastEvalP2(graph)
        best, sc = sweep(case, fe, ev_p2, graph, pipe_of)
        if best is None:
            rec = {'case': case, 'status': 'no_seed'}
        else:
            mk_best, pl_best, P = best
            rec = {'case': case, 'mk_best': mk_best, 'P': P,
                   'sp_best': (sc / mk_best) if pl_best else None,
                   'saved': pl_best is not None}
            if pl_best is not None:
                import os as _os
                mk_off = ev_p2(graph, pl_best)[0]['makespan']
                rec['official_match'] = (mk_off == mk_best)
                if mk_off == mk_best:
                    json.dump({'plan': pl_best, 'mk': mk_off, 'sp': sc / mk_off,
                               'source': f'mechA_v4_shard {P}'},
                              open(f'{N5}/superlinear_analysis/{case}_q2_mechA.json', 'w'))
    except Exception as e:
        rec = {'case': case, 'status': 'ERR:' + str(e)[:50]}
    with open('n5_push/mechA_v4_q2.jsonl', 'a') as f:
        f.write(json.dumps(rec, default=str) + '\n')
print('V4SHARD_DONE', flush=True)
