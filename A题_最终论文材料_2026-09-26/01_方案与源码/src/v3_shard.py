import sys, json, os
sys.path.insert(0, 'n5_push')
sys.argv = ['mechA_v4.py', '2']
import runpy
m = runpy.run_path('n5_push/mechA_v3.py', run_name='not_main')
pass
pass
from mechA_v3 import one_case, DONE
shard, nshard = int(sys.argv[0]) if False else int(os.environ['SHARD']), 4
cases = [f'case_{i:03d}' for i in range(1, 101) if i % 4 == shard and i not in (14, 40)]
for case in cases:
    if case in DONE:
        continue
    try:
        rec = one_case(case)
    except Exception as e:
        rec = {'case': case, 'status': 'ERR:' + str(e)[:50]}
    with open('n5_push/mechA_v3shard_q3.jsonl', 'a') as f:
        f.write(json.dumps(rec, default=str) + '\n')
print('V4SHARD_DONE', flush=True)
