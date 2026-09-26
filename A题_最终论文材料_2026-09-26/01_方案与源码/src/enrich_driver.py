# 纯 python 的 enrich 驱动: 生成目标列表 -> subprocess 逐个跑(ENRICH_ONE) -> 崩溃记 CRASHED
import json, os, subprocess

N5 = r'C:/shumo_live/02_求解/A题_2026/n5_push'
OUTJ = N5 + '/baseline_full_metrics.jsonl'

targets = set()
for fn, algos in (('baseline_results.jsonl', ('topo_equal', 'greedy_balance', 'heft_like', 'kl_partition')),
                  ('baseline_hinted.jsonl', ('shared_input', 'pipe_balance', 'merge_greedy')),
                  ('baseline_extra.jsonl', ('random_search', 'core_resident'))):
    fp = os.path.join(N5, fn)
    if not os.path.exists(fp):
        continue
    for l in open(fp, encoding='utf-8'):
        if not l.strip().startswith('{'):
            continue
        try:
            r = json.loads(l)
        except Exception:
            continue
        if 'sp' in r and r.get('algo') in algos:
            targets.add((r['case'], int(r['q']), r['algo']))

def load_done():
    done = set()
    if os.path.exists(OUTJ):
        for l in open(OUTJ, encoding='utf-8'):
            try:
                d = json.loads(l)
                if 'mk' in d:
                    done.add(d['key'])
            except Exception:
                pass
    return done


done = load_done()

print(f'targets {len(targets)}, done {len(done)}', flush=True)
env = dict(os.environ)
for case, q, algo in sorted(targets):
    key = f'{case}|{q}|{algo}'
    if key in load_done():
        continue
    env['ENRICH_ONE'] = key
    r = subprocess.run(['py', '-3.11', os.path.join(N5, 'enrich_all.py')],
                       capture_output=True, text=True, env=env, timeout=900,
                       cwd=N5)
    ok = False
    if os.path.exists(OUTJ):
        for l in open(OUTJ, encoding='utf-8').readlines()[-5:]:
            try:
                d = json.loads(l)
            except Exception:
                continue
            if d.get('key') == key and 'mk' in d:
                ok = True
    if not ok:
        with open(OUTJ, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'key': key, 'status': 'CRASHED'}) + '\n')
        print(key, 'CRASHED', flush=True)
    else:
        print(key, 'ok', flush=True)
print('DRIVER_DONE', flush=True)
