# run_shard.py — bxcpu SLURM array 分片驱动: 训练(存logits) -> 合规求解(N=2..5)
# 用法: python run_shard.py <shard_id> ; 分片 i 处理 tasks.txt 的第 i::10 行
# 本程序及代码是在人工智能工具辅助下完成的。
# 人工智能工具名称：GLM，版本/型号：zai-api/GLM-5.3，开发机构/公司：智谱AI(Z.ai)，版本颁布日期：2026-09-25
import subprocess, sys, os, json
from concurrent.futures import ThreadPoolExecutor

SHARD = int(sys.argv[1])
NSHARDS = int(os.environ.get('NSHARDS', 10))
PY = os.environ.get('V5PY', sys.executable)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ['A2026_ROOT']
LOGIT_DIR = os.environ.get('A2026_LOGITS', ROOT + '/n5_push/reinforce_v5')
COUT = os.environ.get('A2026_COUT', ROOT + '/n5_push/compliant_out')
STATUS = f'{COUT}/shard_{SHARD}_status.jsonl'
os.makedirs(LOGIT_DIR, exist_ok=True)
os.makedirs(COUT, exist_ok=True)

tasks = [l.split() for l in open(f'{HERE}/tasks.txt') if l.strip()]
mine = [t for i, t in enumerate(tasks) if i % NSHARDS == SHARD]


def log(rec):
    with open(STATUS, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print('SHARDLOG', json.dumps(rec, ensure_ascii=False), flush=True)


def train_one(args):
    case, Q = args
    lg = f'{LOGIT_DIR}/{case}_q{Q}_logits.json'
    if os.path.exists(lg):
        return None
    for extra in ([], ['--isolated', '--pyexe', PY]):
        cmd = [PY, f'{HERE}/reinforce_assign.py', str(Q), case,
               '--iters', '50', '--N', '128', '--workers', '12'] + extra
        try:
            subprocess.run(cmd, cwd=HERE, timeout=5400,
                           stdout=open(f'{COUT}/train_{case}_q{Q}.log', 'w'),
                           stderr=subprocess.STDOUT)
        except Exception:
            pass
        if os.path.exists(lg):
            log({'case': case, 'Q': Q, 'stage': 'train', 'ok': True,
                 'isolated': bool(extra)})
            return True
    log({'case': case, 'Q': Q, 'stage': 'train', 'ok': False})
    return False


def solve_one(args):
    case, Q, N = args
    out = f'{COUT}/{case}_q{Q}_N{N}_compliant.json'
    if os.path.exists(out):
        return None
    cmd = [PY, f'{HERE}/solve_compliant.py', case, str(Q), str(N),
           '--budget_s', '480']
    try:
        subprocess.run(cmd, cwd=HERE, timeout=900,
                       stdout=open(f'{COUT}/solve_{case}_q{Q}_N{N}.log', 'w'),
                       stderr=subprocess.STDOUT)
    except Exception:
        pass
    ok = os.path.exists(out)
    log({'case': case, 'Q': Q, 'N': N, 'stage': 'solve', 'ok': ok})
    return ok


# 阶段1: 训练(每任务 12 worker, 5 并发/节点)
with ThreadPoolExecutor(5) as ex:
    trained = list(ex.map(train_one, mine))

# 阶段2: 合规求解(单核, 48 并发/节点)
solve_tasks = [(c, q, n) for (c, q) in mine
               if os.path.exists(f'{LOGIT_DIR}/{c}_q{q}_logits.json')
               for n in (2, 3, 4, 5)]
with ThreadPoolExecutor(48) as ex:
    list(ex.map(solve_one, solve_tasks))
log({'stage': 'shard_done', 'shard': SHARD, 'n_tasks': len(mine)})
print('SHARD_DONE', SHARD, flush=True)
