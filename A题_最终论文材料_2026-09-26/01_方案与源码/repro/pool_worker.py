# pool_worker.py — 冠军池收割重评子进程: 一个(case,Q)组的全部方案评估后打印JSON退出
# 本程序及代码是在人工智能工具辅助下完成的。
# 人工智能工具名称：GLM，版本/型号：zai-api/GLM-5.3，开发机构/公司：智谱AI(Z.ai)，版本颁布日期：2026-09-26
#
# stdin: {"case":..,"Q":..,"plans":[plan,...],"official":bool(缺省false)}
# stdout: [{"mk":..,"info":{"added_copy_bytes":..,"cache_stats":{..}},"eval":"fast"|"official"},...]
# 两种模式:
#   fast(缺省) — FastEvalP1/P2/P3(numba 复刻,与合规收割同路径)
#   official=true — common.ev_p1/ev_p2/ev_p3(官方评估器,纯Python,用于崩溃类图兜底;
#                   返回的 data_movement_bytes.added_copy_bytes 与 cache_stats 口径与 fast 对齐)
# 原生崩溃(崩溃类图,如 case_014/040 大图堆破坏)由父进程按子进程非零退出后
#   以 official=true 重试兜底。
import sys, os, json

ROOT = os.environ.get('A2026_ROOT', r'C:/shumo_live/02_求解/A题_2026')
sys.path.insert(0, ROOT + '/fast_eval')
sys.path.insert(0, ROOT + '/v3_solver')
from common import load_case, ev_p1, ev_p2, ev_p3          # noqa: E402
from fast_eval_p2 import FastEvalP2, FastEvalP3            # noqa: E402
from fast_eval_p1 import FastEvalP1                        # noqa: E402


def _norm_cache(cs):
    """统一 cache_stats 字段: 补 hit_cnt_rate(次数口径), hit_rate(字节口径)。"""
    if not cs:
        return None
    hits, acc = int(cs.get('hits', 0)), int(cs.get('accesses', 0))
    hb, mb = int(cs.get('hit_bytes', 0)), int(cs.get('miss_bytes', 0))
    out = {'hits': hits, 'accesses': acc, 'hit_bytes': hb, 'miss_bytes': mb,
           'hit_cnt_rate': (hits / acc) if acc else 0.0,
           'hit_rate': (hb / (hb + mb)) if (hb + mb) else 0.0}
    return out


def eval_official(Q, g, plan):
    r, _wt = {1: ev_p1, 2: ev_p2, 3: ev_p3}[Q](g, plan)
    dm = r.get('data_movement_bytes', {}) or {}
    info = {k: int(dm[k]) for k in ('added_copy_bytes', 'spill_added_copy_bytes',
                                    'partition_added_copy_bytes') if k in dm}
    if Q == 3:
        cs = _norm_cache(r.get('cache_stats'))
        if cs:
            info['cache_stats'] = cs
    return int(round(r['makespan'])), info


def eval_fast(fe, plan):
    mk, info = fe.evaluate(plan)
    out_info = {k: info[k] for k in ('added_copy_bytes', 'spill_added_copy_bytes',
                                     'partition_added_copy_bytes') if k in info}
    if 'cache_stats' in info:
        cs = _norm_cache(info['cache_stats'])
        if cs:
            out_info['cache_stats'] = cs
    return int(mk), out_info


def main():
    req = json.loads(sys.stdin.read())
    case, Q, plans = req['case'], req['Q'], req['plans']
    official = bool(req.get('official'))
    g = load_case(case)
    out = []
    if official:
        for p in plans:
            mk, info = eval_official(Q, g, p)
            out.append({'mk': mk, 'info': info, 'eval': 'official'})
    else:
        fe = {1: FastEvalP1, 2: FastEvalP2, 3: FastEvalP3}[Q](g)
        for p in plans:
            mk, info = eval_fast(fe, p)
            out.append({'mk': mk, 'info': info, 'eval': 'fast'})
    print(json.dumps(out))


if __name__ == '__main__':
    main()
