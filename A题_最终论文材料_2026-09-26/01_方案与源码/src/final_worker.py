# final_worker.py — 终版总表收割子进程: 逐(case,Q)组评估, 提取全字段+官方评估计时
# 本程序及代码是在人工智能工具辅助下完成的。
# 人工智能工具名称：GLM，版本/型号：zai-api/GLM-5.3，开发机构/公司：智谱AI(Z.ai)，版本颁布日期：2026-09-26
# stdin: {"case":..,"Q":..,"plans":[plan,...],"time_official":bool}
# stdout: [{"mk":..,"added":..,"partition":..,"spill":..,"cache_hit_rate":..,
#           "mk_nol2":..,"added_nol2":..,"official_seconds":..}, ...]
# Q=3 行自动附 P2(无L2)对照评估。原生崩溃由父进程以 official 模式重试兜底。
import sys, os, json, time

ROOT = os.environ.get('A2026_ROOT', r'C:/shumo_live/02_求解/A题_2026')
sys.path.insert(0, ROOT + '/fast_eval')
sys.path.insert(0, ROOT + '/v3_solver')
from common import load_case, ev_p1, ev_p2, ev_p3    # noqa: E402
from fast_eval_p2 import FastEvalP2, FastEvalP3      # noqa: E402
from fast_eval_p1 import FastEvalP1                  # noqa: E402


def info_fields(info):
    cs = info.get('cache_stats', {}) or {}
    hit = cs.get('hit_rate', cs.get('hit_cnt_rate'))
    if hit is None and cs:
        acc = cs.get('accesses') or cs.get('total')
        hits = cs.get('hits')
        if acc:
            hit = hits / acc
    return (info.get('added_copy_bytes'), info.get('partition_added_copy_bytes'),
            info.get('spill_added_copy_bytes'), hit)


def main():
    req = json.loads(sys.stdin.read())
    case, Q, plans = req['case'], req['Q'], req['plans']
    official = bool(req.get('time_official'))
    g = load_case(case)
    out = []
    if official:
        ev = (ev_p1, ev_p2, ev_p3)[Q - 1]
        ev2 = ev_p2
        for p in plans:
            t0 = time.time()
            mk = ev(g, p)[0]['makespan']
            dt = time.time() - t0
            row = {'mk': int(mk), 'official_seconds': round(dt, 2),
                   'added': None, 'partition': None, 'spill': None,
                   'cache_hit_rate': None, 'mk_nol2': None, 'added_nol2': None}
            # 官方口径不带 info 分解, 崩溃兜底行至少给出 mk/计时
            out.append(row)
    else:
        fe = {1: FastEvalP1, 2: FastEvalP2, 3: FastEvalP3}[Q](g)
        fe2 = FastEvalP2(g)
        for p in plans:
            mk, info = fe.evaluate(p)
            a, pt, sp, hit = info_fields(info)
            row = {'mk': int(mk), 'added': a, 'partition': pt, 'spill': sp,
                   'cache_hit_rate': hit, 'official_seconds': None,
                   'mk_nol2': None, 'added_nol2': None}
            if Q == 3:
                mk2, info2 = fe2.evaluate(p)
                row['mk_nol2'] = int(mk2)
                row['added_nol2'] = info2.get('added_copy_bytes')
            out.append(row)
    print(json.dumps(out))


if __name__ == '__main__':
    main()
