# harvest_worker.py — 收割重评子进程: 一个(case,Q)组的全部方案评估后打印JSON退出
# 本程序及代码是在人工智能工具辅助下完成的。
# 人工智能工具名称：GLM，版本/型号：zai-api/GLM-5.3，开发机构/公司：智谱AI(Z.ai)，版本颁布日期：2026-09-26
# stdin: {"case":..,"Q":..,"plans":[plan,...]} ; stdout: [{"mk":..,"info":{..}},...]
# 原生崩溃(崩溃类图)由父进程按子进程非零退出兜底(info 置 null)。
import sys, os, json

ROOT = os.environ.get('A2026_ROOT', r'C:/shumo_live/02_求解/A题_2026')
sys.path.insert(0, ROOT + '/fast_eval')
sys.path.insert(0, ROOT + '/v3_solver')
from common import load_case                      # noqa: E402
from fast_eval_p2 import FastEvalP2, FastEvalP3   # noqa: E402
from fast_eval_p1 import FastEvalP1               # noqa: E402


def main():
    req = json.loads(sys.stdin.read())
    case, Q, plans = req['case'], req['Q'], req['plans']
    g = load_case(case)
    fe = {1: FastEvalP1, 2: FastEvalP2, 3: FastEvalP3}[Q](g)
    out = []
    for p in plans:
        mk, info = fe.evaluate(p)
        out.append({'mk': int(mk), 'info': {k: info[k] for k in
                    ('added_copy_bytes', 'cache_stats') if k in info}})
    print(json.dumps(out))


if __name__ == '__main__':
    main()
