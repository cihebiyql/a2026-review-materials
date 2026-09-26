# v5_isolated_eval.py — 一次性子进程评估器(平台补丁)
#   用途: FastEval/numba 层在某些图 x 方案形状组合上堆损坏(045/016/062类,
#   跨 numba 0.60/0.65 复现)。本进程只做一次评估后退出, 崩溃由父进程按
#   无效解(None)回收, 主训练进程永不被污染。
#   协议: stdin = plan JSON; stdout = JSON {"mk": int} 或错误信息; 崩溃=非零退出。
#   用法: python v5_isolated_eval.py <case> [Q]
# 本程序及代码是在人工智能工具辅助下完成的。
# 人工智能工具名称：GLM，版本/型号：zai-api/GLM-5.3，开发机构/公司：智谱AI(Z.ai)，版本颁布日期：2026-09-25
import sys, os, json

CASE = sys.argv[1]
Q = int(sys.argv[2]) if len(sys.argv) > 2 else 3

ROOT = os.environ.get('A2026_ROOT', r'C:/shumo_live/02_求解/A题_2026')
N5 = ROOT + '/n5_push'
sys.path.insert(0, N5 + '/superlinear_analysis')
sys.path.insert(0, ROOT + '/fast_eval')
sys.path.insert(0, ROOT + '/v3_solver')
from common import load_case                      # noqa: E402
from fast_eval_p2 import FastEvalP2, FastEvalP3   # noqa: E402


def main():
    plan = json.loads(sys.stdin.read())
    graph = load_case(CASE)
    fe = FastEvalP2(graph) if Q == 2 else FastEvalP3(graph)
    mk, _ = fe.evaluate(plan)
    print(json.dumps({'mk': int(mk)}))


if __name__ == '__main__':
    main()
