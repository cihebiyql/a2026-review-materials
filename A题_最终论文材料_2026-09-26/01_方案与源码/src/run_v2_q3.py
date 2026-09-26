import runpy, sys
m = runpy.run_path(r'C:/shumo_live/02_求解/A题_2026/n5_push/mechA_v2.py', run_name='not_main')
work = m['work']
cases = [l.strip() for l in open(r'C:/shumo_live/02_求解/A题_2026/n5_push/mechA_all100.txt') if l.strip()]
for c in cases:
    print(work(c, 3), flush=True)
print('SEQ_DONE', flush=True)
