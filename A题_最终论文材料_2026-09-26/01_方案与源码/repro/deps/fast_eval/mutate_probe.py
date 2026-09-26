import json,sys,copy,os
from pathlib import Path
sys.path.insert(0,r'C:\shumo_live\02_求解\A题_2026\fast_eval')
from fast_eval_p1 import FastEvalP1
base=Path(r'C:\shumo_live\02_求解\A题_2026\a_lab\runs\SMOKE-BAND2-N5-20260924')
data=Path(r'C:\shumo_live\a_data\data')
for case in ['case_016','case_024','case_051']:
 plan=json.loads((base/f'{case}_q1_N5_s11'/'plan.json').read_text())
 g=json.loads((data/f'{case}.json').read_text())
 fe=FastEvalP1(g); bmk,bi=fe.evaluate(plan); best=(bmk,plan,'base')
 sgs=sorted({int(v) for v in plan['node_to_subgraph'].values()})
 for sg in sgs[:80]:
  src=next((c for c,lst in enumerate(plan['core_schedules']) if sg in lst),None)
  if src is None: continue
  for dst in range(5):
   if dst==src: continue
   q=copy.deepcopy(plan); q['core_schedules'][src].remove(sg); q['core_schedules'][dst].append(sg)
   try:
    mk,info=fe.evaluate(q)
    if mk<best[0]: best=(mk,q,f'move_{sg}_{src}_{dst}')
   except Exception: pass
 print(case,'base',bmk,'best',best[0],best[2],'gain',bmk/best[0] if best[0] else 0)
