import json,sys,copy,itertools
from pathlib import Path
sys.path.insert(0,r'C:\shumo_live\02_求解\A题_2026\fast_eval'); from fast_eval_p1 import FastEvalP1
base=Path(r'C:\shumo_live\02_求解\A题_2026\a_lab\runs\SMOKE-BAND2-N5-20260924'); data=Path(r'C:\shumo_live\a_data\data')
for case in ['case_016','case_024','case_051']:
 p=json.loads((base/f'{case}_q1_N5_s11'/'plan.json').read_text()); fe=FastEvalP1(json.loads((data/f'{case}.json').read_text())); b,_=fe.evaluate(p); best=(b,'base'); by={}
 for c,l in enumerate(p['core_schedules']):
  for sg in l: by[sg]=c
 sgs=list(by)
 for a,bb in itertools.combinations(sgs[:80],2):
  ca,cb=by[a],by[bb]
  if ca==cb: continue
  q=copy.deepcopy(p); q['core_schedules'][ca].remove(a); q['core_schedules'][cb].remove(bb); q['core_schedules'][ca].append(bb); q['core_schedules'][cb].append(a)
  try:
   m,_=fe.evaluate(q)
   if m<best[0]: best=(m,f'swap_{a}_{bb}')
  except: pass
 print(case,best[0],best[1], 'gain',b/best[0])
