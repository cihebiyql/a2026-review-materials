import pymetis, warnings, sys
warnings.filterwarnings('ignore')
sys.path.insert(0, r'C:/shumo_live/02_求解/A题_2026/a_lab/solvers/exp_strand')
from collections import defaultdict, Counter
from common import load_case, op_dag, cycles_map, tensor_views
g = load_case('case_005')
ids, preds, succs = op_dag(g)
w = cycles_map(g)
prod, cons, tsize, _ = tensor_views(g)
ew = defaultdict(float)
for o, ts in cons.items():
    for t in ts:
        for p in prod.get(t, ()):
            if p != o: ew[(p,o)] += tsize.get(t,0)
idx = {v:i for i,v in enumerate(ids)}; n=len(ids)
LAT=15000
adj=[[] for _ in range(n)]
for (a,b),by in ew.items():
    if a in idx and b in idx:
        adj[idx[a]].append((idx[b], int(by+LAT)))
for v in ids:
    for s in succs.get(v,()):
        if (v,s) not in ew:
            adj[idx[v]].append((idx[s], LAT))
for i in range(n):
    seen={}
    for j,wg in adj[i]: seen[j]=max(seen.get(j,0),wg)
    adj[i]=[(j,wg) for j,wg in sorted(seen.items())]
xadj=[0]; adjncy=[]; adjwgt=[]
for lst in adj:
    for j,wg in lst: adjncy.append(j); adjwgt.append(wg)
    xadj.append(xadj[-1]+len(lst))
vw=[int(w[v]) for v in ids]
nc, mem = pymetis.part_graph(5, xadj=xadj, adjncy=adjncy, eweights=adjwgt, vweights=vw)
lw=Counter()
for v in ids: lw[mem[idx[v]]] += w[v]
print('CSR+weights: ncuts=',nc,' parts=', dict(sorted(lw.items())))
