import numpy as np, itertools
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
def solve(nv,edges,m,obj2=False):
    L=m+1; N=L**nv
    idx=np.array(list(itertools.product(range(L),repeat=nv)))
    A=lil_matrix((len(edges)*(2*m+1),N)); b=[]
    r=0
    for (u,v) in edges:
        d=idx[:,v]-idx[:,u]
        for i in range(-m,m+1):
            A[r,np.nonzero(d<=i)[0]]=1; b.append(0.5*(1+i/m)); r+=1
    # fix alpha=1/2, find sparse vertex solution (min random cost to get vertex)
    rng=np.random.default_rng(1)
    c=rng.uniform(0,1,N)
    res=linprog(c,A_ub=A.tocsr(),b_ub=b,A_eq=np.ones((1,N)),b_eq=[1],bounds=(0,None),method='highs-ds')
    x=res.x; s=np.nonzero(x>1e-9)[0]
    return [(tuple(idx[k]),round(x[k],4)) for k in s]
for nv,E,m in [(3,[(0,1),(1,2),(0,2)],4),(4,[(i,j) for i in range(4) for j in range(i+1,4)],4)]:
    sol=solve(nv,E,m); print(nv,"vertices, m=",m,"support",len(sol))
    for t,p in sorted(sol): print("  labels*m",t,"prob",p)
