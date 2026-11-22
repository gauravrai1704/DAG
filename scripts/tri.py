# Best correlated label distribution (grid {0,1/m,..,1}) for a graph, within KKR framework:
# minimise alpha s.t. for every edge (u,v), every i in -m..m: P[l(v)-l(u) <= i/m] <= alpha*(1+i/m)
import numpy as np, itertools, sys
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
def solve(nv,edges,m):
    L=m+1; N=L**nv
    idx=np.array(list(itertools.product(range(L),repeat=nv)))
    rows=[];b=[]
    A=lil_matrix((len(edges)*(2*m+1),N+1))
    r=0
    for (u,v) in edges:
        d=idx[:,v]-idx[:,u]
        for i in range(-m,m+1):
            cols=np.nonzero(d<=i)[0]
            A[r,cols]=1; A[r,N]=-(1+i/m); r+=1
    Aeq=np.ones((1,N+1));Aeq[0,N]=0
    c=np.zeros(N+1);c[N]=1
    res=linprog(c,A_ub=A.tocsr(),b_ub=np.zeros(r),A_eq=Aeq,b_eq=[1],bounds=[(0,None)]*(N+1),method='highs')
    return res.x[N]
if __name__=="__main__":
  for m in [int(a) for a in sys.argv[1:]]:
      print("m=%d  single edge: %.5f  path a->b->c: %.5f  transitive triangle: %.5f"%(m,solve(2,[(0,1)],m),solve(3,[(0,1),(1,2)],m),solve(3,[(0,1),(1,2),(0,2)],m)),flush=True)
