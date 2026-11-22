import numpy as np, itertools, sys
from fractions import Fraction as Fr
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
def build(w,m):
    L=m+1; tups=list(itertools.combinations_with_replacement(range(L),w))
    # for each tuple: histogram of ordered-pair differences d in [-m,m]
    H=np.zeros((len(tups),2*m+1))
    for k,t in enumerate(tups):
        t=np.array(t); d=(t[None,:]-t[:,None])[~np.eye(w,dtype=bool)]
        np.add.at(H[k],d+m,1)
    H/=w*(w-1)
    C=np.cumsum(H,1)  # P[Z<=i]
    return tups,H,C
def solve(w,m):
    tups,H,C=build(w,m); N=len(tups)
    A=np.hstack([C.T,-(1+np.arange(-m,m+1)/m)[:,None]])
    c=np.zeros(N+1);c[-1]=1
    Aeq=np.ones((1,N+1));Aeq[0,-1]=0
    r=linprog(c,A_ub=csr_matrix(A),b_ub=np.zeros(2*m+1),A_eq=Aeq,b_eq=[1],bounds=(0,None),method='highs')
    return tups,H,r.x[:-1],r.x[-1]
def exact_check(w,m,tups,H,x):
    # rationalise, renormalise, compute exact max_i F(i/m)/(1+i/m)
    q=[Fr(v).limit_denominator(10**9) if v>1e-12 else Fr(0) for v in x]; S=sum(q); q=[v/S for v in q]
    Hq=[[Fr(int(round(h*w*(w-1))),w*(w-1)) for h in row] for row in H]
    F=[Fr(0)]*(2*m+1)
    for k,v in enumerate(q):
        if v:
            for i in range(2*m+1): F[i]+=v*Hq[k][i]
    cum=[];s=Fr(0)
    for i in range(2*m+1): s+=F[i]; cum.append(s)
    if cum[0]!=0: return None
    return max(cum[i]/Fr(i,m) for i in range(1,2*m+1))  # (1+(i-m)/m)= i/m
for w,ms in [(int(a.split(':')[0]),[int(b) for b in a.split(':')[1].split(',')]) for a in sys.argv[1:]]:
    for m in ms:
        tups,H,x,al=solve(w,m)
        ex=exact_check(w,m,tups,H,x)
        print("K_%d m=%d: LP alpha=%.6f  m/(2m-1)=%.6f  exact-certified alpha=%s (%.6f)"%(w,m,al,m/(2*m-1),ex,float(ex) if ex else -1),flush=True)
