# rigorous-style upper bound: F nondecreasing, so on [t0,t1]: F(t)/(1+t) <= F(t1)/(1+t0)
import numpy as np, sys
from fractions import Fraction as Fr
p=np.load(sys.argv[1]); N=len(p); h=1/N
q=[Fr(float(v)).limit_denominator(10**12) for v in p]; s=sum(q); q=[v/s for v in q]   # exact rational density
pf=np.array([float(v) for v in q])
c=np.correlate(pf,pf,'full'); D=np.arange(-(N-1),N)
T=lambda u: np.where(u<-1,0,np.where(u<0,0.5*(1+u)**2,np.where(u<1,1-0.5*(1-u)**2,1)))
ts=np.linspace(-1,1,4_000_001)
F=np.concatenate([T((ts[i:i+200000,None]-D[None,:]*h)/h)@c for i in range(0,len(ts),200000)])
ub=(F[1:]/(1+ts[:-1]))[1:].max()     # skip the first cell, where 1+t0=0
# first cell [-1,-1+dt]: F(t) <= p0*p_{N-1}*(1+t)^2/(2h^2) -> ratio <= that/(1+t) -> tiny
dt=ts[1]-ts[0]; first=float(q[0]*q[-1])*dt/(2*h*h)
print("bin probs exact-rational; certified sup ratio <= %.6f (first-cell bound %.2e)"%(max(ub,first),first))
