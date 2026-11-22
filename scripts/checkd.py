import numpy as np, sys
from fractions import Fraction as Fr
p=np.load(sys.argv[1]); N=len(p); h=1/N
# exact: Z=X-Y with X~step density; CDF piecewise quadratic in t with breakpoints at multiples of h.
c=np.correlate(p,p,'full'); D=np.arange(-(N-1),N)
T=lambda u: np.where(u<-1,0,np.where(u<0,0.5*(1+u)**2,np.where(u<1,1-0.5*(1-u)**2,1)))
ts=np.linspace(-1,1,2_000_001)[1:]
best=0;arg=None
for s in range(0,len(ts),200000):
    tt=ts[s:s+200000]; F=T((tt[:,None]-D[None,:]*h)/h)@c; r=F/(1+tt); i=r.argmax()
    if r[i]>best: best=r[i];arg=tt[i]
print("fine-grid (2e6 pts) sup ratio %.6f at t=%.5f"%(best,arg))
# Monte Carlo independent check
rng=np.random.default_rng(7); M=20_000_000
b1=rng.choice(N,M,p=p); b2=rng.choice(N,M,p=p)
Z=(b1+rng.uniform(0,1,M))*h-(b2+rng.uniform(0,1,M))*h
for t in [arg-0.05,arg,arg+0.05,0.0,0.5]:
    print("  MC t=%.3f  P[Z<=t]/(1+t)=%.5f"%(t,(Z<=t).mean()/(1+t)))
