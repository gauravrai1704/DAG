import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, minimize
# 1. Uniform labels: F(t)/(t+1)
F_u=lambda t: (0.5*t*t+t+0.5) if t<0 else (-0.5*t*t+t+0.5)
r=minimize_scalar(lambda t:-F_u(t)/(t+1),bounds=(0,1),method='bounded')
print("uniform: max ratio %.10f at t=%.6f ; 2-sqrt2=%.10f"%(-r.fun,r.x,2-np.sqrt(2)))
# 2. KKR density
p=lambda x: 2/3+(23/3)*(1-2*x)**22
P=lambda x: 0 if x<=0 else (1 if x>=1 else 2*x/3+(1-(1-2*x)**23)/6)
print("mass", quad(p,0,1)[0])
def F(t): # P[X-Y<=t]
    lo=max(0,-t); hi=min(1,1-t)
    a=quad(lambda y:p(y)*P(y+t),lo,hi,limit=200,epsabs=1e-13)[0] if hi>lo else 0
    b=quad(p,max(0,1-t),1)[0] if t>0 else 0  # y+t>=1 -> P=1
    return a+b
ts=np.linspace(-0.99,1,2000); rs=[F(t)/(t+1) for t in ts]; i=int(np.argmax(rs))
r=minimize_scalar(lambda t:-F(t)/(t+1),bounds=(ts[max(i-2,0)],ts[min(i+2,len(ts)-1)]),method='bounded',options={'xatol':1e-10})
print("KKR density: max ratio %.8f at t=%.5f"%(-r.fun,r.x))
# 3. Fourier lower bound, single cosine
f=lambda t: 0.5+(-np.sin(t))/(6*t+2*np.sin(t))
r=minimize_scalar(lambda t:-f(t),bounds=(np.pi,2*np.pi),method='bounded')
print("1-cos bound: at t=4.5: %.6f ; best t=%.5f gives %.6f"%(f(4.5),r.x,-r.fun))
# 4. general multi-cosine bound
xs=np.linspace(0,1,20001)
def bound(a,t):
    a=np.asarray(a);t=np.asarray(t)
    num=-np.sum(a*np.sin(t)/t)
    h=(a[:,None]*t[:,None]*np.sin(t[:,None]*xs[None,:])).sum(0)
    den=2*np.trapezoid(np.where(h>0,h*(1+xs),-h*(1-xs)),xs)
    return 0.5+num/den
print("check single t=4.5 general:",bound([1],[4.5]))
print("paper 2-cos (4.4, 8.9; 1,0.29):",bound([1,0.29],[4.4,8.9]))
best=None
rng=np.random.default_rng(0)
for K in [1,2,3,4,5,6]:
    bk=0
    for s in range(60):
        x0=np.concatenate([rng.uniform(0,1,K-1),np.sort(rng.uniform(3,4*np.pi*K/2+3,K))])
        if K>=2: x0[K-1]=4.4; x0[K]=8.9 if s%2==0 else x0[K]
        obj=lambda v: -bound(np.concatenate([[1],np.abs(v[:K-1])]),v[K-1:]) if np.all(v[K-1:]>0.1) else 0
        res=minimize(obj,x0,method='Nelder-Mead',options={'maxiter':4000,'xatol':1e-7,'fatol':1e-10})
        if -res.fun>bk: bk=-res.fun; bv=res.x
    print("K=%d cosines: bound %.6f  weights %s freqs %s"%(K,bk,np.round(np.concatenate([[1],np.abs(bv[:K-1])]),4),np.round(bv[K-1:],4)))
