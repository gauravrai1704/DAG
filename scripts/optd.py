# optimise a piecewise-constant label density on [0,1] (N bins, iid labels) to minimise sup_t P[X-Y<=t]/(1+t)
import numpy as np
from scipy.optimize import minimize
N=int(__import__('sys').argv[1]); h=1/N
T=lambda u: np.where(u<-1,0,np.where(u<0,0.5*(1+u)**2,np.where(u<1,1-0.5*(1-u)**2,1)))
ts=np.linspace(-1,1,1601)[1:]
D=np.arange(-(N-1),N)                       # d=i-j  -> Z = (j-i)h + ... careful sign: Z=X-Y, X in bin i, Y in bin j
K=T((ts[:,None]-D[None,:]*h)/h)            # P[(i-j)h + h(V-W) <= t]
def F(p):
    c=np.correlate(p,p,'full')             # c[d+(N-1)] = sum_i p_i p_{i-d}  -> pairs with i-j=d
    return K@c
def obj(z):
    p=np.exp(z);p/=p.sum()
    r=F(p)/(1+ts)
    return np.log(np.sum(np.exp(400*(r-r.max()))))/400+r.max()
best=(9,None)
rng=np.random.default_rng(0)
x=(np.arange(N)+0.5)/N
starts=[np.log(2/3+(23/3)*(1-2*x)**22), np.zeros(N)]+[rng.normal(0,0.5,N) for _ in range(4)]
for z0 in starts:
    z=z0
    for beta in [1]:
        r=minimize(obj,z,method='L-BFGS-B',options={'maxiter':3000})
        z=r.x
    p=np.exp(z);p/=p.sum(); val=(F(p)/(1+ts)).max()
    if val<best[0]: best=(val,p)
val,p=best
print("N=%d bins: best sup ratio %.5f"%(N,val))
np.save('best_p_%d.npy'%N,p)
print("density (bin heights):",np.round(p*N,3).tolist())
