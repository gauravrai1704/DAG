import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
a=np.array([1,0.2351,0.0682,0.0093]); t=np.array([4.7207,10.0132,16.5361,30.4391])
h=lambda x: float(np.sum(a*t*np.sin(t*x)))
xs=np.linspace(0,1,200001); hv=np.array([h(x) for x in xs[::10]]); xg=xs[::10]
roots=[0.0]+[brentq(h,xg[i],xg[i+1],xtol=1e-15) for i in range(len(xg)-1) if hv[i]*hv[i+1]<0]+[1.0]
den=0
for lo,hi in zip(roots[:-1],roots[1:]):
    mid=(lo+hi)/2
    f=(lambda x:h(x)*(1+x)) if h(mid)>0 else (lambda x:-h(x)*(1-x))
    den+=quad(f,lo,hi,epsabs=1e-14,epsrel=1e-13)[0]
num=-np.sum(a*np.sin(t)/t)
print("4-cosine certified (quad) bound: alpha >= %.7f"%(0.5+num/(2*den)), "sign changes",len(roots)-2)
