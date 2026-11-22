# Data for Figure 1: Pr[X-Y <= t]/(1+t) for i.i.d. labels X,Y, three label distributions.
# Writes fig1.dat (columns: t uniform kkr ours48)
import numpy as np
from scipy.integrate import quad

ts = np.linspace(-0.6, 1.0, 161)

def uniform(t):
    F = 0.5*(1+t)**2 if t < 0 else 1 - 0.5*(1-t)**2
    return F/(1+t)

p = lambda x: 2/3 + (23/3)*(1-2*x)**22          # KKR density
P = lambda x: 0.0 if x <= 0 else (1.0 if x >= 1 else 2*x/3 + (1-(1-2*x)**23)/6)
def kkr(t):
    lo, hi = max(0, -t), min(1, 1-t)
    a = quad(lambda y: p(y)*P(y+t), lo, hi, limit=200)[0] if hi > lo else 0
    b = quad(p, max(0, 1-t), 1)[0] if t > 0 else 0
    return (a+b)/(1+t)

q = np.load('best_p_48.npy'); N = len(q); c = np.correlate(q, q, 'full'); D = np.arange(-(N-1), N)
T = lambda u: np.where(u < -1, 0, np.where(u < 0, 0.5*(1+u)**2, np.where(u < 1, 1-0.5*(1-u)**2, 1)))
ours = lambda t: float(T((t - D/N)*N) @ c)/(1+t)

with open('fig1.dat', 'w') as f:
    f.write('t uniform kkr ours\n')
    for t in ts:
        f.write(f'{t:.3f} {uniform(t):.5f} {kkr(t):.5f} {ours(t):.5f}\n')
print('peaks:', max(uniform(t) for t in ts), max(kkr(t) for t in ts), max(ours(t) for t in ts))
