# Local search on the Alon et al. / Klein integrality-gap graph (Section 3.4).
# Usage: python3 klein_gap_check.py p n k   (e.g. 11 6000 3)
# Builds the graph from an Erdos-Turan Sidon set of size p on Z_n, drops wrap-around
# edges (so it is a DAG), and searches for a k-level assignment keeping many edges forward.
# The (k+1)/2 gap argument needs the kept fraction to be at most about (k-1)/(2k).
import numpy as np, sys
p, n, k = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
A = sorted(2*p*i + (i*i) % p for i in range(p))
diffs = [b-a for a in A for b in A if a < b]
assert len(set(diffs)) == len(diffs)          # Sidon property
src, dst = [], []
for i in range(n):
    for x in range(p):
        for y in range(x+1, p):
            u, v = (i+A[x]) % n, (i+A[y]) % n
            if u < v:
                src.append(u); dst.append(v)
src, dst = np.array(src), np.array(dst)
print(f"p={p} n={n} k={k} edges={len(src)} target <= {(k-1)/(2*k):.4f}")
rng = np.random.default_rng(0)
def kept(l): return np.mean(l[src] < l[dst])
def local_search(l, rounds=120):
    for _ in range(rounds):
        gain = np.zeros((n, k))
        for L in range(k):
            gain[:, L] = (np.bincount(dst, weights=(l[src] < L), minlength=n)
                          + np.bincount(src, weights=(l[dst] > L), minlength=n))
        best = gain.argmax(1)
        l = np.where(rng.random(n) < 0.2, best, l)   # move a random 20% to their best level
    return l
inits = {"random": rng.integers(0, k, n)}
for P in [3*k, 50, 200, 2*p*p, n//7, n//3, n]:
    inits[f"blocks{P}"] = (np.arange(n) % P) * k // P
best = max(kept(local_search(l.copy())) for l in inits.values())
print(f"best kept fraction found: {best:.4f}")
