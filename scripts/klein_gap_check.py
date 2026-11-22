# Local search on the Alon et al. / Klein integrality-gap graph (Section 3.4 of the report).
# Usage: python3 klein_gap_check.py p n k [rounds]   e.g. 11 6000 3
#
# Graph: Erdos-Turan Sidon set A of size r=p, vertices Z_n, edge i+a -> i+b for a<b in A;
# wrap-around edges (u > v as integers) are dropped so the result is a DAG.
# Search: over level assignments l: V -> {0..k-1}, maximise the number of forward edges
# (l(u) < l(v)), i.e. the edges a k-ordering keeps. 8 starts (1 random + 7 periodic block
# patterns), each followed by best-response local search (each round, a random 20% of the
# vertices move to their best level). Any assignment found is a valid k-ordering, so the
# best value is a lower bound on the true maximum kept fraction for this instance.
import numpy as np, sys
p, n, k = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 120
A = sorted(2*p*i + (i*i) % p for i in range(p))
diffs = [b-a for a in A for b in A if a < b]
assert len(set(diffs)) == len(diffs)                       # Sidon property
src, dst = [], []
total = 0
for i in range(n):
    for x in range(p):
        for y in range(x+1, p):
            u, v = (i+A[x]) % n, (i+A[y]) % n
            total += 1
            if u < v:
                src.append(u); dst.append(v)
src, dst = np.array(src), np.array(dst)
rng = np.random.default_rng(0)
def stats(l):
    f = np.sum(l[src] < l[dst]); b = np.sum(l[src] > l[dst])
    return f/len(src), f/(f+b)
def local_search(l):
    for _ in range(rounds):
        gain = np.zeros((n, k))
        for L in range(k):
            gain[:, L] = (np.bincount(dst, weights=(l[src] < L), minlength=n)
                          + np.bincount(src, weights=(l[dst] > L), minlength=n))
        l = np.where(rng.random(n) < 0.2, gain.argmax(1), l)
    return l
inits = [rng.integers(0, k, n)] + [(np.arange(n) % P) * k // P for P in [3*k, 50, 200, 2*p*p, n//7, n//3, n]]
res = [stats(local_search(l.copy())) for l in inits]
kept = [a for a, _ in res]; fwd = [b for _, b in res]
best = int(np.argmax(kept))
print(f"r={p} n={n} k={k} edges={len(src)} (wrap-around removed: {1-len(src)/total:.2%})")
print(f"  kept fraction: best {kept[best]:.4f}, mean over 8 starts {np.mean(kept):.4f}  "
      f"(the gap argument needs <= {(k-1)/(2*k):.4f})")
print(f"  share of cross-level edges going forward in best: {fwd[best]:.4f}  (the 'at most half' step needs <= 0.5)")
