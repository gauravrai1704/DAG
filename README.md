# Randomized Rounding for Minimum-Weight Edge Deletion in DAGs

CSD319 (Design and Analysis of Algorithms) project, Shiv Nadar University.

Gaurav Rai (2410110133), Shaurya Sharma (2410110317), Sharanya Gupta (2410110315), Kashvi Tak (2410110165)

## The problem

Given a DAG with edge costs and a number `k`, delete the cheapest set of edges so that no path with `k` edges is left (DED(k)). The best previously published algorithm (Kalantarzadeh, Klein, Reis, APPROX 2025) rounds an LP using random vertex labels and achieves ratio `0.549(k+1)`. Under the Unique Games Conjecture nothing better than `floor(k/2)` is possible.

## What is in the report

- the threshold, Klein and label-rounding algorithms, with proofs that their output is always valid, cost bounds and running time
- the known hardness results (`floor(k/2)` under the UGC, `sqrt(2)` NP-hardness for `k >= 4`) and the integrality gap
- our results:
  - a step-shaped label distribution with ratio about `0.5446(k+1)`
  - a lower bound showing no independent label distribution beats `0.5431(k+1)`
  - a `0.5(k+1)` rounding for DAGs that come with a 3-colouring, and `0.5064(k+1)` for planar DAGs
  - a proof that no single label distribution reaches `0.5(k+1)` on every DAG (fails for complete DAGs with `n >= 429` vertices)
- a note on the `(k+1)/2` integrality gap from Klein's thesis: the argument is valid for `k = 2`, but for `k >= 3` its "at most half the edges go forward" step fails on the instances we tested, so we treat that case as unconfirmed

## What is proved and what is numerical

| Result | Status |
|---|---|
| `0.5(k+1)` for 3-colourable DAGs | proof |
| no single distribution reaches `0.5` on all DAGs | proof |
| independent labels cannot beat `0.5431(k+1)` | proof, with the integrals checked numerically |
| `0.5064(k+1)` for planar DAGs | LP solution checked with exact fractions |
| `0.5446(k+1)` with our 48-piece density | numerical check in floating point (not interval arithmetic) |
| integrality-gap step fails for `k = 3` | experiment on small instances, not a disproof of the asymptotic claim |

## Layout

```
report/main.tex   report source (acmart, two column, compile with pdfLaTeX)
report/main.pdf   compiled report
scripts/          code for every number in the report
```

## Running the scripts

```
pip install -r requirements.txt
cd scripts
```

| Script | What it does |
|---|---|
| `verify.py` | known constants (2-sqrt(2), 0.5482, 0.539) and the multi-cosine lower bounds |
| `cert4.py`, `cert5.py` | numerical integration check of the 4- and 5-cosine bounds |
| `optd.py N` | searches for the best step density with `N` pieces, saves `best_p_N.npy` |
| `checkd.py FILE`, `bracket.py FILE` | checks a saved density (fine grid, Monte Carlo, monotone bracket) |
| `kw.py c:m ...` | grid LP for `c` colours on grid `m`, answer re-checked with exact fractions, e.g. `python3 kw.py 4:40` |
| `fig_ratio_curves.py` | data for Figure 1 (the ratio curves for uniform, KKR and our labels) |
| `klein_gap_check.py p n k [rounds]` | local search on the integrality-gap graph of Section 3.4 (kept fraction, and share of cross-level edges going forward), e.g. `11 6000 3` or `23 20000 3 40` |
| `tri.py`, `tour.py`, `sym.py`, `inspect_tri.py` | correlated-label LPs on small graphs and complete DAGs |

`best_p_24.npy` and `best_p_48.npy` are the densities used in the report (ratios 0.54499 and 0.54451).

## References

Main paper: S. Kalantarzadeh, N. Klein, V. Reis, *A Randomized Rounding Approach for DAG Edge Deletion*, APPROX 2025, [arXiv:2507.07943](https://arxiv.org/abs/2507.07943). The full list is in the report.
