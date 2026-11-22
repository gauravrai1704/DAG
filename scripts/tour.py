import sys, itertools, numpy as np
sys.argv=[sys.argv[0]]
from tri import solve
import tri
for n,m in [(4,6),(4,10),(5,6),(5,8),(6,5),(6,6)]:
    E=[(i,j) for i in range(n) for j in range(i+1,n)]
    print("transitive tournament n=%d m=%d: alpha=%.5f"%(n,m,solve(n,E,m)),flush=True)
