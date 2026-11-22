from tri import solve
for w,ms in [(3,[4,8,12,20]),(4,[4,6,8,10]),(5,[4,5,6]),(6,[3,4,5])]:
    for m in ms:
        E=[(i,j) for i in range(w) for j in range(w) if i!=j]
        print("K_%d both directions, m=%d: alpha=%.5f"%(w,m,solve(w,E,m)),flush=True)
