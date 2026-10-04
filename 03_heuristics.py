#heuristics 

 
# 8-puzzle with heuristics (0 is the blank)
import heapq
import random
from collections import deque
 
goal = (1, 2, 3, 4, 5, 6, 7, 8, 0)
 
 
def moves(state):
    # move the blank up, down, left or right
    b = state.index(0)
    r, c = divmod(b, 3)
    result = []
    for dr, dc in (-1, 0), (1, 0), (0, -1), (0, 1):
        nr, nc = r + dr, c + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            n = nr * 3 + nc
            s = list(state)
            s[b], s[n] = s[n], s[b]
            result.append(tuple(s))
    return result
 
 
# relaxed problem: a tile can jump anywhere
def misplaced(s):
    return sum(1 for i in range(9) if s[i] and s[i] != goal[i])
 
 
# relaxed problem: a tile can slide onto an occupied square
def manhattan(s):
    total = 0
    for i, t in enumerate(s):
        if t:
            total += abs(i // 3 - (t - 1) // 3) + abs(i % 3 - (t - 1) % 3)
    return total
 
 
# subproblem: only track a few tiles, hide the rest as -1
def mask(s, pattern):
    return tuple(t if t == 0 or t in pattern else -1 for t in s)
 
 
def build_db(pattern):
    # BFS from the goal gives the distance to the goal for every masked state
    start = mask(goal, pattern)
    dist = {start: 0}
    q = deque([start])
    while q:
        s = q.popleft()
        for n in moves(s):
            if n not in dist:
                dist[n] = dist[s] + 1
                q.append(n)
    return dist
 
 
p1, p2 = {1, 2, 3, 4}, {5, 6, 7, 8}
db1, db2 = build_db(p1), build_db(p2)
 
 
def pattern_h(s):
    return max(db1[mask(s, p1)], db2[mask(s, p2)])
 
 
# w = 1 is normal A*, w > 1 is weighted A* (faster, answer may be longer)
def solve(start, h, w=1):
    pq = [(w * h(start), 0, start)]
    best = {start: 0}
    expanded = 0
    while pq:
        f, g, s = heapq.heappop(pq)
        if g > best[s]:
            continue
        if s == goal:
            return g, expanded
        expanded += 1
        for n in moves(s):
            if n not in best or g + 1 < best[n]:
                best[n] = g + 1
                heapq.heappush(pq, (g + 1 + w * h(n), g + 1, n))
 
 
def scramble(n):
    s = goal
    for _ in range(n):
        s = random.choice(moves(s))
    return s
 
 
random.seed(1)
puzzles = [scramble(60) for _ in range(15)]
heuristics = [("misplaced", misplaced), ("manhattan", manhattan), ("pattern db", pattern_h)]
 
 
def average(h, w=1):
    results = [solve(p, h, w) for p in puzzles]
    return sum(r[1] for r in results) / 15, sum(r[0] for r in results) / 15
 
 
# admissible means h never goes above the real cost
real = [solve(p, manhattan)[0] for p in puzzles]
for name, h in heuristics:
    print(name, "admissible:", all(h(p) <= c for p, c in zip(puzzles, real)))
 
print("\nA* (nodes, length)")
for name, h in heuristics:
    print(name, average(h))
 
print("\nweighted A* with manhattan (nodes, length)")
for w in (1, 1.5, 2, 5):
    print("w =", w, average(manhattan, w))