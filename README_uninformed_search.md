Uninformed Search Techniques

Python implementations of five blind search algorithms. 
They use no heuristic and rely only on the graph structure (and edge costs, for UCS).
Algorithms
Depth-First Search (DFS): follows one branch as deep as possible before backtracking. It uses little memory but is not optimal and can get stuck in infinite or cyclic spaces.
Breadth-First Search (BFS): explores the graph level by level using a queue. It always finds the shallowest goal, so it is optimal when every step costs the same.
Uniform-Cost Search (UCS): expands the node with the lowest path cost so far using a priority queue. It finds the cheapest path even when edge costs differ.
Depth-Limited Search (DLS): DFS with a fixed depth limit, which prevents endless descent. It can miss the goal if the limit is too small.
Iterative Deepening DFS (IDDFS): runs DLS repeatedly with an increasing limit. It combines the low memory use of DFS with the completeness of BFS.

Where These Are Used
DFS: maze solving, cycle detection, backtracking
BFS: shortest path in unweighted graphs, web crawling
UCS: routing with varying edge costs
DLS: searches with a known depth bound
IDDFS: large search spaces with limited memory
