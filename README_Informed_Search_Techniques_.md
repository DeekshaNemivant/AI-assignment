Informed Search Techniques

Python implementations of eight heuristic search algorithms, run on a small example graph. 
Unlike blind search, these algorithms use a heuristic h(n) to estimate how far each node is from the goal.
Algorithms

Here g(n) is the cost from the start to node n, h(n) is the estimated cost from n to the goal, and f(n) = g(n) + h(n).

Greedy Best-First Search: expands the node with the lowest h(n). It is fast but not optimal, since it ignores the cost already travelled.
Best-First Search: a general priority-queue search that always expands the most promising node by its evaluation function.
Dijkstra's Algorithm: expands the node with the lowest g(n) and uses no heuristic. It always finds the cheapest path.
A*: expands the node with the lowest f(n). It is optimal when h never overestimates the true cost (admissible).
Beam Search: keeps only the best k nodes at each level, saving memory but risking a missed solution.
RBFS: a memory-efficient version of best-first search that backtracks when a better alternative path exists.
IDA*: repeats depth-first search with a growing f-limit, using little memory.
Weighted A*: uses f(n) = g(n) + w·h(n) with w > 1 to find a solution faster, at the cost of optimality.

These Are Used
Greedy / Best-First: fast approximate routing, web crawling
Dijkstra: GPS navigation, network routing
A:* game pathfinding, robotics
Beam Search: machine translation, speech recognition
RBFS / IDA:* memory-limited puzzles like the 15-puzzle
Weighted A:* real-time planning where a near-optimal answer is enough
