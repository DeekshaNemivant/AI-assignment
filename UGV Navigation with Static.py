

"""
UGV Navigation with Static (a-priori known) Obstacles
 
- Battlefield: 70 x 70 km, modelled as a 70 x 70 grid (1 cell = 1 km).
- Obstacle density is generated randomly at three levels: low / medium / high.
- The UGV uses A* (8-direction movement, octile heuristic) to find the
  shortest obstacle-free path from a user-specified start to a goal.
- The path is traced on the map and Measures of Effectiveness (MOE) are reported.
"""
import heapq
import math
import random
import time
 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
 
SIZE = 70          # grid is SIZE x SIZE cells
KM_PER_CELL = 1.0  # 70 x 70 km area
DENSITIES = {
    "low": 0.15,
    "medium": 0.25,
    "high": 0.35,
}
MAX_MAP_ATTEMPTS = 500
 
SQRT2 = math.sqrt(2)
MOVES = [
    (-1, 0, 1.0), (1, 0, 1.0),
    (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, SQRT2), (-1, 1, SQRT2),
    (1, -1, SQRT2), (1, 1, SQRT2),
]
 
 
def heuristic(a, b):
    """Octile distance: admissible and consistent for 8-direction movement."""
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy) + (SQRT2 - 1) * min(dx, dy)
 
 
def astar(grid, start, goal):
    """Return (path, distance_in_cells, nodes_expanded); path is None if unreachable."""
    rows, cols = len(grid), len(grid[0])
    open_set = [(heuristic(start, goal), 0.0, start)]
    g_score = {start: 0.0}
    parent = {start: None}
    expanded = 0
 
    while open_set:
        _, current_g, current = heapq.heappop(open_set)
 
        if current_g > g_score.get(current, float("inf")):
            continue  # stale queue entry
 
        expanded += 1
 
        if current == goal:
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            return path, current_g, expanded
 
        r, c = current
        for dr, dc, move_cost in MOVES:
            nr, nc = r + dr, c + dc
 
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid[nr][nc]:
                continue
            # Do not cut the corner between two diagonally adjacent obstacles.
            if dr != 0 and dc != 0 and (grid[r + dr][c] or grid[r][c + dc]):
                continue
 
            neighbour = (nr, nc)
            new_g = current_g + move_cost
            if new_g < g_score.get(neighbour, float("inf")):
                g_score[neighbour] = new_g
                parent[neighbour] = current
                heapq.heappush(
                    open_set, (new_g + heuristic(neighbour, goal), new_g, neighbour)
                )
 
    return None, None, expanded
 
 
def create_grid(density, start, goal, rng):
    """Random obstacle map guaranteed to contain a feasible start->goal route."""
    for _ in range(MAX_MAP_ATTEMPTS):
        grid = [[rng.random() < density for _ in range(SIZE)] for _ in range(SIZE)]
        grid[start[0]][start[1]] = False
        grid[goal[0]][goal[1]] = False
        path, _, _ = astar(grid, start, goal)
        if path:
            return grid
    raise RuntimeError("Could not generate a solvable map; try a lower density.")
 
 
def count_turns(path):
    """Number of heading changes along the path."""
    turns = 0
    prev = None
    for a, b in zip(path, path[1:]):
        direction = (b[0] - a[0], b[1] - a[1])
        if prev is not None and direction != prev:
            turns += 1
        prev = direction
    return turns
 
 
def compute_moe(grid, path, distance, expanded, elapsed, start, goal):
    straight = math.hypot(goal[0] - start[0], goal[1] - start[1]) * KM_PER_CELL
    obstacles = sum(sum(row) for row in grid)
    dist_km = distance * KM_PER_CELL
    return {
        "Actual obstacle density (%)": round(100 * obstacles / (SIZE * SIZE), 2),
        "Obstacle cells": obstacles,
        "Path found": True,
        "Shortest path distance (km)": round(dist_km, 3),
        "Straight-line distance (km)": round(straight, 3),
        "Path efficiency (straight / path)": round(straight / dist_km, 4) if dist_km else 1.0,
        "Detour over straight line (%)": round(100 * (dist_km - straight) / straight, 2) if straight else 0.0,
        "Waypoints (cells visited)": len(path),
        "Number of turns": count_turns(path),
        "Nodes expanded": expanded,
        "Search space explored (%)": round(100 * expanded / (SIZE * SIZE - obstacles), 2),
        "Computation time (ms)": round(elapsed * 1000, 3),
    }
 
 
def print_moe(moe, density_name, start, goal, seed):
    print("\nMEASURES OF EFFECTIVENESS (MOE)")
    print("=" * 52)
    print(f"{'Density level':<36}: {density_name}")
    print(f"{'Start (row, col)':<36}: {start}")
    print(f"{'Goal (row, col)':<36}: {goal}")
    print(f"{'Random seed':<36}: {seed}")
    for key, value in moe.items():
        print(f"{key:<36}: {value}")
    print("=" * 52)
 
 
def plot_grid(grid, path, start, goal, density_name, moe):
    fig, ax = plt.subplots(figsize=(9, 9))
 
    # Row 0 is drawn at the top; obstacles dark, free space light.
    cmap = ListedColormap(["#f2f2f2", "#2b2b2b"])
    ax.imshow(grid, cmap=cmap, origin="upper", interpolation="nearest")
 
    if path:
        ax.plot([c for r, c in path], [r for r, c in path],
                color="red", linewidth=2, label="UGV path")
 
    ax.scatter(start[1], start[0], s=120, c="green", marker="o",
               edgecolors="black", zorder=5, label="Start")
    ax.scatter(goal[1], goal[0], s=200, c="gold", marker="*",
               edgecolors="black", zorder=5, label="Goal")
 
    ax.set_title(
        f"UGV Navigation - {density_name.title()} Obstacle Density\n"
        f"Distance: {moe['Shortest path distance (km)']} km | "
        f"Turns: {moe['Number of turns']} | "
        f"Nodes expanded: {moe['Nodes expanded']} | "
        f"Time: {moe['Computation time (ms)']} ms"
    )
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.15)
    plt.tight_layout()
    plt.savefig(f"ugv_path_{density_name}.png", dpi=150)
    print(f"Map saved as ugv_path_{density_name}.png")
    plt.show()
 
 
def read_point(prompt, default):
    text = input(prompt).strip() or default
    parts = text.replace(" ", "").split(",")
    if len(parts) != 2:
        raise ValueError
    point = (int(parts[0]), int(parts[1]))
    if not all(0 <= v < SIZE for v in point):
        raise ValueError
    return point
 
 
def main():
    print("UGV STATIC OBSTACLE NAVIGATION")
    print(f"Battlefield: {SIZE} x {SIZE} km (1 cell = {KM_PER_CELL:g} km)")
 
    density_name = input("Obstacle density (low/medium/high) [medium]: ").strip().lower()
    if density_name not in DENSITIES:
        print("Unknown level, using 'medium'.")
        density_name = "medium"
 
    try:
        start = read_point(f"Start row,col (0-{SIZE - 1}) [2,2]: ", "2,2")
        goal = read_point(f"Goal row,col (0-{SIZE - 1}) [67,67]: ", "67,67")
    except ValueError:
        print(f"Invalid coordinates. Use two integers between 0 and {SIZE - 1}, e.g. 2,2.")
        return
 
    if start == goal:
        print("Start and goal are the same cell; the UGV is already at the goal.")
        return
 
    seed_text = input("Random seed (blank = random): ").strip()
    seed = int(seed_text) if seed_text.lstrip("-").isdigit() else int(time.time()) % 100000
    rng = random.Random(seed)
 
    grid = create_grid(DENSITIES[density_name], start, goal, rng)
 
    t0 = time.perf_counter()
    path, distance, expanded = astar(grid, start, goal)
    elapsed = time.perf_counter() - t0
 
    moe = compute_moe(grid, path, distance, expanded, elapsed, start, goal)
    print_moe(moe, density_name, start, goal, seed)
 
    print("\nPath (row, col):")
    print(" -> ".join(f"({r},{c})" for r, c in path))
 
    plot_grid(grid, path, start, goal, density_name, moe)
 
 
if __name__ == "__main__":
    main()
 