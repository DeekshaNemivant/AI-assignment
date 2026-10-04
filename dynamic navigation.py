
"""
UGV Navigation with Dynamic Obstacles (online replanning)
 
- Battlefield: 70 x 70 km, modelled as a 70 x 70 grid (1 cell = 1 km).
- The UGV does NOT know the obstacle map in advance. It only senses obstacles
  inside a square window around itself (SENSE_RADIUS cells).
- Obstacles are dynamic: after every UGV step a fraction of them move.
- At every step the UGV:
    1. senses its surroundings and refreshes its internal map
       (new obstacles are added, obstacles that have moved away are removed),
    2. replans with A* (8-direction, octile heuristic) from its current cell,
       treating unknown cells as free (optimistic planning),
    3. moves one cell along the plan, then the world changes again.
- The travelled trajectory is traced on the map and Measures of Effectiveness
  (MOE) are reported.
"""
 
import heapq
import math
import random
import time
 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
 
SIZE = 70            # grid is SIZE x SIZE cells
KM_PER_CELL = 1.0    # 70 x 70 km area
SENSE_RADIUS = 3     # UGV sensor sees a (2R+1) x (2R+1) window
MOVE_PROB = 0.25     # chance that a given obstacle moves after each step
MAX_ITERATIONS = 1500  # safety limit (moves + waits)
 
DENSITIES = {
    "low": 0.08,
    "medium": 0.15,
    "high": 0.22,
}
 
SQRT2 = math.sqrt(2)
MOVES = [
    (-1, 0, 1.0), (1, 0, 1.0),
    (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, SQRT2), (-1, 1, SQRT2),
    (1, -1, SQRT2), (1, 1, SQRT2),
]
 
 
def inside(p):
    return 0 <= p[0] < SIZE and 0 <= p[1] < SIZE
 
 
def heuristic(a, b):
    """Octile distance: admissible and consistent for 8-direction movement."""
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy) + (SQRT2 - 1) * min(dx, dy)
 
 
def astar(blocked, start, goal):
    """Return (path, nodes_expanded); path is None if no route is known."""
    open_set = [(heuristic(start, goal), 0.0, start)]
    g_score = {start: 0.0}
    parent = {start: None}
    expanded = 0
 
    while open_set:
        _, cost, current = heapq.heappop(open_set)
 
        if cost > g_score.get(current, float("inf")):
            continue  # stale queue entry
 
        expanded += 1
 
        if current == goal:
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            return path, expanded
 
        r, c = current
        for dr, dc, move_cost in MOVES:
            nxt = (r + dr, c + dc)
 
            if not inside(nxt) or nxt in blocked:
                continue
            # Do not cut the corner between two diagonally adjacent obstacles.
            if dr != 0 and dc != 0 and ((r + dr, c) in blocked or (r, c + dc) in blocked):
                continue
 
            new_cost = cost + move_cost
            if new_cost < g_score.get(nxt, float("inf")):
                g_score[nxt] = new_cost
                parent[nxt] = current
                heapq.heappush(
                    open_set, (new_cost + heuristic(nxt, goal), new_cost, nxt)
                )
 
    return None, expanded
 
 
def path_length(path):
    """Length of a path in cells (diagonal steps count as sqrt(2))."""
    total = 0.0
    for a, b in zip(path, path[1:]):
        total += SQRT2 if (a[0] != b[0] and a[1] != b[1]) else 1.0
    return total
 
 
def generate_environment(density, start, goal, rng):
    obstacles = set()
    for r in range(SIZE):
        for c in range(SIZE):
            if rng.random() < density:
                obstacles.add((r, c))
    obstacles.discard(start)
    obstacles.discard(goal)
    return obstacles
 
 
def move_dynamic_obstacles(obstacles, ugv, goal, rng):
    """
    A fraction of obstacles step to a free neighbouring cell.
    Obstacles never move onto the UGV or the goal, and never merge with each other.
    """
    protected = {ugv, goal}
    occupied = set(obstacles)
    updated = set()
 
    for ob in sorted(obstacles):  # sorted -> reproducible for a given seed
        if rng.random() < MOVE_PROB:
            options = [
                (ob[0] + dr, ob[1] + dc)
                for dr, dc, _ in MOVES
                if inside((ob[0] + dr, ob[1] + dc))
                and (ob[0] + dr, ob[1] + dc) not in occupied
                and (ob[0] + dr, ob[1] + dc) not in protected
            ]
            if options:
                new_cell = rng.choice(options)
                occupied.discard(ob)
                occupied.add(new_cell)
                updated.add(new_cell)
                continue
        updated.add(ob)
 
    return updated
 
 
def sense_obstacles(actual, known, position):
    """
    Refresh the UGV's map inside its sensor window.
    Cells in view are overwritten with what is really there, so obstacles
    that moved away are forgotten instead of lingering as 'ghost' obstacles.
    Returns the number of obstacle cells that were newly discovered.
    """
    newly_seen = 0
    for r in range(max(0, position[0] - SENSE_RADIUS), min(SIZE, position[0] + SENSE_RADIUS + 1)):
        for c in range(max(0, position[1] - SENSE_RADIUS), min(SIZE, position[1] + SENSE_RADIUS + 1)):
            cell = (r, c)
            if cell in actual:
                if cell not in known:
                    newly_seen += 1
                known.add(cell)
            else:
                known.discard(cell)
    return newly_seen
 
 
def count_turns(trajectory):
    turns = 0
    prev = None
    for a, b in zip(trajectory, trajectory[1:]):
        direction = (b[0] - a[0], b[1] - a[1])
        if prev is not None and direction != prev:
            turns += 1
        prev = direction
    return turns
 
 
def simulate(density, start, goal, rng):
    actual = generate_environment(density, start, goal, rng)
    initial_obstacles = set(actual)
 
    # Hindsight benchmark: shortest path if the initial map had been fully known and static.
    ideal_path, _ = astar(initial_obstacles, start, goal)
 
    known = set()
    current = start
    trajectory = [current]
    expected_path = None  # what the previous plan predicts the next plan to be
 
    stats = {
        "iterations": 0,
        "moves": 0,
        "waits": 0,
        "replans": 0,
        "path_changes": 0,
        "blocked_moves": 0,
        "obstacles_discovered": 0,
        "nodes_expanded": 0,
        "planning_time": 0.0,
    }
 
    t_start = time.perf_counter()
 
    for _ in range(MAX_ITERATIONS):
        if current == goal:
            break
        stats["iterations"] += 1
 
        # 1. Sense
        stats["obstacles_discovered"] += sense_obstacles(actual, known, current)
        known.discard(current)
 
        # 2. Plan with the information the UGV actually has
        t0 = time.perf_counter()
        path, expanded = astar(known, current, goal)
        stats["planning_time"] += time.perf_counter() - t0
        stats["replans"] += 1
        stats["nodes_expanded"] += expanded
 
        if path is None:
            # No known route: wait in place, let the world change, sense again.
            stats["waits"] += 1
            actual = move_dynamic_obstacles(actual, current, goal, rng)
            expected_path = None
            continue
 
        if expected_path is not None and path != expected_path:
            stats["path_changes"] += 1
 
        # 3. Move one cell
        next_cell = path[1] if len(path) > 1 else current
        if next_cell in actual:
            # Safety check (cannot normally happen because the next cell is within sensor range).
            known.add(next_cell)
            stats["blocked_moves"] += 1
            actual = move_dynamic_obstacles(actual, current, goal, rng)
            expected_path = None
            continue
 
        current = next_cell
        trajectory.append(current)
        stats["moves"] += 1
        expected_path = path[1:]
 
        # 4. The environment changes
        actual = move_dynamic_obstacles(actual, current, goal, rng)
 
    stats["total_time"] = time.perf_counter() - t_start
 
    return {
        "success": current == goal,
        "trajectory": trajectory,
        "current": current,
        "initial_obstacles": initial_obstacles,
        "final_obstacles": actual,
        "known": known,
        "ideal_path": ideal_path,
        "stats": stats,
    }
 
 
def compute_moe(result, start, goal):
    s = result["stats"]
    traj = result["trajectory"]
    travelled_km = path_length(traj) * KM_PER_CELL
    straight_km = math.hypot(goal[0] - start[0], goal[1] - start[1]) * KM_PER_CELL
    ideal_km = path_length(result["ideal_path"]) * KM_PER_CELL if result["ideal_path"] else None
 
    moe = {
        "Goal reached": result["success"],
        "Final position (row, col)": result["current"],
        "Distance travelled (km)": round(travelled_km, 3),
        "Straight-line distance (km)": round(straight_km, 3),
        "Static-map ideal distance (km)": round(ideal_km, 3) if ideal_km else "N/A",
        "Path optimality (ideal / travelled)": (
            round(ideal_km / travelled_km, 4) if ideal_km and travelled_km else "N/A"
        ),
        "Detour over straight line (%)": (
            round(100 * (travelled_km - straight_km) / straight_km, 2) if straight_km else 0.0
        ),
        "Moves made": s["moves"],
        "Waiting steps (no known route)": s["waits"],
        "Total time steps": s["iterations"],
        "Number of turns": count_turns(traj),
        "Replanning count": s["replans"],
        "Plans that changed the route": s["path_changes"],
        "Blocked-move events": s["blocked_moves"],
        "Obstacle cells discovered": s["obstacles_discovered"],
        "Obstacles in UGV map at end": len(result["known"]),
        "Total obstacles (actual)": len(result["final_obstacles"]),
        "A* nodes expanded (total)": s["nodes_expanded"],
        "Avg planning time (ms)": round(1000 * s["planning_time"] / s["replans"], 3) if s["replans"] else 0.0,
        "Total computation time (s)": round(s["total_time"], 4),
    }
    return moe
 
 
def print_moe(moe, density_name, start, goal, seed):
    print("\nDYNAMIC UGV - MEASURES OF EFFECTIVENESS (MOE)")
    print("=" * 58)
    print(f"{'Density level':<38}: {density_name}")
    print(f"{'Start (row, col)':<38}: {start}")
    print(f"{'Goal (row, col)':<38}: {goal}")
    print(f"{'Sensor radius (cells)':<38}: {SENSE_RADIUS}")
    print(f"{'Obstacle move probability / step':<38}: {MOVE_PROB}")
    print(f"{'Random seed':<38}: {seed}")
    for key, value in moe.items():
        print(f"{key:<38}: {value}")
    print("=" * 58)
 
 
def plot_result(result, start, goal, density_name, moe):
    fig, ax = plt.subplots(figsize=(9, 9))
 
    # 0 = free, 1 = where obstacles started, 2 = where obstacles are at the end
    image = [[0] * SIZE for _ in range(SIZE)]
    for r, c in result["initial_obstacles"]:
        image[r][c] = 1
    for r, c in result["final_obstacles"]:
        image[r][c] = 2
    cmap = ListedColormap(["#f7f7f7", "#c9c9c9", "#2b2b2b"])
    ax.imshow(image, cmap=cmap, vmin=0, vmax=2, origin="upper", interpolation="nearest")
 
    traj = result["trajectory"]
    ax.plot([c for r, c in traj], [r for r, c in traj],
            color="red", linewidth=2, label="UGV trajectory")
    ax.scatter(start[1], start[0], s=120, c="green", marker="o",
               edgecolors="black", zorder=5, label="Start")
    ax.scatter(goal[1], goal[0], s=200, c="gold", marker="*",
               edgecolors="black", zorder=5, label="Goal")
    cur = result["current"]
    ax.scatter(cur[1], cur[0], s=90, c="deepskyblue", marker="D",
               edgecolors="black", zorder=6, label="UGV (final position)")
 
    handles, labels = ax.get_legend_handles_labels()
    handles += [Patch(color="#c9c9c9"), Patch(color="#2b2b2b")]
    labels += ["Obstacle start positions", "Obstacle end positions"]
    ax.legend(handles, labels, loc="upper right", fontsize=8)
 
    status = "Goal reached" if moe["Goal reached"] else "Goal NOT reached"
    ax.set_title(
        f"UGV Dynamic Obstacle Navigation - {density_name.title()} Density ({status})\n"
        f"Distance: {moe['Distance travelled (km)']} km | "
        f"Replans: {moe['Replanning count']} | "
        f"Route changes: {moe['Plans that changed the route']} | "
        f"Waits: {moe['Waiting steps (no known route)']}",
        fontsize=10,
    )
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.grid(alpha=0.15)
    plt.tight_layout()
    filename = f"ugv_dynamic_{density_name}.png"
    plt.savefig(filename, dpi=150)
    print(f"Map saved as {filename}")
    plt.show()
 
 
def read_point(prompt, default):
    text = input(prompt).strip() or default
    parts = text.replace(" ", "").split(",")
    if len(parts) != 2:
        raise ValueError
    point = (int(parts[0]), int(parts[1]))
    if not inside(point):
        raise ValueError
    return point
 
 
def main():
    print("UGV DYNAMIC OBSTACLE NAVIGATION (online replanning)")
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
 
    result = simulate(DENSITIES[density_name], start, goal, rng)
    moe = compute_moe(result, start, goal)
    print_moe(moe, density_name, start, goal, seed)
 
    print("\nTrajectory (row, col):")
    print(" -> ".join(f"({r},{c})" for r, c in result["trajectory"]))
 
    plot_result(result, start, goal, density_name, moe)
 
 
if __name__ == "__main__":
    main()