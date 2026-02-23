import random
from typing import List, Tuple

# grid[x][y][z] = 1 wall, 0 free
# We carve a "perfect maze" using randomized DFS on odd cells.

Vec3 = Tuple[int, int, int]


def _neighbors_2step(x: int, y: int, z: int, sx: int, sy: int, sz: int) -> List[Vec3]:
    out = []
    for dx, dy, dz in [(2, 0, 0), (-2, 0, 0), (0, 2, 0), (0, -2, 0), (0, 0, 2), (0, 0, -2)]:
        nx, ny, nz = x + dx, y + dy, z + dz
        if 1 <= nx < sx - 1 and 1 <= ny < sy - 1 and 1 <= nz < sz - 1:
            out.append((nx, ny, nz))
    return out


from typing import Optional

def generate_maze(sx: int, sy: int, sz: int, seed: Optional[int] = None):
    """
    Returns:
      grid: 3D list [sx][sy][sz] with 1=wall, 0=free
      start: (x,y,z)
      goal: (x,y,z)
    Notes:
      Use odd sizes for best carving: e.g. 11x11x5
    """
    if seed is not None:
        random.seed(seed)

    # Force odd sizes (carving relies on odd cells)
    if sx % 2 == 0:
        sx += 1
    if sy % 2 == 0:
        sy += 1
    if sz % 2 == 0:
        sz += 1

    grid = [[[1 for _ in range(sz)] for _ in range(sy)] for _ in range(sx)]

    # start at an odd cell
    start = (1, 1, 1)
    stack = [start]
    grid[1][1][1] = 0

    visited = set([start])

    while stack:
        cx, cy, cz = stack[-1]
        nbrs = _neighbors_2step(cx, cy, cz, sx, sy, sz)
        random.shuffle(nbrs)

        moved = False
        for nx, ny, nz in nbrs:
            if (nx, ny, nz) in visited:
                continue
            # carve wall between
            wx, wy, wz = (cx + nx) // 2, (cy + ny) // 2, (cz + nz) // 2
            grid[wx][wy][wz] = 0
            grid[nx][ny][nz] = 0
            visited.add((nx, ny, nz))
            stack.append((nx, ny, nz))
            moved = True
            break

        if not moved:
            stack.pop()

    # Choose a goal: farthest odd-ish cell from start (simple heuristic)
    # We'll scan all free cells and pick max Manhattan distance.
    best = start
    best_d = -1
    for x in range(1, sx - 1):
        for y in range(1, sy - 1):
            for z in range(1, sz - 1):
                if grid[x][y][z] == 0:
                    d = abs(x - start[0]) + abs(y - start[1]) + abs(z - start[2])
                    if d > best_d:
                        best_d = d
                        best = (x, y, z)

    goal = best
    return grid, start, goal