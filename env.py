import math
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
from maze_gen import generate_maze

Vec3 = Tuple[int, int, int]

DIRS: List[Vec3] = [
    (1, 0, 0),   # 0 +X
    (-1, 0, 0),  # 1 -X
    (0, 1, 0),   # 2 +Y
    (0, -1, 0),  # 3 -Y
    (0, 0, 1),   # 4 +Z
    (0, 0, -1),  # 5 -Z
]


@dataclass
class StepResult:
    obs: np.ndarray
    reward: float
    done: bool


class Maze3DEnv:
    """
    3D Maze RL Environment
      grid[x][y][z] = 1 wall, 0 free
    Observation = 18 floats:
      [0:6]  danger in 6 dirs (1 if wall)
      [6:12] goal relative booleans (goal x>agent x, goal x<..., etc)
      [12:15] normalized deltas (dx/(sx-1), dy/(sy-1), dz/(sz-1))
      [15:18] last move vector (dx,dy,dz) in {-1,0,1}
    Actions = 6 (±X ±Y ±Z)
    """

    def __init__(self, sx=11, sy=11, sz=5, max_steps=400, seed=None):
        self.sx = sx
        self.sy = sy
        self.sz = sz
        self.max_steps = max_steps
        self.seed = seed

        self.grid = None
        self.agent: Vec3 = (1, 1, 1)
        self.goal: Vec3 = (1, 1, 1)
        self.steps = 0
        self.last_move = (0, 0, 0)

        self.reset()

    def reset(self):
        self.grid, self.agent, self.goal = generate_maze(self.sx, self.sy, self.sz, seed=self.seed)
        self.steps = 0
        self.last_move = (0, 0, 0)
        return self._obs()

    def _in_bounds(self, p: Vec3) -> bool:
        x, y, z = p
        return 0 <= x < self.sx and 0 <= y < self.sy and 0 <= z < self.sz

    def _is_wall(self, p: Vec3) -> bool:
        x, y, z = p
        if not self._in_bounds(p):
            return True
        return self.grid[x][y][z] == 1

    def _dist(self, a: Vec3, b: Vec3) -> float:
        return abs(a[0]-b[0]) + abs(a[1]-b[1]) + abs(a[2]-b[2])

    def _obs(self) -> np.ndarray:
        ax, ay, az = self.agent
        gx, gy, gz = self.goal

        danger = []
        for dx, dy, dz in DIRS:
            nx, ny, nz = ax + dx, ay + dy, az + dz
            danger.append(1.0 if self._is_wall((nx, ny, nz)) else 0.0)

        goal_rel = [
            1.0 if gx > ax else 0.0,
            1.0 if gx < ax else 0.0,
            1.0 if gy > ay else 0.0,
            1.0 if gy < ay else 0.0,
            1.0 if gz > az else 0.0,
            1.0 if gz < az else 0.0,
        ]

        dxn = (gx - ax) / max(1, (self.sx - 1))
        dyn = (gy - ay) / max(1, (self.sy - 1))
        dzn = (gz - az) / max(1, (self.sz - 1))

        ldx, ldy, ldz = self.last_move

        obs = np.array(danger + goal_rel + [dxn, dyn, dzn] + [ldx, ldy, ldz], dtype=np.float32)
        assert obs.shape == (18,)
        return obs

    def step(self, action: int) -> StepResult:
        self.steps += 1

        # Base step penalty encourages shorter paths
        reward = -0.01
        done = False

        prev_dist = self._dist(self.agent, self.goal)

        action = int(action)
        if action < 0 or action >= 6:
            action = 0

        dx, dy, dz = DIRS[action]
        nx = self.agent[0] + dx
        ny = self.agent[1] + dy
        nz = self.agent[2] + dz
        nxt = (nx, ny, nz)

        # If wall, do not move, penalty
        if self._is_wall(nxt):
            reward -= 0.20
            self.last_move = (0, 0, 0)
        else:
            self.agent = nxt
            self.last_move = (dx, dy, dz)

        # Shaping: reward improvements in distance to goal
        new_dist = self._dist(self.agent, self.goal)
        reward += (prev_dist - new_dist) * 0.10

        # Goal reached
        if self.agent == self.goal:
            reward += 10.0
            done = True

        # Timeout
        if self.steps >= self.max_steps:
            done = True

        return StepResult(obs=self._obs(), reward=float(reward), done=done)

    def render_ascii_slice(self, z: int):
        """Debug: prints a 2D slice at z."""
        z = max(0, min(self.sz - 1, z))
        rows = []
        for y in range(self.sy):
            row = []
            for x in range(self.sx):
                if (x, y, z) == self.agent:
                    row.append("A")
                elif (x, y, z) == self.goal:
                    row.append("G")
                elif self.grid[x][y][z] == 1:
                    row.append("#")
                else:
                    row.append(".")
            rows.append("".join(row))
        print("\n".join(rows))