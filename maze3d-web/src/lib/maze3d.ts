// maze3d-web/src/lib/maze3d.ts
// 3D Maze generator + tiny episode/state helpers for the browser demo.
// Grid convention: 1 = wall, 0 = open.

export type Vec3 = [number, number, number]

export type CreateEpisodeArgs = {
  sx: number
  sy: number
  sz: number
  seed?: number
}

export type Episode = {
  size: Vec3
  grid: Uint8Array
  start: Vec3
  goal: Vec3
  pos: Vec3
  steps: number
  done: boolean
  lastAction: number
}

function clamp01(x: number) {
  return Math.max(0, Math.min(1, x))
}

function idx(size: Vec3, x: number, y: number, z: number) {
  const [sx, sy] = size
  return x + sx * (y + sy * z)
}

function inBounds(size: Vec3, x: number, y: number, z: number) {
  const [sx, sy, sz] = size
  return x >= 0 && y >= 0 && z >= 0 && x < sx && y < sy && z < sz
}

function manhattan(a: Vec3, b: Vec3) {
  return Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]) + Math.abs(a[2] - b[2])
}

// Deterministic RNG for procedural mazes
function mulberry32(seed: number) {
  let t = seed >>> 0
  return () => {
    t += 0x6d2b79f5
    let r = Math.imul(t ^ (t >>> 15), 1 | t)
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r)
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296
  }
}

/**
 * Generate a simple 3D "DFS carved" maze.
 * - Force odd sizes
 * - Start with all walls
 * - Carve passages by walking on odd coordinates and knocking down walls between cells.
 */
export function generateMaze(size: Vec3, rng: () => number = Math.random): { size: Vec3; grid: Uint8Array } {
  let [sx, sy, sz] = size
  if (sx % 2 === 0) sx += 1
  if (sy % 2 === 0) sy += 1
  if (sz % 2 === 0) sz += 1
  const fixed: Vec3 = [sx, sy, sz]

  const grid = new Uint8Array(sx * sy * sz)
  grid.fill(1)

  // helper: mark open
  const open = (x: number, y: number, z: number) => {
    grid[idx(fixed, x, y, z)] = 0
  }

  // neighbors 2 steps away (odd cell graph)
  const neighbors2 = (x: number, y: number, z: number) => {
    const out: Array<[number, number, number, number, number, number]> = []
    const dirs: Vec3[] = [
      [2, 0, 0],
      [-2, 0, 0],
      [0, 2, 0],
      [0, -2, 0],
      [0, 0, 2],
      [0, 0, -2],
    ]
    for (const [dx, dy, dz] of dirs) {
      const nx = x + dx,
        ny = y + dy,
        nz = z + dz
      if (inBounds(fixed, nx, ny, nz)) {
        // also return the wall cell between (1 step)
        out.push([nx, ny, nz, x + dx / 2, y + dy / 2, z + dz / 2])
      }
    }
    return out
  }

  // Pick a start cell (odd coords)
  const start: Vec3 = [1, 1, 1]
  open(start[0], start[1], start[2])

  const stack: Vec3[] = [[...start]]
  const visited = new Uint8Array(sx * sy * sz)
  visited[idx(fixed, start[0], start[1], start[2])] = 1

  while (stack.length) {
    const [x, y, z] = stack[stack.length - 1]
    const nbs = neighbors2(x, y, z).filter(([nx, ny, nz]) => visited[idx(fixed, nx, ny, nz)] === 0)

    if (nbs.length === 0) {
      stack.pop()
      continue
    }

    // random next neighbor
    const pick = nbs[Math.floor(rng() * nbs.length)]
    const [nx, ny, nz, wx, wy, wz] = pick

    // carve: open wall between + open neighbor cell
    open(wx, wy, wz)
    open(nx, ny, nz)

    visited[idx(fixed, nx, ny, nz)] = 1
    stack.push([nx, ny, nz])
  }

  return { size: fixed, grid }
}

/**
 * Stable API the UI imports.
 * Builds a new episode with a procedurally generated maze.
 */
export function createEpisode({ sx, sy, sz, seed }: CreateEpisodeArgs): Episode {
  const rng = seed == null ? Math.random : mulberry32(seed)
  const { size, grid } = generateMaze([sx, sy, sz], rng)

  const start: Vec3 = [1, 1, 1]
  const goal: Vec3 = [size[0] - 2, size[1] - 2, size[2] - 2]

  // ensure start/goal are open
  grid[idx(size, start[0], start[1], start[2])] = 0
  grid[idx(size, goal[0], goal[1], goal[2])] = 0

  return {
    size,
    grid,
    start,
    goal,
    pos: [...start],
    steps: 0,
    done: false,
    lastAction: 0,
  }
}

/**
 * Step function (6 actions):
 * 0:+X 1:-X 2:+Y 3:-Y 4:+Z 5:-Z
 */
export function stepEpisode(ep: Episode, action: number): Episode {
  if (ep.done) return ep

  const dirs: Vec3[] = [
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
  ]
  const d = dirs[action] ?? dirs[0]
  const nx = ep.pos[0] + d[0]
  const ny = ep.pos[1] + d[1]
  const nz = ep.pos[2] + d[2]

  // move only if target is open
  if (inBounds(ep.size, nx, ny, nz) && ep.grid[idx(ep.size, nx, ny, nz)] === 0) {
    ep.pos = [nx, ny, nz]
  }

  ep.lastAction = action
  ep.steps += 1

  // goal check
  if (ep.pos[0] === ep.goal[0] && ep.pos[1] === ep.goal[1] && ep.pos[2] === ep.goal[2]) {
    ep.done = true
  }

  // safety cap so we don't run forever
  const maxSteps = ep.size[0] * ep.size[1] * ep.size[2] * 4
  if (ep.steps >= maxSteps) ep.done = true

  return ep
}

/**
 * 18-dim observation vector (Float32Array length 18).
 * Designed to be simple + stable for ONNX input.
 *
 * Layout:
 * 0..2   : (goal - pos) normalized to [-1, 1]
 * 3..5   : pos normalized to [0, 1]
 * 6..11  : 6 neighbor open flags (1=open, 0=wall/out)
 * 12..14 : direction-to-goal sign on x/y/z (-1,0,1)
 * 15     : steps normalized [0,1]
 * 16     : manhattan distance normalized [0,1]
 * 17     : lastAction normalized [0,1]
 */
export function obs18(ep: Episode): Float32Array {
  const [sx, sy, sz] = ep.size
  const gx = ep.goal[0] - ep.pos[0]
  const gy = ep.goal[1] - ep.pos[1]
  const gz = ep.goal[2] - ep.pos[2]

  const dx = sx > 1 ? (2 * gx) / (sx - 1) : 0
  const dy = sy > 1 ? (2 * gy) / (sy - 1) : 0
  const dz = sz > 1 ? (2 * gz) / (sz - 1) : 0

  const px = sx > 1 ? ep.pos[0] / (sx - 1) : 0
  const py = sy > 1 ? ep.pos[1] / (sy - 1) : 0
  const pz = sz > 1 ? ep.pos[2] / (sz - 1) : 0

  const dirs: Vec3[] = [
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
  ]

  const neighOpen: number[] = dirs.map(([ax, ay, az]) => {
    const nx = ep.pos[0] + ax
    const ny = ep.pos[1] + ay
    const nz = ep.pos[2] + az
    if (!inBounds(ep.size, nx, ny, nz)) return 0
    return ep.grid[idx(ep.size, nx, ny, nz)] === 0 ? 1 : 0
  })

  const sgn = (v: number) => (v > 0 ? 1 : v < 0 ? -1 : 0)

  const maxSteps = ep.size[0] * ep.size[1] * ep.size[2] * 4
  const stepsNorm = maxSteps > 0 ? clamp01(ep.steps / maxSteps) : 0

  const maxMan = (sx - 1) + (sy - 1) + (sz - 1)
  const manNorm = maxMan > 0 ? clamp01(manhattan(ep.pos, ep.goal) / maxMan) : 0

  const lastActNorm = clamp01(ep.lastAction / 5)

  return new Float32Array([
    dx,
    dy,
    dz,
    px,
    py,
    pz,
    ...neighOpen, // 6 values -> indices 6..11
    sgn(gx),
    sgn(gy),
    sgn(gz),
    stepsNorm,
    manNorm,
    lastActNorm,
  ])
}