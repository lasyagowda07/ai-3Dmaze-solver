export type Vec3 = [number, number, number];

export const DIRS: Vec3[] = [
  [ 1, 0, 0], // 0 +X
  [-1, 0, 0], // 1 -X
  [ 0, 1, 0], // 2 +Y
  [ 0,-1, 0], // 3 -Y
  [ 0, 0, 1], // 4 +Z
  [ 0, 0,-1], // 5 -Z
];

export type MazeState = {
  size: Vec3;
  grid: Uint8Array;     // 1 wall, 0 free (flattened)
  agent: Vec3;
  goal: Vec3;
  steps: number;
  maxSteps: number;
  lastMove: Vec3;       // dx,dy,dz
  done: boolean;
};

function idx(size: Vec3, x: number, y: number, z: number) {
  const [sx, sy] = size;
  return x + sx * (y + sy * z);
}

function inBounds(size: Vec3, p: Vec3) {
  const [sx, sy, sz] = size;
  return p[0] >= 0 && p[0] < sx && p[1] >= 0 && p[1] < sy && p[2] >= 0 && p[2] < sz;
}

function isWall(s: MazeState, p: Vec3) {
  if (!inBounds(s.size, p)) return true;
  const i = idx(s.size, p[0], p[1], p[2]);
  return s.grid[i] === 1;
}

function manhattan(a: Vec3, b: Vec3) {
  return Math.abs(a[0]-b[0]) + Math.abs(a[1]-b[1]) + Math.abs(a[2]-b[2]);
}

/**
 * Quick browser maze generator: we keep it simple:
 * - Start with all walls
 * - Carve random DFS on odd cells (same spirit as Python)
 * Note: sizes should be odd for best results.
 */
export function generateMaze(size: Vec3, rng = Math.random) {
  let [sx, sy, sz] = size;
  if (sx % 2 === 0) sx += 1;
  if (sy % 2 === 0) sy += 1;
  if (sz % 2 === 0) sz += 1;
  const fixed: Vec3 = [sx, sy, sz];

  const grid = new Uint8Array(sx * sy * sz);
  grid.fill(1);

  const start: Vec3 = [1, 1, 1];
  const stack: Vec3[] = [start];
  const visited = new Set<string>([key(start)]);
  grid[idx(fixed, 1, 1, 1)] = 0;

  const neighbors2 = (p: Vec3) => {
    const [x,y,z] = p;
    const out: Vec3[] = [];
    const cands: Vec3[] = [
      [x+2,y,z],[x-2,y,z],[x,y+2,z],[x,y-2,z],[x,y,z+2],[x,y,z-2],
    ];
    for (const q of cands) {
      if (q[0] >= 1 && q[0] < sx-1 && q[1] >= 1 && q[1] < sy-1 && q[2] >= 1 && q[2] < sz-1) out.push(q);
    }
    return out;
  };

  while (stack.length) {
    const cur = stack[stack.length - 1];
    const nbrs = neighbors2(cur);

    // shuffle
    for (let i = nbrs.length - 1; i > 0; i--) {
      const j = (rng() * (i + 1)) | 0;
      const tmp = nbrs[i]; nbrs[i] = nbrs[j]; nbrs[j] = tmp;
    }

    let moved = false;
    for (const n of nbrs) {
      const k = key(n);
      if (visited.has(k)) continue;

      const wx = (cur[0] + n[0]) >> 1;
      const wy = (cur[1] + n[1]) >> 1;
      const wz = (cur[2] + n[2]) >> 1;

      grid[idx(fixed, wx, wy, wz)] = 0;
      grid[idx(fixed, n[0], n[1], n[2])] = 0;

      visited.add(k);
      stack.push(n);
      moved = true;
      break;
    }
    if (!moved) stack.pop();
  }

  // choose far goal
  let goal: Vec3 = start;
  let best = -1;
  for (let x = 1; x < sx - 1; x++) for (let y = 1; y < sy - 1; y++) for (let z = 1; z < sz - 1; z++) {
    if (grid[idx(fixed, x,y,z)] === 0) {
      const d = Math.abs(x-1)+Math.abs(y-1)+Math.abs(z-1);
      if (d > best) { best = d; goal = [x,y,z]; }
    }
  }

  return { size: fixed, grid, start, goal };
}

function key(p: Vec3) {
  return `${p[0]},${p[1]},${p[2]}`;
}

export function resetMaze(size: Vec3, maxSteps = 400): MazeState {
  const { size: fixed, grid, start, goal } = generateMaze(size);
  return {
    size: fixed,
    grid,
    agent: start,
    goal,
    steps: 0,
    maxSteps,
    lastMove: [0,0,0],
    done: false,
  };
}

export function stepMaze(s: MazeState, action: number): MazeState {
  if (s.done) return s;

  const next: MazeState = { ...s, steps: s.steps + 1 };

  let reward = -0.01;
  const prevDist = manhattan(s.agent, s.goal);

  action = Math.max(0, Math.min(5, action | 0));
  const [dx,dy,dz] = DIRS[action];
  const cand: Vec3 = [s.agent[0]+dx, s.agent[1]+dy, s.agent[2]+dz];

  if (isWall(s, cand)) {
    reward -= 0.20;
    next.lastMove = [0,0,0];
    next.agent = s.agent;
  } else {
    next.agent = cand;
    next.lastMove = [dx,dy,dz];
  }

  const newDist = manhattan(next.agent, next.goal);
  reward += (prevDist - newDist) * 0.10;

  if (next.agent[0] === next.goal[0] && next.agent[1] === next.goal[1] && next.agent[2] === next.goal[2]) {
    reward += 10.0;
    next.done = true;
  }
  if (next.steps >= next.maxSteps) next.done = true;

  // attach reward for UI/debug if you want:
  // (optional) (next as any).reward = reward;

  return next;
}

export function obs18(s: MazeState): Float32Array {
  const [ax,ay,az] = s.agent;
  const [gx,gy,gz] = s.goal;
  const [sx,sy,sz] = s.size;

  const danger = DIRS.map(([dx,dy,dz]) => {
    const p: Vec3 = [ax+dx, ay+dy, az+dz];
    return isWall(s, p) ? 1 : 0;
  });

  const goalRel = [
    gx > ax ? 1 : 0,
    gx < ax ? 1 : 0,
    gy > ay ? 1 : 0,
    gy < ay ? 1 : 0,
    gz > az ? 1 : 0,
    gz < az ? 1 : 0,
  ];

  const dxn = (gx - ax) / Math.max(1, sx - 1);
  const dyn = (gy - ay) / Math.max(1, sy - 1);
  const dzn = (gz - az) / Math.max(1, sz - 1);

  const [ldx,ldy,ldz] = s.lastMove;

  const obs = new Float32Array([
    ...danger,
    ...goalRel,
    dxn, dyn, dzn,
    ldx, ldy, ldz,
  ]);
  return obs;
}