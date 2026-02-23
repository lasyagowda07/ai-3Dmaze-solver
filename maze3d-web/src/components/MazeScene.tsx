// src/components/MazeScene.tsx
"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Grid } from "@react-three/drei";

import { createEpisode, obs18, stepEpisode, type Episode } from "../lib/maze3d";
import { loadPolicy, chooseAction } from "../lib/onnxPolicy";

const COLORS = {
  wall: new THREE.Color("#f6b7d2"), // pastel pink
  path: new THREE.Color("#ffeaa7"), // translucent yellow
  floor: new THREE.Color("#0e0e0f"),
  goal: new THREE.Color("#ffd6a5"),
  agent: new THREE.Color("#9bf6ff"),
};

type CameraMode = "orbit" | "follow" | "first";

// helpers (idx/inBounds not exported)
function idx3(size: [number, number, number], x: number, y: number, z: number) {
  const [sx, sy] = size;
  return x + sx * (y + sy * z);
}
function inBounds3(size: [number, number, number], x: number, y: number, z: number) {
  const [sx, sy, sz] = size;
  return x >= 0 && y >= 0 && z >= 0 && x < sx && y < sy && z < sz;
}

function solvePathBFS(ep: Episode): Array<[number, number, number]> {
  const [sx, sy, sz] = ep.size;
  const total = sx * sy * sz;

  const start = ep.start;
  const goal = ep.goal;

  const startIdx = idx3(ep.size, start[0], start[1], start[2]);
  const goalIdx = idx3(ep.size, goal[0], goal[1], goal[2]);

  const prev = new Int32Array(total);
  prev.fill(-1);

  const q = new Int32Array(total);
  let qh = 0,
    qt = 0;

  if (ep.grid[startIdx] !== 0 || ep.grid[goalIdx] !== 0) return [];

  prev[startIdx] = startIdx;
  q[qt++] = startIdx;

  const dirs: Array<[number, number, number]> = [
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
  ];

  while (qh < qt) {
    const cur = q[qh++];
    if (cur === goalIdx) break;

    const x = cur % sx;
    const y = Math.floor(cur / sx) % sy;
    const z = Math.floor(cur / (sx * sy));

    for (const [dx, dy, dz] of dirs) {
      const nx = x + dx;
      const ny = y + dy;
      const nz = z + dz;
      if (!inBounds3(ep.size, nx, ny, nz)) continue;

      const ni = idx3(ep.size, nx, ny, nz);
      if (prev[ni] !== -1) continue;
      if (ep.grid[ni] !== 0) continue;

      prev[ni] = cur;
      q[qt++] = ni;
    }
  }

  if (prev[goalIdx] === -1) return [];

  const pathIdx: number[] = [];
  let cur = goalIdx;
  while (true) {
    pathIdx.push(cur);
    if (cur === startIdx) break;
    cur = prev[cur];
    if (cur === -1) return [];
  }
  pathIdx.reverse();

  return pathIdx.map((id) => {
    const x = id % sx;
    const y = Math.floor(id / sx) % sy;
    const z = Math.floor(id / (sx * sy));
    return [x, y, z];
  });
}

function actionFromStep(a: [number, number, number], b: [number, number, number]): number {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const dz = b[2] - a[2];
  if (dx === 1) return 0;
  if (dx === -1) return 1;
  if (dy === 1) return 2;
  if (dy === -1) return 3;
  if (dz === 1) return 4;
  if (dz === -1) return 5;
  return 0;
}

function SceneInner({ sizeHint, cameraMode }: { sizeHint: number; cameraMode: CameraMode }) {
  const { camera } = useThree();

  const [status, setStatus] = useState<"loading" | "running" | "error">("loading");
  const [errorMsg, setErrorMsg] = useState("");
  const [tick, setTick] = useState(0);

  const sessionRef = useRef<any>(null);
  const epRef = useRef<Episode | null>(null);

  const stepAccRef = useRef(0);
  const inFlightRef = useRef(false);

  // manual action (world axes)
  const manualActionRef = useRef<number | null>(null);

  // NEW: path-follow toggle
  const autoPathRef = useRef(true); // default ON so it always moves
  const [autoPathUI, setAutoPathUI] = useState(true);

  // NEW: solved path + pointer
  const solvedPathRef = useRef<Array<[number, number, number]>>([]);
  const pathIndexRef = useRef(0);

  // path overlay (world positions)
  const [pathWorld, setPathWorld] = useState<THREE.Vector3[]>([]);

  const resetEpisode = () => {
    const ep = createEpisode({ sx: sizeHint, sy: sizeHint, sz: sizeHint });
    epRef.current = ep;

    const solved = solvePathBFS(ep);
    solvedPathRef.current = solved;
    pathIndexRef.current = 0;

    const cx = (ep.size[0] - 1) / 2;
    const cy = (ep.size[1] - 1) / 2;
    const cz = (ep.size[2] - 1) / 2;

    const mid = solved.slice(1, Math.max(1, solved.length - 1));
    setPathWorld(mid.map(([x, y, z]) => new THREE.Vector3(x - cx, y - cy, z - cz)));

    setTick((t) => t + 1);
  };

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        setStatus("loading");
        setErrorMsg("");
        const session = await loadPolicy("/maze_dqn3d.onnx");
        if (!alive) return;

        sessionRef.current = session;
        resetEpisode();
        setStatus("running");
      } catch (e: any) {
        if (!alive) return;
        console.error(e);
        setStatus("error");
        setErrorMsg(e?.message ?? String(e));
      }
    })();
    return () => {
      alive = false;
    };
  }, [sizeHint]);

  useEffect(() => {
    const onDown = (e: KeyboardEvent) => {
      // manual control
      if (e.key === "d" || e.key === "D") manualActionRef.current = 0;
      if (e.key === "a" || e.key === "A") manualActionRef.current = 1;
      if (e.key === "w" || e.key === "W") manualActionRef.current = 2;
      if (e.key === "s" || e.key === "S") manualActionRef.current = 3;
      if (e.key === "r" || e.key === "R") manualActionRef.current = 4;
      if (e.key === "f" || e.key === "F") manualActionRef.current = 5;
      if (e.key === " ") manualActionRef.current = null; // return to auto mode

      // NEW: toggle path-follow
      if (e.key === "p" || e.key === "P") {
        autoPathRef.current = !autoPathRef.current;
        setAutoPathUI(autoPathRef.current);
      }
    };
    window.addEventListener("keydown", onDown);
    return () => window.removeEventListener("keydown", onDown);
  }, []);

  const ep = epRef.current;
  const actualSize = (ep?.size ?? [sizeHint, sizeHint, sizeHint]) as [number, number, number];
  const [sx, sy, sz] = actualSize;

  const center = useMemo(() => new THREE.Vector3((sx - 1) / 2, (sy - 1) / 2, (sz - 1) / 2), [sx, sy, sz]);

  const wallPositions = useMemo(() => {
    const e = epRef.current;
    if (!e) return [] as THREE.Vector3[];

    const out: THREE.Vector3[] = [];
    const cx = (e.size[0] - 1) / 2;
    const cy = (e.size[1] - 1) / 2;
    const cz = (e.size[2] - 1) / 2;

    const [gx, gy, gz] = e.size;
    for (let z = 0; z < gz; z++) {
      for (let y = 0; y < gy; y++) {
        for (let x = 0; x < gx; x++) {
          const id = idx3(e.size, x, y, z);
          if (e.grid[id] === 1) out.push(new THREE.Vector3(x - cx, y - cy, z - cz));
        }
      }
    }
    return out;
  }, [tick]);

  const wallInstRef = useRef<THREE.InstancedMesh>(null);
  const pathInstRef = useRef<THREE.InstancedMesh>(null);

  useEffect(() => {
    const inst = wallInstRef.current;
    if (!inst) return;
    const dummy = new THREE.Object3D();
    for (let i = 0; i < wallPositions.length; i++) {
      dummy.position.copy(wallPositions[i]);
      dummy.updateMatrix();
      inst.setMatrixAt(i, dummy.matrix);
    }
    inst.count = wallPositions.length;
    inst.instanceMatrix.needsUpdate = true;
  }, [wallPositions]);

  useEffect(() => {
    const inst = pathInstRef.current;
    if (!inst) return;
    const dummy = new THREE.Object3D();
    for (let i = 0; i < pathWorld.length; i++) {
      dummy.position.copy(pathWorld[i]);
      dummy.updateMatrix();
      inst.setMatrixAt(i, dummy.matrix);
    }
    inst.count = pathWorld.length;
    inst.instanceMatrix.needsUpdate = true;
  }, [pathWorld]);

  useFrame((_, delta) => {
    if (status !== "running") return;
    if (!epRef.current) return;

    stepAccRef.current += delta;
    if (stepAccRef.current < 0.12) return;
    stepAccRef.current = 0;

    if (inFlightRef.current) return;
    inFlightRef.current = true;

    (async () => {
      try {
        const e = epRef.current!;

        let action: number;

        // 1) manual always wins if set
        if (manualActionRef.current != null) {
          action = manualActionRef.current;
        } else if (autoPathRef.current && solvedPathRef.current.length > 1) {
          // 2) NEW: follow solved BFS path (never gets stuck)
          const path = solvedPathRef.current;

          // move pointer forward if we already reached that node
          while (
            pathIndexRef.current < path.length &&
            path[pathIndexRef.current][0] === e.pos[0] &&
            path[pathIndexRef.current][1] === e.pos[1] &&
            path[pathIndexRef.current][2] === e.pos[2]
          ) {
            pathIndexRef.current += 1;
          }

          // if we still have next step
          if (pathIndexRef.current < path.length) {
            const cur: [number, number, number] = [e.pos[0], e.pos[1], e.pos[2]];
            const next = path[pathIndexRef.current];
            action = actionFromStep(cur, next);
          } else {
            // fallback to AI when path finishes
            const obs = obs18(e);
            action = await chooseAction(sessionRef.current, obs);
          }
        } else {
          // 3) AI mode
          const obs = obs18(e);
          action = await chooseAction(sessionRef.current, obs);
        }

        stepEpisode(e, action);
        setTick((t) => t + 1);

        const agent = new THREE.Vector3(e.pos[0] - center.x, e.pos[1] - center.y, e.pos[2] - center.z);
        const goal = new THREE.Vector3(e.goal[0] - center.x, e.goal[1] - center.y, e.goal[2] - center.z);

        if (cameraMode === "follow") {
          const desired = agent.clone().add(new THREE.Vector3(3, 3, 3));
          camera.position.lerp(desired, 0.12);
          camera.lookAt(agent);
        } else if (cameraMode === "first") {
          const desired = agent.clone().add(new THREE.Vector3(0.3, 0.3, 0.3));
          camera.position.lerp(desired, 0.25);
          camera.lookAt(goal);
        }

        if (e.done) {
          setTimeout(() => resetEpisode(), 600);
        }
      } catch (err: any) {
        console.error("Step error:", err);
        setStatus("error");
        setErrorMsg(err?.message ?? String(err));
      } finally {
        inFlightRef.current = false;
      }
    })();
  });

  const pos = ep?.pos ?? ([1, 1, 1] as [number, number, number]);
  const goal = ep?.goal ?? ([sx - 2, sy - 2, sz - 2] as [number, number, number]);

  const agentCenter: [number, number, number] = [pos[0] - center.x, pos[1] - center.y, pos[2] - center.z];
  const goalCenter: [number, number, number] = [goal[0] - center.x, goal[1] - center.y, goal[2] - center.z];

  const floorY = -sy / 2 - 0.6;

  if (status === "error") {
    console.warn("[scene error]", errorMsg);
    return (
      <>
        <ambientLight intensity={0.7} />
        <directionalLight position={[10, 15, 8]} intensity={1.25} />
        <mesh>
          <boxGeometry args={[1, 1, 1]} />
          <meshStandardMaterial color={"#ff5555"} />
        </mesh>
      </>
    );
  }

  return (
    <>
      <ambientLight intensity={0.85} />
      <directionalLight position={[10, 15, 8]} intensity={1.35} />

      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, floorY, 0]}>
        <planeGeometry args={[80, 80]} />
        <meshStandardMaterial color={COLORS.floor} />
      </mesh>

      <Grid
        position={[0, floorY + 0.01, 0]}
        args={[60, 60]}
        cellSize={1}
        cellThickness={1}
        sectionSize={5}
        sectionThickness={1.5}
        fadeDistance={45}
        fadeStrength={1}
        infiniteGrid={false}
      />

      {/* PATH overlay */}
      <instancedMesh ref={pathInstRef} args={[undefined as any, undefined as any, Math.max(1, pathWorld.length)]}>
        <boxGeometry args={[0.62, 0.62, 0.62]} />
        <meshStandardMaterial
          color={COLORS.path}
          transparent
          opacity={0.34}
          roughness={0.2}
          metalness={0}
          depthWrite={false}
          emissive={COLORS.path}
          emissiveIntensity={0.2}
        />
      </instancedMesh>

      {/* WALLS */}
      <instancedMesh ref={wallInstRef} args={[undefined as any, undefined as any, Math.max(1, wallPositions.length)]}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial
          color={COLORS.wall}
          transparent
          opacity={0.25}
          roughness={0.25}
          metalness={0}
          depthWrite={false}
        />
      </instancedMesh>

      {/* Goal */}
      <mesh position={goalCenter}>
        <sphereGeometry args={[0.38, 20, 20]} />
        <meshStandardMaterial color={COLORS.goal} emissive={COLORS.goal} emissiveIntensity={0.6} />
      </mesh>

      {/* Agent */}
      <mesh position={agentCenter}>
        <sphereGeometry args={[0.34, 20, 20]} />
        <meshStandardMaterial color={COLORS.agent} roughness={0.25} metalness={0.1} />
      </mesh>

      {cameraMode === "orbit" ? <OrbitControls enableDamping dampingFactor={0.08} /> : null}
    </>
  );
}

export default function MazeScene() {
  const size = 10;
  const [cameraMode, setCameraMode] = useState<CameraMode>("orbit");
  const [autoPathUI, setAutoPathUI] = useState(true);

  // sync label quickly (simple approach)
  useEffect(() => {
    const onDown = (e: KeyboardEvent) => {
      if (e.key === "p" || e.key === "P") setAutoPathUI((v) => !v);
    };
    window.addEventListener("keydown", onDown);
    return () => window.removeEventListener("keydown", onDown);
  }, []);

  return (
    <div className="relative h-full w-full">
      <div className="absolute left-4 top-4 z-10 rounded-full border border-white/10 bg-black/50 px-3 py-1 text-[11px] text-white/80">
        Path-Follow: <span className="text-white">{autoPathUI ? "ON" : "OFF"}</span> (press P)
      </div>

      <div className="absolute right-4 top-4 z-10 flex items-center gap-2">
        <div className="flex overflow-hidden rounded-full border border-white/10 bg-black/50 backdrop-blur">
          {(["orbit", "follow", "first"] as const).map((m) => (
            <button
              key={m}
              onClick={() => setCameraMode(m)}
              className={[
                "px-3 py-1 text-xs",
                cameraMode === m ? "bg-white/10 text-white" : "text-white/70 hover:bg-white/5",
              ].join(" ")}
            >
              {m}
            </button>
          ))}
        </div>

        <div className="hidden md:block rounded-full border border-white/10 bg-black/50 px-3 py-1 text-[11px] text-white/70">
          Manual: WASD (XY) + R/F (Z) • Space = Auto
        </div>
      </div>

      <Canvas camera={{ position: [8, 8, 8], fov: 45 }} dpr={[1, 2]}>
        <color attach="background" args={["#070709"]} />
        <SceneInner sizeHint={size} cameraMode={cameraMode} />
      </Canvas>
    </div>
  );
}