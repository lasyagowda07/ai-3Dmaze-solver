"use client";

import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Line } from "@react-three/drei";
import { useMemo, useRef } from "react";
import * as THREE from "three";
import type { MazeState, Vec3 } from "../lib/maze3d";

export type CameraMode = "manual" | "follow" | "first" | "top" | "iso";

export default function MazeScene({ state, cameraMode }: { state: MazeState; cameraMode: CameraMode }) {
  return (
    <Canvas camera={{ position: [18, 14, 18], fov: 55, near: 0.1, far: 260 }} dpr={[1, 2]}>
      <color attach="background" args={["#07070a"]} />
      <fog attach="fog" args={["#07070a", 25, 120]} />

      <ambientLight intensity={0.65} />
      <directionalLight position={[12, 18, 10]} intensity={0.95} />
      <pointLight position={[-14, -10, -12]} intensity={0.35} />

      <SceneContent state={state} />
      <CameraRig state={state} mode={cameraMode} />

      {cameraMode === "manual" ? (
        <OrbitControls enablePan={false} minDistance={10} maxDistance={110} target={[0, 0, 0]} />
      ) : null}
    </Canvas>
  );
}

function CameraRig({ state, mode }: { state: MazeState; mode: CameraMode }) {
  const { camera } = useThree();
  const [sx, sy, sz] = state.size;

  const toWorld = (p: Vec3) =>
    new THREE.Vector3(p[0] - (sx - 1) / 2, p[1] - (sy - 1) / 2, p[2] - (sz - 1) / 2);

  useFrame(() => {
    const head = toWorld(state.agent);
    const forward = new THREE.Vector3(...state.lastMove);
    if (forward.lengthSq() === 0) forward.set(1, 0, 0);
    forward.normalize();

    const up = new THREE.Vector3(0, 1, 0);
    let right = new THREE.Vector3().crossVectors(forward, up);
    if (right.lengthSq() === 0) right = new THREE.Vector3(0, 0, 1);
    right.normalize();

    if (mode === "top") {
      const pos = head.clone().add(new THREE.Vector3(0, Math.max(sy, 10) * 1.8, 0));
      camera.position.lerp(pos, 0.12);
      camera.lookAt(head.x, head.y, head.z);
      return;
    }

    if (mode === "iso") {
      const d = Math.max(sx, sy, sz) * 1.85;
      const pos = new THREE.Vector3(d, d * 0.95, d);
      camera.position.lerp(pos, 0.08);
      camera.lookAt(0, 0, 0);
      return;
    }

    if (mode === "first") {
      const pos = head.clone().add(up.clone().multiplyScalar(0.18)).add(forward.clone().multiplyScalar(0.18));
      const aim = head.clone().add(forward.clone().multiplyScalar(3.4));
      camera.position.lerp(pos, 0.25);
      camera.lookAt(aim.x, aim.y, aim.z);
      return;
    }

    if (mode === "follow") {
      const pos = head
        .clone()
        .add(forward.clone().multiplyScalar(-6.2))
        .add(up.clone().multiplyScalar(3.0))
        .add(right.clone().multiplyScalar(0.6));

      camera.position.lerp(pos, 0.10);
      camera.lookAt(head.x, head.y, head.z);
      return;
    }
  });

  return null;
}

function SceneContent({ state }: { state: MazeState }) {
  const [sx, sy, sz] = state.size;

  const toWorld = (p: Vec3) =>
    new THREE.Vector3(p[0] - (sx - 1) / 2, p[1] - (sy - 1) / 2, p[2] - (sz - 1) / 2);

  const boundaryWire = useMemo(
    () => new THREE.MeshBasicMaterial({ color: "#ffffff", wireframe: true, transparent: true, opacity: 0.18 }),
    []
  );
  const boundarySolid = useMemo(
    () => new THREE.MeshStandardMaterial({ color: "#0d0d16", transparent: true, opacity: 0.10, roughness: 1.0 }),
    []
  );

  const wallGeom = useMemo(() => new THREE.BoxGeometry(0.98, 0.98, 0.98), []);
  const wallMat = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: "#1b1b2a",
        roughness: 0.9,
        metalness: 0.0,
        emissive: "#0b0b12",
        emissiveIntensity: 0.05,
      }),
    []
  );

  const agentGeom = useMemo(() => new THREE.SphereGeometry(0.45, 28, 28), []);
  const agentMat = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: "#5DEBFF",
        emissive: "#5DEBFF",
        emissiveIntensity: 0.55,
        roughness: 0.18,
        metalness: 0.08,
      }),
    []
  );

  const goalGeom = useMemo(() => new THREE.SphereGeometry(0.42, 26, 26), []);
  const goalMat = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: "#ff6b9a",
        emissive: "#ff6b9a",
        emissiveIntensity: 0.55,
        roughness: 0.25,
        metalness: 0.05,
      }),
    []
  );

  // Build wall instances positions
  const wallPositions = useMemo(() => {
    const out: THREE.Vector3[] = [];
    const grid = state.grid;

    const idx = (x: number, y: number, z: number) => x + sx * (y + sy * z);

    for (let x = 0; x < sx; x++) for (let y = 0; y < sy; y++) for (let z = 0; z < sz; z++) {
      if (grid[idx(x, y, z)] === 1) {
        const p = toWorld([x, y, z]);
        out.push(p);
      }
    }
    return out;
  }, [state.grid, sx, sy, sz]);

  const wallsRef = useRef<THREE.InstancedMesh>(null);

  // Write instance matrices once (since maze is fixed until reset)
  useMemo(() => {
    if (!wallsRef.current) return;
    const mesh = wallsRef.current;
    const m = new THREE.Matrix4();

    for (let i = 0; i < wallPositions.length; i++) {
      const p = wallPositions[i];
      m.makeTranslation(p.x, p.y, p.z);
      mesh.setMatrixAt(i, m);
    }
    mesh.instanceMatrix.needsUpdate = TrueFix();
  }, [wallPositions]);

  const agentPos = toWorld(state.agent);
  const goalPos = toWorld(state.goal);

  // Grid planes on 3 faces for depth
  const gridLinesXY = useMemo(() => makeGridLinesXY(sx, sy, sz), [sx, sy, sz]);
  const gridLinesXZ = useMemo(() => makeGridLinesXZ(sx, sy, sz), [sx, sy, sz]);
  const gridLinesYZ = useMemo(() => makeGridLinesYZ(sx, sy, sz), [sx, sy, sz]);

  return (
    <>
      {/* Boundary cube */}
      <mesh>
        <boxGeometry args={[sx, sy, sz]} />
        <primitive object={boundarySolid} attach="material" />
      </mesh>
      <mesh>
        <boxGeometry args={[sx, sy, sz]} />
        <primitive object={boundaryWire} attach="material" />
      </mesh>

      {/* Grid planes */}
      <GridLines lines={gridLinesXY} opacity={0.08} />
      <GridLines lines={gridLinesXZ} opacity={0.08} />
      <GridLines lines={gridLinesYZ} opacity={0.08} />

      {/* Walls */}
      <instancedMesh ref={wallsRef} args={[wallGeom, wallMat, wallPositions.length]} />

      {/* Agent + Goal */}
      <mesh geometry={agentGeom} material={agentMat} position={[agentPos.x, agentPos.y, agentPos.z]} />
      <mesh geometry={goalGeom} material={goalMat} position={[goalPos.x, goalPos.y, goalPos.z]} />
    </>
  );
}

// small helper because TS/JS sometimes hates setting boolean on instanceMatrix in memo
function TrueFix() {
  return true;
}

function GridLines({ lines, opacity }: { lines: Array<[THREE.Vector3, THREE.Vector3]>; opacity: number }) {
  return (
    <>
      {lines.map(([a, b], i) => (
        <Line key={i} points={[a, b]} color="#ffffff" transparent opacity={opacity} lineWidth={1} />
      ))}
    </>
  );
}

function makeGridLinesXY(sx: number, sy: number, sz: number) {
  const z = -((sz - 1) / 2);
  const lines: Array<[THREE.Vector3, THREE.Vector3]> = [];
  for (let x = 0; x < sx; x++) {
    const X = x - (sx - 1) / 2;
    lines.push([new THREE.Vector3(X, -(sy - 1) / 2, z), new THREE.Vector3(X, (sy - 1) / 2, z)]);
  }
  for (let y = 0; y < sy; y++) {
    const Y = y - (sy - 1) / 2;
    lines.push([new THREE.Vector3(-(sx - 1) / 2, Y, z), new THREE.Vector3((sx - 1) / 2, Y, z)]);
  }
  return lines;
}
function makeGridLinesXZ(sx: number, sy: number, sz: number) {
  const y = -((sy - 1) / 2);
  const lines: Array<[THREE.Vector3, THREE.Vector3]> = [];
  for (let x = 0; x < sx; x++) {
    const X = x - (sx - 1) / 2;
    lines.push([new THREE.Vector3(X, y, -(sz - 1) / 2), new THREE.Vector3(X, y, (sz - 1) / 2)]);
  }
  for (let z = 0; z < sz; z++) {
    const Z = z - (sz - 1) / 2;
    lines.push([new THREE.Vector3(-(sx - 1) / 2, y, Z), new THREE.Vector3((sx - 1) / 2, y, Z)]);
  }
  return lines;
}
function makeGridLinesYZ(sx: number, sy: number, sz: number) {
  const x = -((sx - 1) / 2);
  const lines: Array<[THREE.Vector3, THREE.Vector3]> = [];
  for (let y = 0; y < sy; y++) {
    const Y = y - (sy - 1) / 2;
    lines.push([new THREE.Vector3(x, Y, -(sz - 1) / 2), new THREE.Vector3(x, Y, (sz - 1) / 2)]);
  }
  for (let z = 0; z < sz; z++) {
    const Z = z - (sz - 1) / 2;
    lines.push([new THREE.Vector3(x, -(sy - 1) / 2, Z), new THREE.Vector3(x, (sy - 1) / 2, Z)]);
  }
  return lines;
}