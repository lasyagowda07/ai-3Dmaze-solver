"use client";
import React from "react";
import { useEffect, useRef, useState } from "react";
import MazeScene, { type CameraMode } from "../components/MazeScene";
import { loadPolicy, chooseAction } from "../lib/onnxPolicy";
import { obs18, resetMaze, stepMaze, type MazeState } from "../lib/maze3d";

export default function Page() {
  const [mode, setMode] = useState<CameraMode>("follow");
  const [state, setState] = useState<MazeState>(() => resetMaze([11, 11, 5], 400));
  const sessionRef = useRef<any>(null);
  const runningRef = useRef(true);
  const stateRef = useRef(state);

  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  // load ONNX model
  useEffect(() => {
    (async () => {
      sessionRef.current = await loadPolicy("/maze_dqn3d.onnx");
    })();
    return () => {
      runningRef.current = false;
    };
  }, []);

  // main loop
  useEffect(() => {
    let alive = true;

    const loop = async () => {
      while (alive && runningRef.current) {
        const session = sessionRef.current;
        if (!session) {
          await sleep(50);
          continue;
        }

        let s = stateRef.current;

        // reset on done
        if (s.done) {
          s = resetMaze(s.size, s.maxSteps);
        }

        const obs = obs18(s);
        const action = await chooseAction(session, obs);
        const next = stepMaze(s, action);

        stateRef.current = next.done ? resetMaze(next.size, next.maxSteps) : next;
        setState(stateRef.current);

        await sleep(80); // ~12.5 FPS
      }
    };

    loop();
    return () => {
      alive = false;
    };
  }, []);

  return (
    <div style={{ maxWidth: 1180, margin: "0 auto", padding: "26px 18px 38px", color: "#f3f3f6" }}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 16, flexWrap: "wrap", alignItems: "end" }}>
        <div>
          <h1 style={{ margin: 0, fontSize: 34, lineHeight: 1.05, letterSpacing: "-0.02em" }}>
            3D Maze Solver <span style={{ opacity: 0.7 }}>(Browser AI)</span>
          </h1>
          <p style={{ marginTop: 10, opacity: 0.75, fontSize: 13 }}>
            A reinforcement learning agent navigating procedurally generated 3D mazes.
          </p>
          <p style={{ marginTop: 6, opacity: 0.6, fontSize: 12 }}>
            Runs fully client-side using ONNX + onnxruntime-web. No backend.
          </p>
        </div>

        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          <Pill label={`steps: ${state.steps}/${state.maxSteps}`} />
          <Pill label={`size: ${state.size[0]}×${state.size[1]}×${state.size[2]}`} />
        </div>
      </div>

      <div style={{ marginTop: 14, display: "flex", gap: 10, flexWrap: "wrap" }}>
        <ModeButton label="manual" active={mode === "manual"} onClick={() => setMode("manual")} />
        <ModeButton label="follow" active={mode === "follow"} onClick={() => setMode("follow")} />
        <ModeButton label="first person" active={mode === "first"} onClick={() => setMode("first")} />
        <ModeButton label="top-down" active={mode === "top"} onClick={() => setMode("top")} />
        <ModeButton label="isometric" active={mode === "iso"} onClick={() => setMode("iso")} />
        <ModeButton label="new maze" active={false} onClick={() => setState(resetMaze(state.size, state.maxSteps))} />
      </div>

      <div
        style={{
          marginTop: 18,
          height: "74vh",
          minHeight: 520,
          borderRadius: 22,
          overflow: "hidden",
          background: "rgba(10,10,14,0.55)",
          border: "1px solid rgba(255,255,255,0.07)",
          boxShadow: "0 30px 90px rgba(0,0,0,0.55)",
        }}
      >
        <MazeScene state={state} cameraMode={mode} />
      </div>
    </div>
  );
}

function Pill({ label }: { label: string }) {
  return (
    <div
      style={{
        padding: "8px 12px",
        borderRadius: 999,
        background: "rgba(18,18,28,0.62)",
        border: "1px solid rgba(255,255,255,0.08)",
        boxShadow: "0 10px 30px rgba(0,0,0,0.35)",
        fontSize: 13,
      }}
    >
      {label}
    </div>
  );
}

function ModeButton({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: "10px 14px",
        borderRadius: 999,
        border: active ? "1px solid rgba(93,235,255,0.55)" : "1px solid rgba(255,255,255,0.10)",
        background: active ? "rgba(93,235,255,0.10)" : "rgba(18,18,28,0.60)",
        color: "rgba(243,243,246,0.92)",
        boxShadow: active ? "0 10px 30px rgba(93,235,255,0.10)" : "0 10px 30px rgba(0,0,0,0.35)",
        cursor: "pointer",
        fontSize: 13,
        backdropFilter: "blur(12px)",
      }}
    >
      {label}
    </button>
  );
}

function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}