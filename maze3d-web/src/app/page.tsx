import MazeScene from "../components/MazeScene";

export default function Page() {
  return (
    <div className="min-h-screen bg-black text-white">
      <header className="px-6 pt-8 pb-4">
        <h1 className="text-3xl font-semibold tracking-tight">3D Maze AI (Live)</h1>
        <p className="mt-2 text-sm text-white/60">
          ONNX model runs fully in your browser. Refresh to regenerate a new maze.
        </p>
      </header>

      <main className="px-6 pb-10">
        <div className="h-[78vh] rounded-2xl overflow-hidden border border-white/10 bg-white/5">
          <MazeScene />
        </div>
      </main>
    </div>
  );
}