import * as ort from "onnxruntime-web";

export async function loadPolicy(modelUrl = "/maze_dqn3d.onnx") {
  const session = await ort.InferenceSession.create(modelUrl, {
    executionProviders: ["wasm"],
  });
  return session;
}

export async function chooseAction(session: ort.InferenceSession, obs: Float32Array) {
  const input = new ort.Tensor("float32", obs, [1, obs.length]);
  const out = await session.run({ obs: input });
  const q = out.q.data as Float32Array;

  let best = 0;
  for (let i = 1; i < q.length; i++) if (q[i] > q[best]) best = i;
  return best;
}