// src/lib/onnxPolicy.ts
import * as ort from "onnxruntime-web";

type ORTSession = ort.InferenceSession;

function argMax(arr: Float32Array | number[]) {
  let bestI = 0;
  let bestV = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    const v = Number(arr[i]);
    if (v > bestV) {
      bestV = v;
      bestI = i;
    }
  }
  return bestI;
}

export async function loadPolicy(url: string): Promise<ORTSession> {
  // WASM is the simplest for browser demos
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;

  const session = await ort.InferenceSession.create(url, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });

  console.log("[onnx] loaded", {
    inputs: session.inputNames,
    outputs: session.outputNames,
  });

  return session;
}

/**
 * Choose an action from an ONNX policy.
 * - If anything is mismatched, returns a random action so the agent still moves.
 */
export async function chooseAction(session: ORTSession, obs: Float32Array): Promise<number> {
  try {
    // Most common: input is [1, 18]
    const inputName = session.inputNames[0];
    const tensor = new ort.Tensor("float32", obs, [1, obs.length]);

    const feeds: Record<string, ort.Tensor> = {};
    feeds[inputName] = tensor;

    const out = await session.run(feeds);

    // Pick first output
    const outName = session.outputNames[0];
    const y = out[outName]?.data;

    if (!y) throw new Error("ONNX output missing");

    // Output could be logits [1, 6] or [6]
    // onnxruntime-web returns TypedArray
    const data = y as Float32Array;

    // If it’s [1,6], it still comes as length 6
    const action = argMax(data);

    // sanity clamp 0..5
    return Math.max(0, Math.min(5, action));
  } catch (e) {
    console.warn("[onnx] chooseAction failed, using random action", e);
    return Math.floor(Math.random() * 6);
  }
}