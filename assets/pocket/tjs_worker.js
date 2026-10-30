// GLM Pocket: the compatibility chat engine (transformers.js 3.8.1 + ONNX Runtime Web), used where WebLLM can't run:
// Firefox / Zen (their WebGPU allows 9 storage buffers per shader stage, WebLLM needs 10) and browsers without WebGPU
// (device "wasm": CPU, slower). Runs off the page's main thread. Only the library (jsDelivr) and the model files
// (Hugging Face) are downloaded; no prompt or reply leaves this device.
import { AutoTokenizer, AutoModelForCausalLM, TextStreamer, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

env.allowLocalModels = false;
let tok = null, model = null;
const files = new Map();   // file -> [loaded, total], for one overall progress fraction

function progress(p) {
  if (p.status === "progress" && p.file) files.set(p.file, [p.loaded || 0, p.total || 0]);
  if (p.status === "done" && p.file && files.has(p.file)) { const [, t] = files.get(p.file); files.set(p.file, [t, t]); }
  let l = 0, t = 0;
  for (const [a, b] of files.values()) { l += a; t += b; }
  const mb = x => (x / 1048576).toFixed(0);
  self.postMessage({ type: "progress", frac: t ? l / t : null, text: t ? `Downloading the model: ${mb(l)} of ${mb(t)} MB` : `Loading ${p.file || "the model"}…` });
}

self.onmessage = async ({ data: m }) => {
  if (m.type === "load") {
    try {
      tok = await AutoTokenizer.from_pretrained(m.model, { progress_callback: progress });
      model = await AutoModelForCausalLM.from_pretrained(m.model, { dtype: m.dtype, device: m.device, progress_callback: progress });
      self.postMessage({ type: "progress", frac: 1, text: "Warming up…" });
      const warm = tok.apply_chat_template([{ role: "user", content: "Hi" }], { add_generation_prompt: true, return_dict: true });
      await model.generate({ ...warm, max_new_tokens: 1 });
      self.postMessage({ type: "ready" });
    } catch (e) { self.postMessage({ type: "error", error: (e && e.message) || String(e) }); }
  }
  if (m.type === "generate") {
    try {
      const inputs = tok.apply_chat_template(m.messages, { add_generation_prompt: true, return_dict: true });
      const streamer = new TextStreamer(tok, { skip_prompt: true, skip_special_tokens: true,
        callback_function: t => self.postMessage({ type: "delta", id: m.id, t }) });
      const out = await model.generate({ ...inputs, max_new_tokens: m.max_tokens || 400, do_sample: (m.temperature || 0) > 0,
        temperature: m.temperature || 1, top_p: m.top_p || 1, repetition_penalty: 1.05, streamer });
      self.postMessage({ type: "done", id: m.id, tokens: out.dims[1] - inputs.input_ids.dims[1] });
    } catch (e) { self.postMessage({ type: "error", id: m.id, error: (e && e.message) || String(e) }); }
  }
};
