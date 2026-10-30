// Chat model adapter: the one interface the chat page (assets/chat/chat.js) talks to.
//
// The chat UI never touches the engine directly. A model is any object with:
//   id, name                                   short id and display name
//   init(onProgress)                -> Promise  load runtime + encoder; onProgress({text, frac|null})
//   catalog()                       -> [{id, title, pick, detail, status, default, preview}]
//   loadPack(id, onProgress)        -> Promise<{id, title, bytes, total_bytes, lazy}>
//   unloadPack(id)                  -> void
//   packState(id)                   -> null | "loading" | {id, title, bytes, ...}
//   examples(packId)                -> [{q, tag}]  starter questions
//   answer(question, {pack})        -> Promise<Reply>
//
// Reply = {
//   ok: bool,              true when the model committed to an answer
//   answers: [string],     the answer(s), raw strings from the model (the UI formats large numbers)
//   type: string,          answer type: "pick" | "count" | "yes / no" | "largest" | "smallest" | "compare"
//   explain: string,       one line from the model on how the answer type was applied (empty if none)
//   path: { from, steps: [{relation, n, sample: [string]}] } | null    the walked graph path, human-readable hops
//   names: [string],       names the linker found in the question
//   score: number|null,    the chosen path's raw score (not a calibrated probability)
//   ms: number,            time to answer on this device
//   bytes: number,         graph bytes downloaded for this question
//   reason: string,        why there is no answer (when ok is false)
//   pack, model            which pack and model answered
// }
//
// Future models plug in here: a better shortlist / linker, idea-vec reasoning, or a words-out decoder only need to
// implement the same methods (a decoder could add e.g. `text` to the Reply; the UI shows answers + path as now).
// GLMModel below wraps the current GLM engine (assets/glm/engine.js) without changing its answer logic.
import { Engine } from "../glm/engine.js";

const STEP_TEXT = {
  "runtime script": "Loading the runtime", "runtime wasm": "Loading the runtime (WebAssembly)",
  "encoder manifest": "Loading the encoder manifest", "vocabulary": "Loading the vocabulary",
  "encoder download": "Downloading the encoder", "encoder session": "Starting the encoder", "warm-up": "Warming up the encoder",
};
const mb = b => (b / 1048576).toFixed(1) + " MB";
const TYPE = { set: "pick", count: "count", yesno: "yes / no", argmax: "largest", argmin: "smallest", compare: "compare" };

export class GLMModel {
  constructor(base, { encoder = "fp16" } = {}) {
    this.id = "glm-v1"; this.name = "GLM v1";
    this.base = base; this.encoder = encoder;
    this.engine = null; this.cat = []; this.ex = {}; this.packs = new Map(); this.progress = () => {};
  }

  async init(onProgress = () => {}) {
    this.progress = onProgress;
    const get = f => fetch(this.base + f).then(r => { if (!r.ok) throw new Error(`${f}: HTTP ${r.status}`); return r.json(); });
    const [cat, ex, exg] = await Promise.all([get("glm/packs.json"), get("glm/examples.json"), get("glm/examples_general.json")]);
    this.cat = cat.packs;
    this.ex = { movies: ex.examples.map(e => ({ q: e.q, tag: `${e.hop}-step` })), general: exg.examples.map(e => ({ q: e.q, tag: e.tag })) };
    this.engine = new Engine(this.base, m => this.progress(this.statusText(m)));
    const m = await this.engine.init(this.encoder);
    this.info = m;
    return m;
  }

  statusText({ step, done, total }) {
    let t = STEP_TEXT[step] || (step.startsWith("pack ") ? `Loading the ${step.slice(5).split(":")[0]} pack (${step.split(": ")[1]})` : step);
    if (done != null) t += ` · ${total ? `${mb(done)} of ${mb(total)}` : mb(done)}`;
    return { text: t, frac: done != null && total ? done / total : null };
  }

  catalog() { return this.cat; }
  examples(id) { return this.ex[id] || []; }
  packState(id) { return this.packs.get(id) || null; }

  async loadPack(id, onProgress) {
    if (onProgress) this.progress = onProgress;
    this.packs.set(id, "loading");
    try { const m = await this.engine.load(id); this.packs.set(id, m); return m; }
    catch (e) { this.packs.delete(id); throw e; }
  }

  unloadPack(id) { this.engine.unload(id); this.packs.delete(id); }

  async answer(question, { pack } = {}) {
    const t0 = performance.now();
    const base = { ok: false, answers: [], type: "", explain: "", path: null, names: [], score: null, ms: 0, bytes: 0, reason: "", pack, model: this.id };
    let r;
    try { r = await this.engine.ask(question, { pack }); }
    catch (e) { return { ...base, reason: "Something failed: " + (e && e.message || e), ms: performance.now() - t0 }; }
    const tm = r.timings || {};
    const out = { ...base, ms: tm.total_ms != null ? tm.total_ms : performance.now() - t0, bytes: tm.download_bytes || 0, pack: r.pack || pack };
    if (r.link && r.link.name) out.names = [r.link.name];
    if (r.topics_named) out.names = r.topics_named;
    if (!r.ok) return { ...out, reason: r.reason || "" };
    const answers = (r.answers || []).map(String);
    out.answers = answers;
    out.ok = answers.length > 0;
    out.type = r.kind === "wikidata" ? (TYPE[r.op] || r.op) : "pick";
    out.explain = r.explain_op && r.explain_op !== "no answer" ? r.explain_op : "";
    out.score = typeof r.score === "number" ? r.score : null;
    // the next-best paths the graph considered (from + relation labels + their answers); used by GLM Pocket as LM context
    out.alts = (r.alternatives || []).map(a => ({ from: a.from || r.topic, labels: a.labels || [], answers: (a.answers || []).map(String), score: a.score }));
    if (r.hops && r.hops.length) out.path = { from: r.topic, steps: r.hops.map(h => ({ relation: h.label, n: h.n, sample: h.sample.map(String) })) };
    if (!out.ok) out.reason = r.reason || "";
    return out;
  }
}
