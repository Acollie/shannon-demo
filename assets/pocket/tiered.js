// GLM Pocket model adapter: two tiers behind the same interface as GLMModel (assets/chat/model.js).
//   0. Each message is sorted first (intentOf): chit-chat and requests ("hi", "write me a haiku", "plan my day") never go to
//      the graph as an answer; questions about the user go to their saved "About me" facts; fact questions go to the graph.
//   1. Fact questions: Shannon's graph engine answers (~0.1 s). Its answer is shown only when it passes the calibrated
//      confidence rule (assets/pocket/calibration.json, set offline on Mintaka dev: "pick" answers with path score >= cut).
//   2. Everything else goes to the on-device chat model (WebLLM, WebGPU, in a Web Worker) when it is on. For fact questions
//      below the rule it is given the graph's candidate paths, labelled "possible facts from the graph, may be wrong".
//   3. Chat model off or unavailable: a plain "notice" reply (what GLM can do without it, how to turn it on); the graph's
//      best guess, if any, is attached for a collapsed "graph's best guess" view, never as the answer.
// Reply = GLMModel's Reply plus: tier ("graph" | "lm" | "memory" | "notice"), intent ("fact" | "chat" | "personal"),
// text (LM prose), route (one line on why this tier answered), graph_tried (the graph's attempt), facts_used.
import { GLMModel } from "../chat/model.js";

const WEBLLM = "https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm@0.2.85/+esm";
// Chat models. WebLLM (fast, WebGPU) runs in Chrome / Edge / Safari:
//   small: phones and low-memory devices. large: the default on desktops/laptops, Qwen3-1.7B, picked by a check on 10
//   prompts against Qwen2.5-1.5B: same facts right, but it keeps to the "be concise" prompt and plans/advises better;
//   923 MB vs 828 MB, ~47 vs ~52 tok/s on an M3 Pro. (Llama-3.2-1B, gemma-2-2b and SmolLM2-1.7B could not be fetched in
//   the test browser: Hugging Face CDN errors.) f32: the q4f32 build for GPUs without shader-f16.
// compat: transformers.js + ONNX Runtime Web on the CPU (WASM, int8; assets/pocket/tjs_worker.js) where WebLLM can't run:
//   Firefox / Zen (WebGPU allows only 9 storage buffers per shader stage, WebLLM needs 10; and ONNX Runtime's WebGPU kernels
//   give garbage text there in q4f16, q4 and fp16, tested on Firefox 155 / macOS) and browsers without WebGPU.
//   Measured in Firefox 155 on an M3 Pro: ~13 s load from cache, ~0.8 s to first token, ~12 tokens/s.
export const LMS = {
  small: { engine: "webllm", id: "Qwen2.5-0.5B-Instruct-q4f16_1-MLC", f32: "Qwen2.5-0.5B-Instruct-q4f32_1-MLC", label: "Qwen 0.5B", mb: 274 },
  large: { engine: "webllm", id: "Qwen3-1.7B-q4f16_1-MLC", f32: "Qwen3-1.7B-q4f32_1-MLC", label: "Qwen3 1.7B", mb: 930 },
  compat: { engine: "tjs", id: "onnx-community/Qwen2.5-0.5B-Instruct", label: "Qwen 0.5B (CPU)", mb: 512, dtype: "q8", device: "wasm" },
};
const isFirefox = () => /Firefox\//.test(navigator.userAgent || "");
const browserName = () => isFirefox() ? "Firefox / Zen" : /Edg\//.test(navigator.userAgent) ? "Edge" : /Chrome\//.test(navigator.userAgent) ? "Chrome"
  : /Safari\//.test(navigator.userAgent) ? "Safari" : "this browser";

// the WebLLM engine behind one generate() call
class WebLLMBackend {
  constructor(engine, worker, id) { this.engine = engine; this.worker = worker; this.id = id; }
  async generate(messages, opt, onDelta) {
    const req = { messages, stream: true, temperature: opt.temperature, top_p: opt.top_p, max_tokens: opt.max_tokens, stream_options: { include_usage: true } };
    if (/Qwen3/.test(this.id)) req.extra_body = { enable_thinking: false };
    let usage = null;
    for await (const c of await this.engine.chat.completions.create(req)) {
      if (c.usage) usage = c.usage;
      const d = c.choices[0]?.delta?.content || "";
      if (d) onDelta(d);
    }
    return { tokens: usage ? usage.completion_tokens : null };
  }
  async unload() { try { await this.engine.unload(); } catch { /* ignore */ } this.worker.terminate(); }
}
// the transformers.js worker behind the same generate()
class TJSBackend {
  constructor(url) { this.worker = new Worker(url, { type: "module" }); this.n = 0; this.jobs = new Map(); }
  load(model, dtype, device, onProgress) {
    return new Promise((ok, bad) => {
      this.worker.onmessage = ({ data: m }) => {
        if (m.type === "progress") onProgress({ text: m.text, frac: m.frac });
        else if (m.type === "ready") ok();
        else if (m.type === "error" && m.id == null) bad(new Error(m.error));
        else if (m.id != null && this.jobs.has(m.id)) this.jobs.get(m.id)(m);
      };
      this.worker.onerror = e => bad(new Error(e.message || "the chat model worker failed to start"));
      this.worker.postMessage({ type: "load", model, dtype, device });
    });
  }
  generate(messages, opt, onDelta) {
    const id = ++this.n;
    return new Promise((ok, bad) => {
      this.jobs.set(id, m => {
        if (m.type === "delta") onDelta(m.t);
        else { this.jobs.delete(id); m.type === "done" ? ok({ tokens: m.tokens }) : bad(new Error(m.error)); }
      });
      this.worker.postMessage({ type: "generate", id, messages, ...opt });
    });
  }
  async unload() { this.worker.terminate(); }
}

// small models (0.5B) follow a long prompt badly and the CPU engine pays for every prompt token, so they get a short one
export const SYSTEM_PROMPT_SMALL = "You are GLM Pocket, a friendly assistant. Reply in one to three short sentences unless asked for more. If you are not sure of a fact, say so.";
// "My name is Sam" -> "the user's name is Sam" (small models confuse a first-person note with themselves)
const third = t => t.replace(/\b(I'm|I am)\b/gi, "the user is").replace(/\bmy\b/gi, "the user's").replace(/\bme\b/gi, "the user").replace(/\bI\b/g, "the user").replace(/[.\s]+$/, "");
export const SYSTEM_PROMPT = [
  "You are GLM Pocket, a helpful assistant that runs entirely on the user's own device, with no internet access.",
  "- Be warm and concise: usually one to four sentences, or a short list when it helps. Write more only when asked.",
  "- For writing and planning tasks (poems, emails, plans, ideas), just do the task well.",
  "- If you are not sure of a fact, say so plainly instead of guessing. Never invent sources, links, quotes or numbers.",
  "- You cannot browse or look anything up. Only mention the knowledge graph if graph facts are given below, and use them only if they clearly answer the question.",
  "- Use the user's saved \"About me\" facts only when they matter for the request; do not repeat them back otherwise.",
].join("\n");

// ---- message sorting (cheap rules; nothing here answers anything, it only decides who may answer)
const PERSONAL = /\b(my|mine|about me|who am i)\b/i;      // "what's my name?" -> the user's saved facts, not the world graph
const GREET = /^(hi|hello|hey|hiya|yo|howdy|sup|good (morning|afternoon|evening|night)|thanks|thank you|thx|cheers|ok|okay|cool|nice|great|bye|goodbye|see you|sorry|lol|haha|wow)\b/i;
const SMALLTALK = /\b(how are you|how're you|how (is|was) your|how's it going|what's up|whats up|who are you|what are you|what can you do|what is your name|what's your name|are you (a|an|real|human|alive|ok)|do you (like|love|think|know me|feel|remember))\b/i;
const TASK = /^(please |pls |can you |could you |would you |will you |can u |i want you to |i need you to |i'd like you to |let's |lets )?(write|compose|draft|rewrite|reword|rephrase|summari[sz]e|translate|plan|help|make|create|give|suggest|recommend|brainstorm|tell me (a|an|some) |tell me about|explain|describe|teach|show me how|generate|imagine|pretend|act as|fix|improve|edit|correct|proofread|outline|design|convert|calculate|solve|compare|list (some|a few|ideas|ways))\b/i;
const OPEN = /\b(should i|should we|could i|can i|do you think|in your opinion|advice|tips?|ideas?|how (do|can|should|would) (i|we|you)|why\b|what if|what would|is it (ok|okay|good|bad|safe|worth)|best way|a good|any good|some good)\b/i;
const MATHY = /\d\s*([-+*/x×÷^%]|times|plus|minus|divided by|squared)\s*\d|\b(sqrt|square root)\b/i;
const QSTART = /^(who|whom|whose|what|which|when|where|how (many|much|old|tall|long|big|far|large|high|deep)|is|are|was|were|did|does|do|has|have|had|name|in (what|which))\b/i;

export function intentOf(text) {
  const t = String(text || "").trim().replace(/\s+/g, " ");
  const words = t.split(" ").length;
  if (!t) return "chat";
  if (GREET.test(t) && (words <= 6 || !/\?$/.test(t))) return "chat";
  if (SMALLTALK.test(t) || TASK.test(t) || OPEN.test(t) || MATHY.test(t)) return "chat";
  const asks = QSTART.test(t) || /\?$/.test(t);
  if (!asks) return "chat";   // statements ("I'm vegetarian", "my favourite colour is green") and anything else open-ended
  if (PERSONAL.test(t)) return "personal";
  return "fact";
}

// does a short model answer name one of the graph's answers? (full name, a year, or the last word of a name)
const norm = x => String(x).normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase();
export function agrees(text, answers) {
  const t = norm(text);
  return (answers || []).slice(0, 8).some(a => {
    const n = norm(a);
    if (n && t.includes(n)) return true;
    const y = n.match(/\b(1\d{3}|20\d{2})\b/);
    if (y) return t.includes(y[1]);
    const w = n.split(/[^a-z0-9]+/).filter(x => x.length >= 4);
    return w.length > 0 && new RegExp("\\b" + w[w.length - 1] + "\\b").test(t);
  });
}

// small models copy the previous reply when given history, so they only see the last turn when the message reads as a follow-up
const FOLLOWUP = /^(and|but|also|so|then|ok|okay|what about|how about|make it|now|again|more|shorter|longer|another|why|explain that|tell me more|that|it|this|they|he|she|yes|no)\b|^\S+(\s+\S+){0,2}\??$/i;

const fmtPath = (from, labels, end) => [from, ...labels].join(" → ") + (end ? " → " + end : "");

export class TieredModel extends GLMModel {
  constructor(base, opts = {}) {
    super(base, opts);
    this.id = "glm-pocket"; this.name = "GLM Pocket";
    this.cal = null; this.graphMin = null;
    this.lm = null; this.lmKey = null; this.lmId = null; this.lmState = "off"; this.lmError = ""; this.lmLoading = null; this.lmFrac = null;
    this.workerUrl = base + "assets/pocket/lm_worker.js";
  }

  async init(onProgress) {
    try { this.cal = await fetch(this.base + "assets/pocket/calibration.json").then(r => r.json()); this.graphMin = this.cal.threshold; }
    catch (e) { console.warn("[pocket] no calibration file; every committed graph answer counts as confident", e); }
    return super.init(onProgress);
  }

  // the cut is calibrated on the general (Wikidata) pack; the movies pack's scorer has its own scale, so there any
  // committed answer counts (documented in calibration.json)
  trusted(r) {
    if (!r || !r.ok) return false;
    if (r.pack !== (this.cal && this.cal.pack) || this.graphMin == null) return true;
    if (this.cal.types && !this.cal.types.includes(r.type)) return false;   // e.g. count / yes-no stay near base rate at any score
    return r.score != null && r.score >= this.graphMin;
  }
  ruleText() { return this.cal ? `${(this.cal.types || []).join("/")} answers with path score ≥ ${this.graphMin}` : "no rule"; }

  // ---- the on-device LM
  // What this device can run. key: which WebLLM model to offer (phones / small memory: 0.5B, else the larger default);
  // engine: "webllm" (WebGPU with enough limits) or "tjs" (CPU / WASM: Firefox / Zen, or no WebGPU);
  // note: one line naming the browser and the fix when the fast engine can't run.
  static async webgpu() {
    const ua = navigator.userAgent || "";
    const phone = /Android|iPhone|iPad|iPod|Mobile/i.test(ua) || (matchMedia("(pointer: coarse)").matches && Math.min(screen.width, screen.height) < 820);
    const mem = navigator.deviceMemory;   // Chrome only, capped at 8 (GB); absent elsewhere
    const lowMem = typeof mem === "number" && mem < 4;
    const base = { ok: true, phone, lowMem, browser: browserName() };
    const cpu = why => ({ ...base, engine: "tjs", device: "wasm", key: "compat", f16: false, slow: true, note: why });
    try {
      if (!navigator.gpu) return cpu(isFirefox()
        ? "Firefox / Zen: WebGPU is off. Turn it on in about:config (dom.webgpu.enabled = true) and reload, or use Chrome or Safari for the faster, larger model. Until then the chat model runs on the CPU (slower)."
        : `${browserName()} has no WebGPU, so the chat model runs on the CPU (slower). Chrome, Edge or Safari 26 run a faster, larger model on the GPU.`);
      const a = await navigator.gpu.requestAdapter();
      if (!a) return cpu(isFirefox()
        ? "Firefox / Zen: WebGPU found no usable GPU (check about:config: dom.webgpu.enabled = true, and that hardware acceleration is on), so the chat model runs on the CPU (slower). Chrome or Safari run a faster, larger model."
        : `${browserName()} has WebGPU but no usable GPU, so the chat model runs on the CPU (slower).`);
      const f16 = a.features.has("shader-f16"), sb = a.limits.maxStorageBuffersPerShaderStage;
      if (sb < 10 || isFirefox()) return { ...cpu(""), sb, note: isFirefox()
        ? `Firefox / Zen: its WebGPU allows ${sb} storage buffers per shader and the fast chat engine needs 10, so here the chat model runs on the CPU: a smaller model, slower (about 12 words a second on a recent Mac). Chrome, Edge or Safari run the faster, larger model on the GPU.`
        : `${browserName()}'s WebGPU allows ${sb} storage buffers per shader and the fast chat engine needs 10, so the chat model runs on the CPU (smaller, slower).` };
      return { ...base, engine: "webllm", f16, sb, key: phone || lowMem ? "small" : "large", note: "" };
    } catch (e) { return cpu("The WebGPU check failed (" + (e.message || e) + "), so the chat model runs on the CPU (slower)."); }
  }
  static modelId(key, f16 = true) { return f16 || !LMS[key].f32 ? LMS[key].id : LMS[key].f32; }
  static sizeMB(key) { return LMS[key].mb; }

  async enableLM(key = "small", onProgress = () => {}) {
    if (this.lm && this.lmKey === key) return this.lm;
    if (this.lmLoading) return this.lmLoading;
    this.lmState = "loading"; this.lmError = ""; this.lmKey = key; this.lmFrac = null;
    const prog = r => { this.lmFrac = r.frac; onProgress(r); };
    this.lmLoading = (async () => {
      try {
        const g = await TieredModel.webgpu();
        if (this.lm) { await this.lm.unload(); this.lm = null; }
        if (LMS[key].engine === "webllm" && g.engine === "webllm") {
          try {
            try { this.lm = await this.loadWebLLM(key, g, prog); }
            catch (e) {   // a dropped download (Hugging Face CDN hiccup): try once more; finished files are already cached
              const msg = String(e && e.message || e);   // errors from the WebLLM worker can arrive as plain strings
              if (!/network|fetch|cache/i.test(msg)) throw e;
              console.warn("[pocket] chat model download failed, retrying:", msg);
              this.lm = await this.loadWebLLM(key, g, prog);
            }
          } catch (e) {   // e.g. "requested maxStorageBuffersPerShaderStage exceeds limit": use the compatibility engine
            console.warn("[pocket] WebLLM failed, switching to the compatibility engine:", e && e.message || e);
            key = this.lmKey = "compat";
            this.lm = await this.loadTJS(prog);
          }
        } else { key = this.lmKey = "compat"; this.lm = await this.loadTJS(prog); }
        this.lmState = "ready";
        return this.lm;
      } catch (e) {
        this.lm = null; this.lmId = null; this.lmState = "error"; this.lmError = e.message || String(e);
        throw e;
      } finally { this.lmLoading = null; }
    })();
    return this.lmLoading;
  }
  async loadWebLLM(key, g, prog) {
    const id = TieredModel.modelId(key, g.f16);
    const webllm = await import(/* webpackIgnore: true */ WEBLLM);
    const worker = new Worker(this.workerUrl, { type: "module" });
    try {
      const eng = await webllm.CreateWebWorkerMLCEngine(worker, id,
        { initProgressCallback: r => prog({ text: r.text, frac: typeof r.progress === "number" ? r.progress : null }) }, { context_window_size: 4096 });
      this.lmId = id; this.lmDevice = "webgpu";
      return new WebLLMBackend(eng, worker, id);
    } catch (e) { worker.terminate(); throw e; }
  }
  async loadTJS(prog) {
    const m = LMS.compat, b = new TJSBackend(this.base + "assets/pocket/tjs_worker.js");
    try { await b.load(m.id, m.dtype, m.device, prog); this.lmId = `${m.id} (${m.dtype}, CPU)`; this.lmDevice = m.device; return b; }
    catch (e) { b.unload(); throw e; }
  }
  sysPrompt() { return this.lmKey === "large" ? SYSTEM_PROMPT : SYSTEM_PROMPT_SMALL; }
  lmReady() { return this.lmState === "ready" && !!this.lm; }

  // the graph's candidate facts, as short lines for the LM ("possible facts from the graph, may be wrong")
  graphFacts(g) {
    if (!g || !g.ok) return [];
    const lines = [];
    if (g.path && g.answers.length) lines.push(fmtPath(g.path.from, g.path.steps.map(s => s.relation), g.answers.slice(0, 3).join(", ")));
    else if (g.answers.length) lines.push(`${g.type || "answer"}: ${g.answers.slice(0, 3).join(", ")}`);
    for (const a of g.alts || []) if (a.answers && a.answers.length) lines.push(fmtPath(a.from, a.labels, a.answers.slice(0, 3).join(", ")));
    return lines.slice(0, 4);
  }

  // ---- answer
  // opts: pack, history [{role, content}] (this chat's earlier turns), facts [{text}] (relevant About-me facts),
  //       onToken(textSoFar) for streaming LM text, onStatus(text) while waiting for the chat model to finish loading
  async answer(question, { pack, history = [], facts = [], onToken = () => {}, onStatus = () => {} } = {}) {
    const t0 = performance.now();
    const intent = intentOf(question);
    let g = null;
    if (intent === "fact") {
      g = await super.answer(question, { pack });
      if (this.trusted(g)) {
        const route = `A fact question, and the graph's answer passes the confidence rule (${this.ruleText()}; this one scored ${g.score != null ? g.score.toFixed(2) : "n/a"}).`;
        if (!this.lmReady()) return { ...g, tier: "graph", intent, route };
        // the rule is only ~60% precise, so with the chat model on, the graph's answer is shown only if the model,
        // asked on its own (no graph facts), names the same thing; otherwise the model answers and the graph's
        // answer goes to the details
        onStatus("Checking the graph's answer with the chat model…");
        let closed = "";
        try {
          await this.lm.generate([{ role: "system", content: this.sysPrompt() + " Answer in a few words only." }, { role: "user", content: question }],
            { temperature: 0, top_p: 1, max_tokens: 32 }, d => { closed += d; });
          closed = closed.replace(/<think>[\s\S]*?(<\/think>|$)/g, "").trim();
        } catch (e) { console.warn("[pocket] cross-check failed", e); closed = null; }
        if (closed == null || agrees(closed, g.answers)) return { ...g, tier: "graph", intent, check: closed == null ? null : { agree: true, text: closed },
          ms: performance.now() - t0, route: route + (closed == null ? "" : " The chat model, asked on its own, gave the same answer.") };
        g.disagree = closed;
      }
    }
    const graph_tried = g ? { ok: g.ok, answers: g.answers, score: g.score, path: g.path, type: g.type, pack: g.pack, disagree: g.disagree || null } : null;
    if (!this.lmReady() && this.lmLoading) {   // the user already said yes: wait for it rather than giving a non-answer
      onStatus("Waiting for the chat model to finish loading…");
      try { await this.lmLoading; } catch { /* falls through to the notice below */ }
    }
    if (!this.lmReady()) {
      if (intent === "personal") return { ok: facts.length > 0, tier: "memory", intent, answers: [], facts_used: facts, ms: performance.now() - t0, pack, model: "about-me",
        route: "A question about you: answered from your saved facts (the chat model is off)." };
      return { ok: false, tier: "notice", intent, answers: [], graph_tried, names: g ? g.names : [], ms: performance.now() - t0, bytes: g ? g.bytes : 0, pack, model: this.id,
        route: intent === "fact"
          ? (g && g.ok ? `The graph's best guess is below its confidence rule (${this.ruleText()}), so it is not shown as an answer; the chat model is off.` : "The graph found no answer, and the chat model is off.")
          : "Not a fact question, so the graph does not answer it; that needs the chat model, which is off." };
    }
    const gf = intent === "fact" && !(g && g.disagree) ? this.graphFacts(g) : [];
    const small = this.lmKey !== "large";
    const sys = [this.sysPrompt(),
      facts.length ? (small ? "Facts about the user you are talking to: " + facts.map(f => third(f.text)).join("; ") + ". Use them only if relevant."
        : "About the user (notes the user wrote about themselves, so \"I\" and \"my\" in them mean the user; when you use them, speak to the user as \"you\"):\n" + facts.map(f => "- " + f.text).join("\n")) : "",
      gf.length ? (small ? "Possibly useful facts (may be wrong): " + gf.join("; ") + "."
        : "Possible facts from Shannon's knowledge graph. They may be wrong or off-topic; use one only if it clearly answers the question, and say it came from the graph:\n" + gf.map(x => "- " + x).join("\n")) : ""].filter(Boolean).join("\n\n");
    const messages = [{ role: "system", content: sys }, ...(small ? (FOLLOWUP.test(question.trim()) ? history.slice(-2) : []) : history.slice(-6)), { role: "user", content: question }];
    let raw = "", text = "", tFirst = null;
    const { tokens } = await this.lm.generate(messages, { temperature: small ? 0 : 0.4, top_p: 0.9, max_tokens: small ? 220 : 400 }, d => {
      raw += d;
      text = raw.replace(/<think>[\s\S]*?(<\/think>|$)/g, "").replace(/^\s+/, "");
      if (text) { tFirst ??= performance.now(); onToken(text); }
    });
    return { ok: true, tier: "lm", intent, text: text.trim(), answers: [], type: "", explain: "", path: null, names: g ? g.names : [], score: null,
      ms: performance.now() - t0, first_token_ms: tFirst ? tFirst - t0 : null, tokens,
      bytes: g ? g.bytes : 0, reason: "", pack, model: this.lmId, facts_used: facts, graph_facts: gf, graph_tried,
      route: intent === "personal" ? "A question about you: the chat model answered, with your saved facts as context."
        : intent === "chat" ? "Not a fact question, so the chat model answered (the graph was not asked)."
        : g && g.disagree ? `The graph's answer passed its confidence rule, but the chat model, asked on its own, disagreed (“${g.disagree.slice(0, 80)}”), so the graph's answer is not shown; the chat model answered without it.`
        : g && g.ok ? `The graph's answer is below its confidence rule (${this.ruleText()}), so the chat model answered, given the graph's candidate facts marked “may be wrong”.`
        : "The graph found no answer, so the chat model answered from its own knowledge." };
  }
}
