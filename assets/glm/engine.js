// GLM v0 engine: MiniLM (onnxruntime-web, WASM, one thread) + packs + path scorer, on the page's main thread.
// No Web Worker: hosts that sandbox pages (e.g. Claude artifacts) can block module workers silently. One question is ~15 ms.
// Runtime: onnxruntime-web 1.20.1 "bundle" build, which inlines the Emscripten glue (the plain UMD/ESM builds import
// ort-wasm-simd-threaded.mjs at run time). The .wasm binary is fetched from this site and handed over as wasmBinary,
// so nothing is fetched from a CDN at run time except the bundle script itself (with a same-site copy as fallback).
//   const e = new Engine(base, onStatus); await e.init("fp16"); await e.load("movies"); await e.ask("who directed Memento")
// onStatus({step, text, done, total}) is called for every load step and every progress tick.
import { Pack, Encoder, GLM } from "./glm.js";
import { WikiPack } from "./wiki.js";

const ORT_VER = "1.20.1";
const ORT_SRC = [
  `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VER}/dist/ort.bundle.min.mjs`,
  "vendor/ort/ort.bundle.min.js",                     // same-site copy (relative to base)
];
const STALL_MS = 30000;
const now = () => performance.now();

// Runs fn(tick) as a named step. Fails with "<step> hung" when no progress is reported for STALL_MS.
function step(name, onStatus, fn) {
  return new Promise((resolve, reject) => {
    let timer = null, over = false;
    const arm = () => { clearTimeout(timer); timer = setTimeout(() => { over = true; reject(new Error(`step "${name}" hung (no progress for ${STALL_MS / 1000} s)`)); }, STALL_MS); };
    const tick = (done = null, total = null) => { if (!over) { arm(); onStatus({ step: name, done, total }); } };
    tick();
    Promise.resolve().then(() => fn(tick)).then(
      v => { clearTimeout(timer); if (!over) resolve(v); },
      e => { clearTimeout(timer); if (!over) { const err = e instanceof Error ? e : new Error(String(e)); err.message = `step "${name}" failed: ${err.message}`; reject(err); } });
  });
}

const withTimeout = (p, ms, fallback) => Promise.race([p, new Promise(r => setTimeout(() => r(fallback), ms))]);

let cachePromise = null;
function cacheOpen() {   // Cache Storage when the browser allows it; never wait on it for more than 3 s
  if (!cachePromise) cachePromise = (typeof caches === "undefined") ? Promise.resolve(null)
    : withTimeout(caches.open("shannon-glm-v0").catch(() => null), 3000, null);
  return cachePromise;
}

function isBinaryPath(url) { return /\.(bin|wasm|part\d+)$/.test(url); }

// fetch with byte progress
async function getBytes(url, tick) {
  const cache = await cacheOpen();
  let res = null;
  if (cache) { try { res = await withTimeout(cache.match(url), 3000, null); } catch { res = null; } }
  const hit = !!res;
  if (!res) {
    res = await fetch(url);
    if (!res.ok || (isBinaryPath(url) && /text\/html/.test(res.headers.get("content-type") || ""))) throw new Error(`${url}: HTTP ${res.status}`);
    if (cache) { try { await withTimeout(cache.put(url, res.clone()), 3000); } catch { /* storage full or blocked */ } }
  }
  const total = +res.headers.get("content-length") || 0;
  if (!res.body || hit) { const b = new Uint8Array(await res.arrayBuffer()); tick(b.length, b.length); return b; }
  const rd = res.body.getReader(), chunks = [];
  let done = 0;
  for (;;) {
    const { done: end, value } = await rd.read();
    if (end) break;
    chunks.push(value); done += value.length;
    tick(done, total);
  }
  const out = new Uint8Array(done);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

function fromBase64(t) {
  t = t.trim();
  if (Uint8Array.fromBase64) return Uint8Array.fromBase64(t);
  const bin = atob(t), out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

// binary file; hosts that only serve text get a base64 copy next to it ("<name>.b64.txt")
// (after one file is found only as base64, later files try the base64 copy first)
let preferB64 = false;
const getB64 = async (url, tick) => fromBase64(new TextDecoder().decode(await getBytes(url + ".b64.txt", tick)));
async function getBin(url, tick) {
  if (preferB64) { try { return await getB64(url, tick); } catch (e) { console.warn(`[glm] ${url}.b64.txt failed (${e.message}); trying ${url}`); return getBytes(url, tick); } }
  try { return await getBytes(url, tick); }
  catch (e) {
    console.warn(`[glm] ${url} not served as binary (${e.message}); trying ${url}.b64.txt`);
    const b = await getB64(url, tick);
    preferB64 = true;
    return b;
  }
}

const text = async (url, tick) => new TextDecoder().decode(await getBytes(url, tick));

export class Engine {
  constructor(base, onStatus = () => {}) { this.base = base; this.onStatus = onStatus; this.glm = null; this.ort = null; }

  async loadRuntime() {
    return step("runtime script", this.onStatus, async tick => {
      const errs = [];
      for (const src of ORT_SRC) {
        const url = new URL(src, this.base).href;
        tick();
        try {
          const ort = await withTimeout(import(/* webpackIgnore: true */ url), 20000, null);
          if (!ort) throw new Error("timed out after 20 s");
          if (!ort.InferenceSession) throw new Error("module has no InferenceSession");
          console.info(`[glm] onnxruntime-web ${ORT_VER} from ${url}`);
          return ort;
        } catch (e) { errs.push(`${url}: ${e.message || e}`); console.error("[glm] runtime import failed", url, e); }
      }
      throw new Error(errs.join("; "));
    });
  }

  async init(encoder) {
    const t0 = now(), base = this.base, S = this.onStatus;
    const ort = this.ort || (this.ort = await this.loadRuntime());
    if (!this.wasm) {
      this.wasm = await step("runtime wasm", S, tick => getBin(base + "vendor/ort/ort-wasm-simd-threaded.wasm", tick));
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.proxy = false;
      ort.env.wasm.simd = true;
      ort.env.wasm.wasmBinary = this.wasm.buffer.slice(this.wasm.byteOffset, this.wasm.byteOffset + this.wasm.byteLength);
    }
    const mm = JSON.parse(await step("encoder manifest", S, tick => text(base + "models/glm/manifest.json", tick)));
    const v = mm.variants[encoder] ? encoder : mm.default;
    const vocab = await step("vocabulary", S, tick => text(base + "models/glm/" + mm.vocab, tick));
    const total = mm.variants[v].bytes;
    const bytes = await step("encoder download", S, async tick => {
      const bufs = []; let before = 0;
      for (const p of mm.variants[v].parts) {
        const b = await getBin(base + "models/glm/" + p, (d) => tick(before + Math.min(d || 0, total - before), total));
        bufs.push(b); before += b.length; tick(before, total);
      }
      const out = new Uint8Array(before);
      let o = 0;
      for (const b of bufs) { out.set(b, o); o += b.length; }
      if (out.length !== total) console.warn(`[glm] encoder bytes ${out.length} != manifest ${total}`);
      return out;
    });
    const t1 = now();
    const enc = await step("encoder session", S, () => new Encoder(ort, vocab).init(bytes, { executionProviders: ["wasm"] }));
    await step("warm-up", S, () => enc.tokens("warm up NE"));
    const prev = this.glm;
    this.glm = new GLM(enc);
    if (prev) for (const p of prev.packs.values()) this.glm.load(p);   // keep loaded packs across an encoder switch
    return { encoder: v, bytes: total, load_ms: now() - t0, session_ms: now() - t1 };
  }

  async load(id) {
    const t0 = now(), S = this.onStatus;
    const root = this.base + "packs/" + id + "/";
    const man = JSON.parse(await step(`pack ${id}: manifest`, S, tick => text(root + "manifest.json", tick)));
    if (man.kind === "wikidata-sharded") {     // general pack: small core now, shards on demand per question
      const get = p => getBytes(root + p, () => {});
      const pack = await step(`pack ${id}: files`, S, () => new WikiPack(man, get).init());
      this.glm.load(pack);
      return { id, title: man.title, counts: man.counts, relations: man.relations.length, bytes: pack.stats.bytes, load_ms: now() - t0, build_ms: pack.buildMs, lazy: true, total_bytes: man.sizes.total.bytes };
    }
    const f = man.files;
    const [ents, graph, weights, aliases] = await step(`pack ${id}: files`, S, tick => Promise.all([
      getBytes(root + f.entities.path, tick), getBin(root + f.graph.path, tick), getBin(root + f.weights.path, tick),
      f.aliases ? getBytes(root + f.aliases.path, tick) : null]));
    const td = new TextDecoder();
    const t1 = now();
    const ab = b => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
    const pack = await step(`pack ${id}: build`, S, () => new Pack(man, { entities: td.decode(ents), graph: ab(graph), weights: ab(weights), aliases: aliases && td.decode(aliases) }));
    this.glm.load(pack);
    const bytes = ents.length + graph.length + weights.length + (aliases ? aliases.length : 0);
    return { id, title: man.title, counts: man.counts, relations: man.relations.length, bytes, load_ms: now() - t0, build_ms: now() - t1 };
  }

  unload(id) { return this.glm ? this.glm.unload(id) : false; }
  ask(q, opts) { return this.glm.ask(q, opts); }
  pack(id) { return this.glm ? this.glm.packs.get(id) : null; }
}
