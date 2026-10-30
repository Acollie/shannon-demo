// GLM Pocket "About me": short facts the user saves about themselves. Stored only in this browser (localStorage);
// nothing is sent anywhere. Every storage call is wrapped: private windows or blocked storage just mean the list
// lives for this tab only.
const KEY = "glm-pocket-about-me-v1", MAX = 50, MAXLEN = 200;

export class Memory {
  constructor() { this.items = this.read(); this.vecs = new Map(); this.durable = this.probe(); }
  probe() { try { const k = KEY + "-probe"; localStorage.setItem(k, "1"); localStorage.removeItem(k); return true; } catch { return false; } }
  read() {
    try { const v = JSON.parse(localStorage.getItem(KEY) || "[]"); return Array.isArray(v) ? v.filter(x => x && typeof x.text === "string" && x.id) : []; }
    catch { return []; }
  }
  save() { try { localStorage.setItem(KEY, JSON.stringify(this.items)); } catch { /* blocked or full: kept for this tab */ } }
  list() { return this.items.slice(); }
  add(text) {
    const t = String(text || "").replace(/\s+/g, " ").trim().slice(0, MAXLEN);
    if (!t || this.items.some(x => x.text.toLowerCase() === t.toLowerCase())) return null;
    const it = { id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6), text: t, t: Date.now() };
    this.items.unshift(it); this.items = this.items.slice(0, MAX); this.save();
    return it;
  }
  remove(id) { this.items = this.items.filter(x => x.id !== id); this.vecs.delete(id); this.save(); }
  forget() { this.items = []; this.vecs.clear(); try { localStorage.removeItem(KEY); } catch { /* blocked */ } }

  // Facts relevant to a message: all of them when there are few, else the top k by MiniLM cosine similarity
  // (the GLM engine's own encoder; sentence vectors are L2-normalised so the dot product is the cosine).
  async relevant(message, enc, k = 5) {
    if (this.items.length <= k || !enc) return this.items.slice(0, k);
    const miss = this.items.filter(x => !this.vecs.has(x.id));
    if (miss.length) (await enc.sent(miss.map(x => x.text))).forEach((v, i) => this.vecs.set(miss[i].id, v));
    const [q] = await enc.sent([message]);
    const dot = v => { let s = 0; for (let i = 0; i < v.length; i++) s += v[i] * q[i]; return s; };
    return this.items.map(x => [x, dot(this.vecs.get(x.id))]).sort((a, b) => b[1] - a[1]).slice(0, k).map(([x]) => x);
  }
}

// A message that states a fact about the user ("My name is Sam", "I live in Leeds", "I'm vegetarian"): the page offers to
// save it (the user confirms). Questions are never offered.
export function suggestFact(text) {
  const t = text.trim();
  if (!t || t.length > MAXLEN || /\?\s*$/.test(t)) return null;
  if (/^(my\s+\w+(\s+\w+)?\s+(is|are|was)\b|i\s+(am|'m|live|work|like|love|prefer|have|hate|don't|do not|was born|study|speak)\b|i'm\b|call me\b)/i.test(t)) return t.replace(/[.!\s]+$/, "");
  return null;
}
