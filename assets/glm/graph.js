// Thinking graph: the local graph around the question's topic, drawn left to right on a Canvas 2D (no library).
// Column = steps from the topic. The chosen path lights up hop by hop, the reached answers glow, the next-best paths
// show in a second colour, taught facts (conversation pack) in a third, and the other neighbours that were looked at fade.
//   const g = new ThinkGraph(canvas, { onSelect(node) {...} }); g.show(viz); g.replay();

const css = (el, k) => getComputedStyle(el).getPropertyValue(k).trim();
const reduced = () => typeof matchMedia !== "undefined" && matchMedia("(prefers-reduced-motion: reduce)").matches;
const clip = (s, n = 22) => (s.length > n ? s.slice(0, n - 1) + "…" : s);
const T0 = 350, HOP = 650, FADE = 600;

export class ThinkGraph {
  constructor(canvas, { onSelect = () => {} } = {}) {
    this.c = canvas; this.ctx = canvas.getContext("2d"); this.onSelect = onSelect;
    this.sel = null; this.t0 = 0; this.raf = 0; this.viz = null;
    this.ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(() => this.resize()) : null;
    if (this.ro) this.ro.observe(canvas);
    canvas.addEventListener("click", e => this.pick(e));
    canvas.addEventListener("mousemove", e => { canvas.style.cursor = this.hit(e) ? "pointer" : "default"; });
    this.mq = typeof matchMedia !== "undefined" ? matchMedia("(prefers-color-scheme: dark)") : null;
    this.onTheme = () => this.draw();
    if (this.mq && this.mq.addEventListener) this.mq.addEventListener("change", this.onTheme);
  }

  destroy() {
    cancelAnimationFrame(this.raf);
    if (this.ro) this.ro.disconnect();
    if (this.mq && this.mq.removeEventListener) this.mq.removeEventListener("change", this.onTheme);
  }

  show(viz) {
    this.viz = viz; this.sel = null;
    const byId = this.byId = new Map(viz.nodes.map(n => [n.id, n]));
    this.edges = viz.edges.filter(e => byId.has(e.a) && byId.has(e.b));
    // when the chosen path reaches each node (ms from start)
    const k = Math.max(0, ...this.edges.filter(e => e.kind === "chosen").map(e => e.hop + 1));
    this.k = k; this.end = T0 + k * HOP;
    for (const n of viz.nodes) n.litAt = n.topic ? T0 : Infinity;
    for (const e of this.edges) if (e.kind === "chosen") { const b = byId.get(e.b); b.litAt = Math.min(b.litAt, T0 + (e.hop + 1) * HOP); }
    this.resize(true);
    this.replay();
  }

  replay() {
    cancelAnimationFrame(this.raf);
    this.t0 = performance.now();
    if (reduced()) { this.t = Infinity; this.draw(); return; }
    const tick = () => {
      this.t = performance.now() - this.t0;
      this.draw();
      if (this.t < this.end + FADE + 500) this.raf = requestAnimationFrame(tick);
      else { this.t = Infinity; this.draw(); }
    };
    tick();
  }

  resize(force = false) {
    const w = Math.max(240, Math.round(this.c.clientWidth || 320));
    const h = w < 520 ? 360 : 440;
    if (!force && w === this.w && h === this.h) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
    this.w = w; this.h = h; this.dpr = dpr;
    this.c.width = Math.round(w * dpr); this.c.height = Math.round(h * dpr);
    this.c.style.height = h + "px";
    if (this.viz) this.layout();
    this.draw();
  }

  // layered layout: x = steps from the topic; y order = kind band (alternatives above, chosen in the middle,
  // other neighbours below), then the mean y of each node's parents. Crowded columns zig-zag.
  layout() {
    const { w, h } = this, nodes = this.viz.nodes, padX = w < 520 ? 34 : 70, padY = 22;
    const maxD = Math.max(1, ...nodes.map(n => n.depth));
    const band = { alt: 0, chosen: 1, context: 2 };
    const parents = new Map();
    for (const e of this.edges) { const l = parents.get(e.b) || []; l.push(e.a); parents.set(e.b, l); }
    for (let d = 0; d <= maxD; d++) {
      const col = nodes.filter(n => n.depth === d);
      const bary = n => { const p = (parents.get(n.id) || []).map(id => this.byId.get(id)).filter(q => q && q.depth < d && q.y != null); return p.length ? p.reduce((a, q) => a + q.y, 0) / p.length : h / 2; };
      col.forEach(n => { n.b = bary(n); });
      col.sort((a, b) => band[a.kind] - band[b.kind] || (b.answer - a.answer) || a.b - b.b || a.id - b.id);
      const x = padX + (d / maxD) * (w - 2 * padX);
      const crowded = col.length * 13 > h - 2 * padY;
      col.forEach((n, i) => {
        n.x = x + (crowded && d > 0 ? (i % 2 ? 9 : -9) : 0);
        n.y = col.length === 1 ? h / 2 : padY + (i + 0.5) * ((h - 2 * padY) / col.length);
      });
    }
    // one label per relation group (path, hop): the middle edge of the group
    const groups = new Map();
    for (const e of this.edges) {
      const key = e.kind === "context" ? "c" + e.e : e.pi + ":" + e.hop;
      const l = groups.get(key) || []; l.push(e); groups.set(key, l);
    }
    for (const e of this.edges) e.lab = false;
    for (const l of groups.values()) { l.sort((a, b) => this.byId.get(a.b).y - this.byId.get(b.b).y); l[Math.floor(l.length / 2)].lab = true; }
    for (const e of this.edges) if (e.taught) e.lab = true;
  }

  colors() {
    const r = document.documentElement;
    return { bg: css(r, "--panel"), ink: css(r, "--ink"), muted: css(r, "--muted"), line: css(r, "--line"), chosen: css(r, "--accent"),
      alt: css(r, "--alt"), taught: css(r, "--taught"), edge: css(r, "--edge") };
  }

  bez(e, t) {
    const a = this.byId.get(e.a), b = this.byId.get(e.b);
    let dx = (b.x - a.x) / 2;
    if (Math.abs(dx) < 20) dx = 40;   // same column: bow out to the right
    const c1x = a.x + dx, c1y = a.y, c2x = b.x - (b.x - a.x >= 20 ? dx : -dx), c2y = b.y, u = 1 - t;
    return [u * u * u * a.x + 3 * u * u * t * c1x + 3 * u * t * t * c2x + t * t * t * b.x, u * u * u * a.y + 3 * u * u * t * c1y + 3 * u * t * t * c2y + t * t * t * b.y];
  }

  strokeEdge(e, upto) {
    const g = this.ctx, steps = 18, n = Math.max(1, Math.round(steps * upto));
    g.beginPath();
    let [x, y] = this.bez(e, 0); g.moveTo(x, y);
    for (let i = 1; i <= n; i++) { [x, y] = this.bez(e, Math.min(upto, i / steps)); g.lineTo(x, y); }
    g.stroke();
  }

  // node look at time t: {col, r, alpha, glow}
  look(n, t, C) {
    const lit = t >= n.litAt;
    const altOn = Math.min(1, Math.max(0, (t - this.end) / FADE));
    const fadeIn = Math.min(1, t / T0);
    if (n.kind === "chosen" && lit) {
      const r = n.topic ? 9 : n.answer ? 7 : 5;
      const glow = n.answer ? Math.min(1, (t - n.litAt) / 400) : n.topic ? 0.6 : 0;
      return { col: n.taught && !n.topic && !n.answer ? C.taught : C.chosen, r, alpha: 1, glow };
    }
    if (n.kind === "alt") return { col: altOn > 0 ? C.alt : C.muted, r: 4, alpha: 0.55 * fadeIn + 0.15 * altOn, glow: 0 };
    // visited but not chosen (and chosen nodes not yet reached)
    return { col: n.taught ? C.taught : C.muted, r: n.kind === "chosen" ? 4.5 : 3.5, alpha: fadeIn * (n.kind === "chosen" ? 0.55 : 0.55 - 0.35 * altOn), glow: 0 };
  }

  draw() {
    if (!this.viz || !this.w) return;
    const g = this.ctx, C = this.colors(), t = this.t ?? Infinity;
    g.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    g.clearRect(0, 0, this.w, this.h);
    const altOn = Math.min(1, Math.max(0, (t - this.end) / FADE)), fadeIn = Math.min(1, t / T0);
    const order = { context: 0, alt: 1, chosen: 2 };
    const es = [...this.edges].sort((a, b) => order[a.kind] - order[b.kind] || a.taught - b.taught);
    g.lineCap = "round";
    for (const e of es) {
      g.setLineDash([]);
      if (e.kind === "chosen") {
        const s = T0 + e.hop * HOP, p = Math.min(1, Math.max(0, (t - s) / (HOP - 50)));
        g.globalAlpha = 0.25 * fadeIn; g.strokeStyle = C.muted; g.lineWidth = 1; this.strokeEdge(e, 1);
        if (p > 0) { g.globalAlpha = 1; g.strokeStyle = e.taught ? C.taught : C.chosen; g.lineWidth = e.taught ? 3 : 2.2; this.strokeEdge(e, p); }
      } else if (e.kind === "alt") {
        g.globalAlpha = 0.22 * fadeIn + 0.38 * altOn; g.strokeStyle = e.taught ? C.taught : altOn > 0 ? C.alt : C.muted;
        g.lineWidth = 1.3; g.setLineDash([4, 3]); this.strokeEdge(e, 1);
      } else {
        g.globalAlpha = e.taught ? 0.9 : (0.3 - 0.18 * altOn) * fadeIn; g.strokeStyle = e.taught ? C.taught : C.muted;
        g.lineWidth = e.taught ? 2.4 : 1; this.strokeEdge(e, 1);
      }
    }
    g.setLineDash([]);
    // nodes
    const ns = [...this.viz.nodes].sort((a, b) => order[a.kind] - order[b.kind] || a.answer - b.answer);
    for (const n of ns) {
      const L = this.look(n, t, C);
      g.globalAlpha = L.alpha;
      if (L.glow > 0) { g.shadowColor = L.col; g.shadowBlur = 14 * L.glow; }
      g.fillStyle = L.col; g.beginPath(); g.arc(n.x, n.y, L.r, 0, 7); g.fill();
      g.shadowBlur = 0;
      if (n.isNew || (n.taught && n.kind !== "chosen")) { g.strokeStyle = C.taught; g.lineWidth = 2; g.beginPath(); g.arc(n.x, n.y, L.r + 3, 0, 7); g.stroke(); }
      if (n === this.sel) { g.globalAlpha = 1; g.strokeStyle = C.ink; g.lineWidth = 1.5; g.beginPath(); g.arc(n.x, n.y, L.r + 4, 0, 7); g.stroke(); }
    }
    g.globalAlpha = 1;
    // relation labels
    const small = this.w < 520;
    g.font = `500 ${small ? 10 : 11}px "IBM Plex Mono", ui-monospace, monospace`;
    g.textBaseline = "middle"; g.textAlign = "center";
    const boxes = [], hitBox = (x, y, w2, h2) => boxes.some(b => Math.abs(b[0] - x) < (b[2] + w2) / 2 + 2 && Math.abs(b[1] - y) < (b[3] + h2) / 2 + 1);
    for (const e of [...es].reverse()) {
      if (!e.lab) continue;
      let a, col;
      if (e.kind === "chosen") { const s = T0 + e.hop * HOP; a = t >= s + HOP / 2 ? 1 : 0; col = e.taught ? C.taught : C.edge; }
      else if (e.kind === "alt") { a = altOn * 0.85; col = e.taught ? C.taught : C.alt; }
      else { a = e.taught ? 1 : 0.55 * fadeIn * (1 - 0.5 * altOn); col = e.taught ? C.taught : C.muted; if (small && !e.taught) continue; }
      if (a <= 0) continue;
      const [x, y] = this.bez(e, 0.5), txt = e.taught ? e.label + " (taught)" : e.label, tw = g.measureText(txt).width;
      if (hitBox(x, y, tw + 6, 14)) continue;   // keep labels readable: skip one that would sit on another
      boxes.push([x, y, tw + 6, 14]);
      g.globalAlpha = a * 0.92; g.fillStyle = C.bg; g.fillRect(x - tw / 2 - 3, y - 7, tw + 6, 14);
      g.globalAlpha = a; g.fillStyle = col; g.fillText(txt, x, y);
    }
    // node labels: topic, answers, small chosen layers, taught nodes, the selected node
    const perLayer = new Map();
    for (const n of this.viz.nodes) if (n.kind === "chosen" && !n.topic) perLayer.set(n.depth, (perLayer.get(n.depth) || 0) + 1);
    let shownAns = 0;
    g.textAlign = "left";
    const labels = [];
    for (const n of [...this.viz.nodes].sort((a, b) => a.y - b.y)) {
      const lit = t >= n.litAt;
      let show = n === this.sel || n.topic || (n.isNew && (n.kind !== "chosen" || lit));
      if (!show && n.kind === "chosen" && lit) {
        if (n.answer) show = shownAns++ < (small ? 6 : 10) || n.taught;
        else show = (perLayer.get(n.depth) || 0) <= (small ? 3 : 5) || n.taught;
      }
      if (show) labels.push(n);
    }
    let lastY = -1e9, lastX = 0;
    for (const n of labels) {
      const L = this.look(n, t, C), right = n.x > this.w * 0.62;
      const txt = clip(n.name, small ? 18 : 26);
      g.font = `${n.topic || n.answer ? 600 : 400} ${small ? 11.5 : 12.5}px "IBM Plex Sans", system-ui, sans-serif`;
      const tw = g.measureText(txt).width;
      let x = right ? n.x - L.r - 5 - tw : n.x + L.r + 5, y = n.y;
      if (Math.abs(y - lastY) < 13 && Math.abs(x - lastX) < 60 && n !== this.sel && !n.topic) continue;   // would overlap
      if (n.topic) { x = Math.max(4, Math.min(n.x - tw / 2, this.w - tw - 4)); y = n.y - L.r - 10; }
      lastY = n.y; lastX = x;
      g.globalAlpha = 1; g.lineWidth = 3.5; g.strokeStyle = C.bg; g.lineJoin = "round"; g.strokeText(txt, x, y);
      g.fillStyle = n.taught || n.isNew ? C.taught : n.kind === "chosen" ? C.ink : C.muted; g.fillText(txt, x, y);
    }
    g.globalAlpha = 1;
  }

  hit(ev) {
    if (!this.viz) return null;
    const r = this.c.getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top;
    let best = null, bd = 18 * 18;
    for (const n of this.viz.nodes) { const d = (n.x - x) ** 2 + (n.y - y) ** 2; if (d < bd) { bd = d; best = n; } }
    return best;
  }

  pick(ev) {
    const n = this.hit(ev);
    if (!n) return;
    this.sel = n; this.draw(); this.onSelect(n);
  }

  select(id) { const n = this.byId && this.byId.get(id); if (n) { this.sel = n; this.draw(); } }
}
