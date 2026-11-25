'use strict';
/* ---------------------------------------------------------------- palette
   Drawing-sheet identity: whiteprint (navy ink on cool white) and blueprint (pale ink on Prussian blue).
   Each planet keeps its own line pattern and glyph as well as its colour, so no information rests on colour. */
const PALS = {
  white: {ink:'#14284b', ink2:'#3d5272', muted:'#5f708a', grid:'#dde4ec', axis:'#14284b', surface:'#fbfcfd', surface2:'#eef2f6',
    dot:'rgba(20,40,75,.30)', rule:'#a8b7c9', ochre:'#946800', carmine:'#cf3a2c', band:'rgba(201,141,0,.16)', halo:'rgba(251,252,253,.94)',
    Earth:'#1f63c6', Mars:'#c4511c', Moon:'#646b74', wrist:'#0d7f70', pelvis:'#97368f', thigh:'#97368f', deck:'#3d5273', rigid:'#97a3b3'},
  blue: {ink:'#eaf2ff', ink2:'#bcd0ec', muted:'#8fa9cf', grid:'#2a5590', axis:'#eaf2ff', surface:'#123766', surface2:'#174279',
    dot:'rgba(234,242,255,.35)', rule:'#5b80b3', ochre:'#ffd35a', carmine:'#ff8f78', band:'rgba(255,211,90,.18)', halo:'rgba(18,55,102,.92)',
    Earth:'#8fd3ff', Mars:'#ffb27a', Moon:'#c9d3df', wrist:'#7ff0d8', pelvis:'#ffa8f0', thigh:'#ffa8f0', deck:'#bcd0ed', rigid:'#7f97b8'}};
const PAL = Object.assign({}, PALS.white);
const FONT = '"B612","DejaVu Sans",Arial,sans-serif';
const MONO = '"B612","DejaVu Sans",Arial,sans-serif';   // numbers in the proportional face: its mono face spaces decimal points widely
const PLANETS = ['Earth', 'Mars', 'Moon'];
const PDASH = {Earth: [], Mars: [9, 2.6], Moon: [1.8, 2.6, 7, 2.6]};
const REFDASH = [[7, 3], [2, 2.5], [12, 3, 2, 3]];   // reference lines in bar charts, by order
const CHARTS = [];

/* planet glyphs: Earth circled cross, Mars circle with arrow, Moon crescent */
function glyphPath(c, planet, x, y, r){
  c.beginPath();
  if (planet === 'Earth'){ c.arc(x, y, r, 0, 2 * Math.PI); c.moveTo(x, y - r); c.lineTo(x, y + r); c.moveTo(x - r, y); c.lineTo(x + r, y); }
  else if (planet === 'Mars'){ const rr = r * 0.78, cx = x - r * 0.2, cy = y + r * 0.2, s = Math.SQRT1_2;
    c.arc(cx, cy, rr, 0, 2 * Math.PI); const ax = cx + rr * s, ay = cy - rr * s, tx = x + r * 1.05, ty = y - r * 1.05;
    c.moveTo(ax, ay); c.lineTo(tx, ty); c.moveTo(tx - r * 0.62, ty); c.lineTo(tx, ty); c.lineTo(tx, ty + r * 0.62); }
  else { const d = 0.45 * r, rr = 0.85 * r, xi = (r * r - rr * rr + d * d) / (2 * d), a = Math.atan2(Math.sqrt(r * r - xi * xi), xi),
    b = Math.atan2(Math.sqrt(r * r - xi * xi), xi - d), ox = x - 0.12 * r;
    c.arc(ox, y, r, a, 2 * Math.PI - a, false); c.arc(ox + d, y, rr, 2 * Math.PI - b, b, true); c.closePath(); }
}
function drawGlyph(c, planet, x, y, r, color, fillBg){
  c.save(); c.setLineDash([]); c.lineWidth = Math.max(1, r * 0.32); c.strokeStyle = color; c.fillStyle = color;
  if (planet === 'Moon'){ glyphPath(c, planet, x, y, r); c.fill(); }
  else { if (fillBg){ c.beginPath(); c.arc(x - (planet === 'Mars' ? r * 0.2 : 0), y + (planet === 'Mars' ? r * 0.2 : 0), r * (planet === 'Mars' ? 0.78 : 1), 0, 7); c.fillStyle = fillBg; c.fill(); }
    glyphPath(c, planet, x, y, r); c.stroke(); }
  c.restore();
}
function glyphSVG(planet, size, color){
  const col = color || ('var(--' + planet.toLowerCase() + ')'), s = size || 15;
  const st = (f, w) => ' style="fill:' + f + ';stroke:' + (w ? col : 'none') + (w ? ';stroke-width:' + w : '') + '"';
  const body = planet === 'Earth' ? '<circle cx="8" cy="8" r="6"' + st('none', 1.5) + '/><path d="M8 2v12M2 8h12"' + st('none', 1.5) + '/>'
    : planet === 'Mars' ? '<circle cx="6.6" cy="9.4" r="4.6"' + st('none', 1.5) + '/><path d="M9.9 6.1L14.2 1.8M10.4 1.8h3.8v3.8"' + st('none', 1.5) + '/>'
    : '<path class="fill" d="M10.2 2.3A6 6 0 1 0 10.2 13.7A4.9 4.9 0 1 1 10.2 2.3Z"' + st(col, 0) + '/>';
  return '<svg class="pl" viewBox="0 0 16 16" width="' + s + '" height="' + s + '" aria-hidden="true">' + body + '</svg>';
}
function planetOf(s){
  if (s.planet) return s.planet;
  for (const p of PLANETS) if (s.color === PAL[p]) return p;
  return null;
}
const PATTERNS = {};
function hatch(c, planet, color){
  const key = planet + color + PAL.surface; if (PATTERNS[key]) return PATTERNS[key];
  const cv = document.createElement('canvas'); cv.width = cv.height = 8; const g = cv.getContext('2d');
  if (planet === 'Mars'){ g.fillStyle = PAL.surface; g.fillRect(0, 0, 8, 8); g.strokeStyle = color; g.lineWidth = 1.6;
    g.beginPath(); g.moveTo(-2, 10); g.lineTo(10, -2); g.moveTo(-2, 2); g.lineTo(2, -2); g.moveTo(6, 10); g.lineTo(10, 6); g.stroke(); }
  else if (planet === 'Moon'){ g.fillStyle = PAL.surface; g.fillRect(0, 0, 8, 8); g.fillStyle = color; g.beginPath(); g.arc(2, 2, 1.25, 0, 7); g.arc(6, 6, 1.25, 0, 7); g.fill(); }
  else { g.fillStyle = color; g.fillRect(0, 0, 8, 8); }
  return (PATTERNS[key] = c.createPattern(cv, 'repeat'));
}

/* ----------------------------------------------------------------- helpers */
function el(tag, attrs, kids){
  const e = document.createElement(tag);
  if (attrs) for (const k in attrs){
    if (k === 'class') e.className = attrs[k];
    else if (k === 'html') e.innerHTML = attrs[k];
    else if (k === 'text') e.textContent = attrs[k];
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), attrs[k]);
    else e.setAttribute(k, attrs[k]);
  }
  (kids || []).forEach(c => c && e.appendChild(typeof c === 'string' ? document.createTextNode(c) : c));
  return e;
}
function fmt(v, sig){
  if (v === null || v === undefined || !isFinite(v)) return '–';
  sig = sig || 3;
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a >= 1e5 || a < 1e-3) return v.toExponential(Math.max(0, sig - 1)).replace('e+', 'e').replace(/(\.\d*?)0+e/, '$1e').replace(/\.e/, 'e');
  const d = Math.max(0, sig - 1 - Math.floor(Math.log10(a)));
  return v.toFixed(Math.min(d, 6));
}
function pct(v, d){ return (v === null || v === undefined || !isFinite(v)) ? '–' : (100 * v).toFixed(d === undefined ? 0 : d) + ' %'; }
function niceStep(range, n){
  const raw = range / Math.max(1, n), p = Math.pow(10, Math.floor(Math.log10(raw))), r = raw / p;
  return (r < 1.5 ? 1 : r < 3.5 ? 2 : r < 7.5 ? 5 : 10) * p;
}
function linTicks(lo, hi, n){
  const st = niceStep(hi - lo, n), out = [];
  for (let v = Math.ceil(lo / st - 1e-9) * st; v <= hi + st * 1e-9; v += st) out.push(Math.abs(v) < st * 1e-9 ? 0 : v);
  return out;
}
function logTicks(lo, hi){
  if (!(hi > 0) || !isFinite(hi)) hi = 1;
  if (!(lo > 0) || !isFinite(lo) || lo >= hi) lo = hi / 1e6;
  const out = [], a = Math.floor(Math.log10(lo)), b = Math.ceil(Math.log10(hi)), few = (b - a) <= 3;
  for (let e = a; e <= b; e++) for (const m of (few ? [1, 2, 5] : [1])){ const v = m * Math.pow(10, e); if (v >= lo * 0.999 && v <= hi * 1.001) out.push(v); }
  return out;
}
function tickLabel(v, log, step){
  if (log){ if (v >= 0.01 && v < 10000) return String(+v.toPrecision(3)); const e = Math.round(Math.log10(v)), m = v / Math.pow(10, e); return (Math.abs(m - 1) < 1e-6 ? '' : fmt(m, 1) + '×') + '10' + sup(e); }
  if (step){ const d = Math.max(0, Math.min(8, Math.ceil(-Math.log10(step) - 1e-9))); return v.toFixed(d); }
  return String(+v.toPrecision(6));
}
function sup(n){ const m = {'-':'⁻','0':'⁰','1':'¹','2':'²','3':'³','4':'⁴','5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹'}; return String(n).split('').map(c => m[c]).join(''); }
function download(name, href){ const a = el('a', {href: href, download: name}); document.body.appendChild(a); a.click(); a.remove(); }
function showTip(html, x, y){
  const t = document.getElementById('tip'); t.innerHTML = html; t.hidden = false;
  const r = t.getBoundingClientRect(); let px = x + 14, py = y + 14;
  if (px + r.width > innerWidth - 8) px = x - r.width - 14;
  if (py + r.height > innerHeight - 8) py = y - r.height - 14;
  t.style.left = Math.max(4, px) + 'px'; t.style.top = Math.max(4, py) + 'px';
}
function hideTip(){ document.getElementById('tip').hidden = true; }
function swatch(color){ const p = PLANETS.find(q => PAL[q] === color); return p ? glyphSVG(p, 11).replace('class="pl"', 'style="vertical-align:-1px;margin-right:6px"') : '<i style="background:' + color + '"></i>'; }

/* a card that holds one chart, with "save image" and "save data" buttons */
function makeCard(parent, o){
  const tools = el('div', {class: 'tools'});
  const head = el('div', {class: 'head'}, [el('div', null, [el('div', {class: 'ttl', text: o.title || ''}),
    o.desc ? el('div', {class: 'desc', html: o.desc}) : null]), tools]);
  const opts = el('div', {class: 'opts'});
  const canvas = el('canvas');
  canvas.style.height = (o.height || 260) + 'px';
  const note = o.note ? el('div', {class: 'note', html: o.note}) : null;
  const card = el('div', {class: 'card'}, [head, opts, canvas, note]);
  parent.appendChild(card);
  return {card, head, tools, opts, canvas, note, title: o.title || '', desc: (o.desc || '').replace(/<[^>]+>/g, ''), file: o.file || 'figure'};
}
function segControl(parent, label, items, value, onChange){
  const seg = el('div', {class: 'seg', role: 'radiogroup'});
  if (label) seg.setAttribute('aria-label', label);
  const set = v => { [...seg.children].forEach(b => { const on = b.dataset.v === String(v); b.classList.toggle('on', on); b.setAttribute('aria-checked', on ? 'true' : 'false'); }); };
  items.forEach(it => seg.appendChild(el('button', {'data-v': it.v, role: 'radio', html: it.html || it.label, title: it.title || '',
    onclick: e => { if (e.currentTarget.disabled) return; set(it.v); onChange(it.v); }})));
  set(value);
  const wrap = label ? el('div', {class: 'ctl'}, [el('span', {text: label}), seg]) : seg;
  parent.appendChild(wrap);
  return {seg, set, enable: (v, on) => { const b = [...seg.children].find(x => x.dataset.v === String(v)); if (b) b.disabled = !on; }};
}

/* ------------------------------------------------------------------ chart */
class Chart {
  constructor(card, o){
    this.card = card; this.cv = card.canvas; this.o = o || {}; this.hidden = {}; this.hits = []; this.hover = null;
    card.tools.appendChild(el('button', {class: 'btn small', text: 'Save image', title: 'Download this chart as a PNG picture', onclick: () => this.savePNG()}));
    card.tools.appendChild(el('button', {class: 'btn small', text: 'Save data', title: 'Download the plotted numbers as a CSV table', onclick: () => this.saveCSV()}));
    this.cv.addEventListener('mousemove', e => this.onMove(e));
    this.cv.addEventListener('mouseleave', () => { this.hover = null; hideTip(); this.draw(); });
    this.cv.addEventListener('click', e => this.onClick(e));
    CHARTS.push(this);
  }
  set(o){ Object.assign(this.o, o); this.draw(); }
  size(){ const r = this.cv.getBoundingClientRect(); return [Math.max(120, r.width), Math.max(80, r.height)]; }
  draw(){
    const [w, h] = this.size(); if (!this.cv.offsetParent) return;
    const dpr = window.devicePixelRatio || 1;
    if (this.cv.width !== Math.round(w * dpr) || this.cv.height !== Math.round(h * dpr)){ this.cv.width = Math.round(w * dpr); this.cv.height = Math.round(h * dpr); }
    const c = this.cv.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.render(c, w, h, true);
  }
  vis(){ return (this.o.series || []).filter(s => !this.hidden[s.label]); }
  /* legend row(s) at the top; returns the height used */
  legend(c, w, live){
    const items = (this.o.series || []).filter(s => s.label && !s.noLegend);
    // bars fill the plot, so a bar chart names its reference lines in the key, not on the line
    const refs = this.o.type === 'bar' ? (this.o.refLines || []).filter(r => r.label) : [];
    if (this.o.legend === false || items.length + refs.length < 1) return 4;
    c.font = '12px ' + FONT; c.textBaseline = 'middle'; c.textAlign = 'left';
    let x = 8, y = 12; if (live) this.hits = [];
    for (const s of items){
      const tw = c.measureText(s.label).width + 40;
      if (x + tw > w - 6 && x > 8){ x = 8; y += 19; }
      const off = !!this.hidden[s.label], pl = planetOf(s);
      c.globalAlpha = off ? 0.3 : 1;
      c.strokeStyle = s.color; c.fillStyle = s.color; c.lineWidth = Math.min(s.width || 1.6, 2); c.setLineDash(this.dashOf(s));
      if (this.o.type === 'bar'){ c.fillStyle = pl ? hatch(c, pl, s.color) : s.color; c.fillRect(x, y - 5, 13, 10); c.strokeStyle = s.color; c.lineWidth = 1; c.setLineDash([]); c.strokeRect(x + 0.5, y - 4.5, 12, 9); }
      else { c.beginPath(); c.moveTo(x, y); c.lineTo(x + 24, y); c.stroke(); if (pl) drawGlyph(c, pl, x + 12, y, 4.2, s.color, PAL.surface); else if (s.marker){ c.setLineDash([]); c.fillStyle = PAL.surface; c.fillRect(x + 9, y - 3, 6, 6); c.lineWidth = 1.2; c.strokeRect(x + 9, y - 3, 6, 6); } }
      c.setLineDash([]); c.fillStyle = off ? PAL.muted : PAL.ink; c.fillText(s.label, x + (this.o.type === 'bar' ? 18 : 29), y + 0.5);
      c.globalAlpha = 1;
      if (live) this.hits.push({x0: x, x1: x + tw, y0: y - 9, y1: y + 9, label: s.label});
      x += tw + 6;
    }
    refs.forEach((r, k) => {
      const tw = c.measureText(r.label).width + 40;
      if (x + tw > w - 6 && x > 8){ x = 8; y += 19; }
      c.strokeStyle = r.color || PAL.ochre; c.lineWidth = 1.2; c.setLineDash(REFDASH[k % REFDASH.length]); const yy = Math.round(y) + 0.5; c.beginPath(); c.moveTo(x, yy); c.lineTo(x + 24, yy); c.stroke(); c.setLineDash([]);
      c.fillStyle = PAL.ink; c.fillText(r.label, x + 29, y + 0.5);
      x += tw + 6;
    });
    return y + 14;
  }
  dashOf(s){ if (s.dash) return s.dash; const p = planetOf(s); return p ? PDASH[p] : []; }
  /* open frame: hairline grid at the ticks, axis lines on the left and below, short outward ticks */
  frame(c, L, top, pw, ph, xs, ys){
    c.save(); c.setLineDash([]);
    c.strokeStyle = PAL.grid; c.lineWidth = 1; c.beginPath();
    for (const x of xs){ if (x <= L + 1 || x >= L + pw - 1) continue; c.moveTo(x, top); c.lineTo(x, top + ph); }
    for (const y of ys){ if (y <= top + 1 || y >= top + ph - 1) continue; c.moveTo(L, y); c.lineTo(L + pw, y); }
    c.stroke();
    c.strokeStyle = PAL.ink; c.lineWidth = 1.4; c.beginPath();
    c.moveTo(L + 0.5, top); c.lineTo(L + 0.5, top + ph + 0.5); c.lineTo(L + pw, top + ph + 0.5); c.stroke();
    c.lineWidth = 1; c.beginPath();
    for (const x of xs){ c.moveTo(x, top + ph + 1); c.lineTo(x, top + ph + 5); }
    for (const y of ys){ c.moveTo(L - 1, y); c.lineTo(L - 5, y); }
    c.stroke(); c.restore();
  }
  render(c, w, h, live){
    c.clearRect(0, 0, w, h); c.fillStyle = PAL.surface; c.fillRect(0, 0, w, h);
    const o = this.o, top = this.legend(c, w, live) + (o.guides && o.guides.length ? 14 : 2);
    const two = (o.cats || o.xCats || []).some(c => String(c).indexOf('\n') >= 0 && String(c).split('\n')[1] !== '');
    const L = o.left || 58, R = o.right || 14, B = (o.xLabel ? 42 : 26) + (two ? 14 : 0), pw = w - L - R, ph = h - top - B;
    this.box = {L, R, top, B, pw, ph, w, h};
    if (o.type === 'bar') return this.renderBars(c, live);
    const S = this.vis();
    // ranges
    let x0 = o.xMin, x1 = o.xMax, y0 = o.yMin, y1 = o.yMax;
    if (x0 === undefined || x1 === undefined || y0 === undefined || y1 === undefined){
      let ax = Infinity, bx = -Infinity, ay = Infinity, by = -Infinity;
      for (const s of S) for (let i = 0; i < s.x.length; i++){
        const xv = s.x[i], yv = s.y[i]; if (!isFinite(yv) || (o.yLog && yv <= 0)) continue;
        if ((o.xMin !== undefined && xv < o.xMin) || (o.xMax !== undefined && xv > o.xMax)) continue;
        if (xv < ax) ax = xv; if (xv > bx) bx = xv; if (yv < ay) ay = yv; if (yv > by) by = yv;
      }
      if (!isFinite(ax)){ ax = 0; bx = 1; ay = o.yLog ? 1e-6 : 0; by = 1; }
      if (x0 === undefined) x0 = ax; if (x1 === undefined) x1 = bx;
      if (o.yLog){ if (y0 === undefined) y0 = Math.pow(10, Math.floor(Math.log10(ay))); if (y1 === undefined) y1 = Math.pow(10, Math.ceil(Math.log10(by))); }
      else { const pad = (by - ay) * 0.08 || 1; if (y0 === undefined) y0 = (o.yZero && ay >= 0) ? 0 : ay - pad; if (y1 === undefined) y1 = by + pad; }
    }
    const lx = v => o.xLog ? Math.log10(v) : v, ly = v => o.yLog ? Math.log10(Math.max(v, 1e-300)) : v;
    const X = v => L + (lx(v) - lx(x0)) / (lx(x1) - lx(x0)) * pw, Y = v => top + ph - (ly(v) - ly(y0)) / (ly(y1) - ly(y0)) * ph;
    this.map = {X, Y, x0, x1, y0, y1, invX: px => { const u = lx(x0) + (px - L) / pw * (lx(x1) - lx(x0)); return o.xLog ? Math.pow(10, u) : u; }};
    // frame, ticks and dots
    c.font = '11px ' + MONO; c.lineWidth = 1;
    const yt = o.yLog ? logTicks(y0, y1) : linTicks(y0, y1, Math.max(3, Math.round(ph / 46)));
    c.textAlign = 'right'; c.textBaseline = 'middle';
    const ys = yt.length > 1 ? Math.abs(yt[1] - yt[0]) : 1, ypx = [], xpx = [];
    for (const v of yt){ const y = Math.round(Y(v)) + 0.5; ypx.push(y); c.fillStyle = PAL.ink; c.fillText(tickLabel(v, o.yLog, ys), L - 7, y); }
    c.textAlign = 'center'; c.textBaseline = 'top';
    if (o.xCats){ c.font = '12px ' + FONT; o.xCats.forEach((lab, i) => { xpx.push(Math.round(X(i)) + 0.5); c.fillStyle = PAL.ink; c.fillText(lab, X(i), top + ph + 7); }); }
    else {
      const xt = o.xLog ? logTicks(x0, x1) : linTicks(x0, x1, Math.max(3, Math.round(pw / 80)));
      const xs = xt.length > 1 ? Math.abs(xt[1] - xt[0]) : 1;
      for (const v of xt){ const x = Math.round(X(v)) + 0.5; xpx.push(x); c.fillStyle = PAL.ink; c.fillText(tickLabel(v, o.xLog, xs), x, top + ph + 7); }
    }
    this.frame(c, L, top, pw, ph, xpx, ypx);
    c.fillStyle = PAL.ink2; c.font = '12px ' + FONT;
    if (o.xLabel){ c.textAlign = 'center'; c.textBaseline = 'bottom'; c.fillText(o.xLabel, L + pw / 2, h - 4); }
    if (o.yLabel){ c.save(); c.translate(13, top + ph / 2); c.rotate(-Math.PI / 2); c.textAlign = 'center'; c.textBaseline = 'middle'; c.fillText(o.yLabel, 0, 0); c.restore(); }
    // shaded bands and vertical guides
    c.save(); c.beginPath(); c.rect(L, top - 14, pw, ph + 14); c.clip();
    for (const b of (o.bands || [])){ c.fillStyle = b.color || PAL.band; c.fillRect(X(b.x0), top, X(b.x1) - X(b.x0), ph); }
    c.font = '10px ' + MONO; c.textBaseline = 'bottom';
    let lastLab = -1e9;
    for (const g of (o.guides || [])){
      if (g.x < x0 || g.x > x1) continue; const x = Math.round(X(g.x)) + 0.5;
      c.strokeStyle = g.color || PAL.muted; c.lineWidth = 1; c.setLineDash(g.dash || [1, 3]); c.beginPath(); c.moveTo(x, top); c.lineTo(x, top + ph); c.stroke(); c.setLineDash([]);
      if (g.label && x - lastLab > c.measureText(g.label).width + 4){ c.fillStyle = g.color || PAL.ink2; c.textAlign = 'center'; c.fillText(g.label, x, top - 2); lastLab = x; }
    }
    c.restore();
    for (const r of (o.refLines || [])){ const y = Math.round(Y(r.y)) + 0.5; c.strokeStyle = r.color || PAL.ochre; c.lineWidth = 1.2; c.setLineDash([7, 3]); c.beginPath(); c.moveTo(L, y); c.lineTo(L + pw, y); c.stroke(); c.setLineDash([]);
      // the right end belongs to the line-end labels when they are shown
      if (r.label){ const left = r.pos === 'left' || (o.endLabels && r.pos !== 'right'), lx = left ? L + 6 : L + pw - 4;
        c.fillStyle = r.color || PAL.ochre; c.font = '10.5px ' + MONO; c.textAlign = left ? 'left' : 'right'; c.textBaseline = 'bottom'; c.lineWidth = 3; c.strokeStyle = PAL.halo; c.strokeText(r.label, lx, y - 2); c.fillText(r.label, lx, y - 2); } }
    // series
    c.save(); c.beginPath(); c.rect(L, top, pw, ph); c.clip();
    for (const s of S){
      c.strokeStyle = s.color; c.fillStyle = s.color; c.lineWidth = Math.min(s.width || 1.6, 2.2); c.setLineDash(this.dashOf(s)); c.lineJoin = 'miter'; c.lineCap = 'butt'; c.globalAlpha = s.alpha || 1;
      const n = s.x.length; c.beginPath(); let pen = false;
      if (n > 3 * pw && !o.xLog){        // min/max per pixel column
        let col = null, mn = 0, mx = 0, first = 0, last = 0;
        const flush = () => { if (col === null) return; const xx = L + col; if (!pen){ c.moveTo(xx, Y(first)); pen = true; } else c.lineTo(xx, Y(first)); c.lineTo(xx, Y(mn)); c.lineTo(xx, Y(mx)); c.lineTo(xx, Y(last)); };
        for (let i = 0; i < n; i++){ const xv = s.x[i]; if (xv < x0 || xv > x1) continue; const yv = s.y[i]; if (!isFinite(yv)) continue;
          const cc = Math.round(X(xv) - L); if (cc !== col){ flush(); col = cc; mn = mx = first = yv; } if (yv < mn) mn = yv; if (yv > mx) mx = yv; last = yv; }
        flush();
      } else {
        for (let i = 0; i < n; i++){ const yv = s.y[i]; if (!isFinite(yv) || (o.yLog && yv <= 0)){ pen = false; continue; } const xx = X(s.x[i]), yy = Y(yv); if (!pen){ c.moveTo(xx, yy); pen = true; } else c.lineTo(xx, yy); }
      }
      if (!s.noLine) c.stroke(); c.setLineDash([]);
      if (s.marker){ const pl = planetOf(s); for (let i = 0; i < n; i++){ if (!isFinite(s.y[i])) continue; const xx = X(s.x[i]), yy = Y(s.y[i]);
        if (pl) drawGlyph(c, pl, xx, yy, s.marker + 1.4, s.color, PAL.surface);
        else { c.setLineDash([]); c.fillStyle = PAL.surface; c.lineWidth = 1.3; c.fillRect(xx - s.marker, yy - s.marker, 2 * s.marker, 2 * s.marker); c.strokeRect(xx - s.marker, yy - s.marker, 2 * s.marker, 2 * s.marker); } } }
      c.globalAlpha = 1;
    }
    c.restore();
    // labels at the right end of each line
    if (o.endLabels){ c.font = '12px ' + FONT; c.textAlign = 'left'; c.textBaseline = 'middle'; const labs = [], GAP = 14;
      for (const s of S){ let i = s.x.length - 1; while (i > 0 && !isFinite(s.y[i])) i--; labs.push({s, x: X(s.x[i]), y0: Y(s.y[i]), y: Y(s.y[i])}); }
      labs.sort((a, b) => a.y0 - b.y0);
      for (let k = 1; k < labs.length; k++) if (labs[k].y - labs[k - 1].y < GAP) labs[k].y = labs[k - 1].y + GAP;
      const over = labs.length ? labs[labs.length - 1].y - (top + ph - 4) : 0; if (over > 0) labs.forEach(l => { l.y -= over; });
      for (let k = labs.length - 2; k >= 0; k--) if (labs[k + 1].y - labs[k].y < GAP) labs[k].y = labs[k + 1].y - GAP;
      for (const l of labs){ c.fillStyle = l.s.color;
        if (Math.abs(l.y - l.y0) > 2){ c.strokeStyle = l.s.color; c.lineWidth = 0.8; c.setLineDash([]); c.beginPath(); c.moveTo(l.x + 2, l.y0); c.lineTo(l.x + 9, l.y); c.stroke(); }
        const t = l.s.endLabel || l.s.label; c.lineWidth = 3; c.strokeStyle = PAL.halo; c.strokeText(t, l.x + 11, l.y); c.fillText(t, l.x + 11, l.y); } }
    if (o.cursorX !== undefined && o.cursorX >= x0 && o.cursorX <= x1){ const x = Math.round(X(o.cursorX)) + 0.5; c.strokeStyle = PAL.carmine; c.lineWidth = 1.2; c.setLineDash([]); c.beginPath(); c.moveTo(x, top); c.lineTo(x, top + ph); c.stroke();
      c.fillStyle = PAL.carmine; c.beginPath(); c.moveTo(x - 4, top); c.lineTo(x + 4, top); c.lineTo(x, top + 5); c.closePath(); c.fill(); }
    if (live && this.hover){ const x = Math.round(X(this.hover.x)) + 0.5; c.strokeStyle = PAL.ink; c.lineWidth = 1; c.setLineDash([2, 2]); c.beginPath(); c.moveTo(x, top); c.lineTo(x, top + ph); c.stroke(); c.setLineDash([]);
      for (const p of this.hover.pts){ const pl = PLANETS.find(q => PAL[q] === p.color); if (pl) drawGlyph(c, pl, X(p.x), Y(p.y), 5.2, p.color, PAL.surface);
        else { c.fillStyle = PAL.surface; c.strokeStyle = p.color; c.lineWidth = 1.6; c.fillRect(X(p.x) - 4, Y(p.y) - 4, 8, 8); c.strokeRect(X(p.x) - 4, Y(p.y) - 4, 8, 8); } } }
  }
  renderBars(c, live){
    const o = this.o, {L, top, pw, ph, h} = this.box, S = this.vis(), cats = o.cats || [];
    let y1 = o.yMax; if (y1 === undefined){ y1 = 0; for (const s of S) for (const v of s.values) if (isFinite(v) && v > y1) y1 = v; for (const r of (o.refLines || [])) if (r.y > y1) y1 = r.y; y1 = y1 * 1.12 || 1; }
    let y0 = o.yMin; if (y0 === undefined){ y0 = 0; for (const s of S) for (const v of s.values) if (isFinite(v) && v < y0) y0 = v; y0 = y0 < 0 ? y0 * 1.15 : 0; }
    const Y = v => top + ph - (v - y0) / (y1 - y0) * ph, Y0 = Y(Math.max(0, y0));
    c.font = '11px ' + MONO; c.lineWidth = 1; c.textAlign = 'right'; c.textBaseline = 'middle';
    const bt = linTicks(y0, y1, Math.max(3, Math.round(ph / 46))), bs = bt.length > 1 ? bt[1] - bt[0] : 1, ypx = [];
    for (const v of bt){ const y = Math.round(Y(v)) + 0.5; ypx.push(y); c.fillStyle = PAL.ink; c.fillText(tickLabel(v, false, bs), L - 7, y); }
    const gw = pw / Math.max(1, cats.length), bw = Math.min(34, (gw * 0.78) / Math.max(1, S.length)) - 2;
    this.frame(c, L, top, pw, ph, cats.map((_, i) => Math.round(L + gw * (i + 0.5)) + 0.5), ypx);
    this.bars = [];
    cats.forEach((cat, i) => {
      const cx = L + gw * (i + 0.5), tot = S.length * (bw + 2) - 2;
      S.forEach((s, j) => { const v = s.values[i]; if (v === null || v === undefined || !isFinite(v)) return; const x = cx - tot / 2 + j * (bw + 2), y = Y(Math.max(v, y0));
        const yt = Math.min(y, Y0), yb = Math.max(y, Y0), hh = yb - yt;
        const pl = planetOf(s);
        c.fillStyle = pl ? hatch(c, pl, s.color) : s.color; c.fillRect(x, yt, bw, hh);
        c.strokeStyle = pl ? s.color : PAL.ink; c.lineWidth = 1; c.setLineDash([]); c.strokeRect(x + 0.5, yt + 0.5, bw - 1, Math.max(0, hh - 1));
        if (o.valueLabels !== false && bw >= 16){ c.font = '10px ' + MONO; const vl = (o.valueFmt || (z => fmt(z, 2)))(v);
          if (c.measureText(vl).width <= bw + 3){ c.fillStyle = PAL.ink; c.textAlign = 'center'; c.textBaseline = v >= 0 ? 'bottom' : 'top'; c.fillText(vl, x + bw / 2, v >= 0 ? yt - 2 : yb + 2); } }
        this.bars.push({x0: x, x1: x + bw, y0: yt, y1: yb, cat, s, v, i}); });
      c.fillStyle = PAL.ink; c.font = '12px ' + FONT; c.textAlign = 'center'; c.textBaseline = 'top';
      String(cat).split('\n').forEach((ln, k) => { if (k) c.font = '11px ' + FONT; c.fillText(ln, cx, top + ph + 7 + 13 * k); });
    });
    if (y0 < 0){ c.strokeStyle = PAL.ink2; c.beginPath(); c.moveTo(L, Math.round(Y0) + 0.5); c.lineTo(L + pw, Math.round(Y0) + 0.5); c.stroke(); }
    (o.refLines || []).forEach((r, k) => { const y = Math.round(Y(r.y)) + 0.5; c.strokeStyle = r.color || PAL.ochre; c.lineWidth = 1.2; c.setLineDash(REFDASH[k % REFDASH.length]); c.beginPath(); c.moveTo(L, y); c.lineTo(L + pw, y); c.stroke(); c.setLineDash([]); });
    c.fillStyle = PAL.ink2; c.font = '12px ' + FONT;
    if (o.xLabel){ c.textAlign = 'center'; c.textBaseline = 'bottom'; c.fillText(o.xLabel, L + pw / 2, h - 3); }
    if (o.yLabel){ c.save(); c.translate(13, top + ph / 2); c.rotate(-Math.PI / 2); c.textAlign = 'center'; c.textBaseline = 'middle'; c.fillText(o.yLabel, 0, 0); c.restore(); }
  }
  onClick(e){
    const r = this.cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
    for (const hbox of this.hits) if (x >= hbox.x0 && x <= hbox.x1 && y >= hbox.y0 && y <= hbox.y1){ this.hidden[hbox.label] = !this.hidden[hbox.label]; this.draw(); return; }
    if (this.o.onPick && this.map && this.o.type !== 'bar'){ const b = this.box; if (x >= b.L && x <= b.L + b.pw) this.o.onPick(this.map.invX(x)); }
  }
  onMove(e){
    const r = this.cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top, o = this.o, b = this.box;
    if (!b) return;
    this.cv.style.cursor = this.hits.some(q => x >= q.x0 && x <= q.x1 && y >= q.y0 && y <= q.y1) ? 'pointer' : 'default';
    if (o.type === 'bar'){
      const hit = (this.bars || []).find(q => x >= q.x0 - 1 && x <= q.x1 + 1 && y >= Math.min(q.y0, q.y1 - 8) && y <= q.y1);
      if (!hit){ hideTip(); return; }
      showTip('<b>' + String(hit.cat).replace('\n', ' ') + '</b><br>' + swatch(hit.s.color) + hit.s.label + ': <b>' + (o.tipFmt || o.valueFmt || (z => fmt(z, 3)))(hit.v, hit) + '</b>' + (o.unit ? ' ' + o.unit : ''), e.clientX, e.clientY);
      return;
    }
    if (!this.map || x < b.L || x > b.L + b.pw || y < b.top || y > b.top + b.ph){ if (this.hover){ this.hover = null; this.draw(); } hideTip(); return; }
    const xv = this.map.invX(x), pts = [];
    for (const s of this.vis()){ if (s.noTip) continue;
      let lo = 0, hi = s.x.length - 1; if (hi < 0) continue;
      while (hi - lo > 1){ const m = (lo + hi) >> 1; if (s.x[m] < xv) lo = m; else hi = m; }
      const i = (Math.abs(s.x[lo] - xv) <= Math.abs(s.x[hi] - xv)) ? lo : hi; if (!isFinite(s.y[i])) continue;
      pts.push({x: s.x[i], y: s.y[i], color: s.color, label: s.label}); }
    if (!pts.length){ hideTip(); return; }
    const xs = o.xCats ? o.xCats[Math.round(pts[0].x)] : fmt(pts[0].x, 4) + (o.xUnit ? ' ' + o.xUnit : '');
    this.hover = {x: pts[0].x, pts}; this.draw();
    showTip('<b>' + xs + '</b>' + pts.map(p => '<br>' + swatch(p.color) + p.label + ': <b>' + (o.tipFmt || (z => fmt(z, 3)))(p.y) + '</b>' + (o.unit ? ' ' + o.unit : '')).join(''), e.clientX, e.clientY);
  }
  savePNG(){
    const [w, h] = this.size(), k = 3, th = this.card.desc ? 46 : 28;
    const cv = el('canvas'); cv.width = w * k; cv.height = (h + th) * k; const c = cv.getContext('2d'); c.scale(k, k);
    c.fillStyle = PAL.surface; c.fillRect(0, 0, w, h + th); c.fillStyle = PAL.ink; c.font = 'bold 14.5px ' + FONT; c.textBaseline = 'top'; c.fillText(this.card.title, 8, 6);
    if (this.card.desc){ c.fillStyle = PAL.ink2; c.font = '12px ' + FONT; let d = this.card.desc; while (c.measureText(d).width > w - 16 && d.length > 10) d = d.slice(0, -4) + '…'; c.fillText(d, 8, 26); }
    c.translate(0, th); const hv = this.hover; this.hover = null; this.render(c, w, h, false); this.hover = hv; this.draw();
    download(this.card.file + '.png', cv.toDataURL('image/png'));
  }
  saveCSV(){
    const o = this.o, S = this.vis(); let rows = [];
    const q = v => '"' + String(v).replace(/"/g, '""').replace(/\n/g, ' ') + '"';
    if (o.type === 'bar'){ rows.push([q(o.xLabel || 'category')].concat(S.map(s => q(s.label))).join(',')); (o.cats || []).forEach((cat, i) => rows.push([q(cat)].concat(S.map(s => s.values[i] ?? '')).join(','))); }
    else {
      const same = S.every(s => s.x.length === S[0].x.length && s.x[0] === S[0].x[0] && s.x[s.x.length - 1] === S[0].x[S[0].x.length - 1]);
      if (same && S.length){ rows.push([q(o.xLabel || 'x')].concat(S.map(s => q(s.label))).join(',')); for (let i = 0; i < S[0].x.length; i++) rows.push([o.xCats ? q(o.xCats[S[0].x[i]]) : S[0].x[i]].concat(S.map(s => s.y[i])).join(',')); }
      else { rows.push('series,' + q(o.xLabel || 'x') + ',' + q(o.yLabel || 'y')); for (const s of S) for (let i = 0; i < s.x.length; i++) rows.push(q(s.label) + ',' + s.x[i] + ',' + s.y[i]); }
    }
    download(this.card.file + '.csv', 'data:text/csv;charset=utf-8,' + encodeURIComponent('# ' + this.card.title + '\n' + rows.join('\n')));
  }
}
window.addEventListener('resize', () => { clearTimeout(window.__rz); window.__rz = setTimeout(() => CHARTS.forEach(ch => ch.draw()), 80); });
