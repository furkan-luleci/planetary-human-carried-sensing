/* ------------------------------------------------------------------ scene */
function lcg(seed){ let s = seed >>> 0; return () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; }; }
/* drawing palette: planet grounds with navy ink outlines (drawing-sheet identity) */
const GROUND = {
  Earth: {top: '#8fae95', face: '#6e5a3e', deep: '#5f8a70', line: '#2e5a4a', deck: '#fbfcfd', deckEdge: '#14284b'},
  Mars:  {top: '#c98a64', face: '#7a3a22', deep: '#a5583a', line: '#6b2416', deck: '#fbfcfd', deckEdge: '#14284b'},
  Moon:  {top: '#a8a9ad', face: '#6d6e73', deep: '#8c8d91', line: '#45464b', deck: '#fbfcfd', deckEdge: '#14284b'}};
const BODY = {skin: '#f7f9fb', shirt: '#f7f9fb', shirtFar: '#cdd5de', legs: '#9fabbb', legsFar: '#7f8b9b', shoe: '#26324a', shoeFar: '#1c2538', line: '#14284b'};
/* habitat interior (the habitat floor): a schematic section through the vessel, in the same engraved-plate manner;
   outside the vessel a pale ground tinted by planet, no landscape */
const HAB = {
  Earth: {out: '#e2e9e8', hatch: 'rgba(31,99,198,.11)'},
  Mars:  {out: '#eedfcf', hatch: 'rgba(196,81,28,.11)'},
  Moon:  {out: '#dedee1', hatch: 'rgba(60,62,70,.13)'},
  inside: '#f7f9fb', lower: '#e6ebf0', shell: '#aab5c2', panel: '#e4eaf0', beam: '#c3ccd6', rigid: '#c8d0d9', column: '#8994a3',
  ink: '#14284b', ink2: '#3d5272'};
/* a small label on a paper box, as the sensor labels */
function boxLabel(c, text, x, y, align){
  c.font = '11px ' + MONO; c.textBaseline = 'middle'; c.textAlign = 'left';
  const tw = c.measureText(text).width, x0 = align === 'right' ? x - tw - 10 : align === 'center' ? x - tw / 2 - 5 : x;
  c.fillStyle = 'rgba(251,252,253,.92)'; c.fillRect(x0, y - 8, tw + 10, 16); c.strokeStyle = 'rgba(20,40,75,.55)'; c.lineWidth = 1; c.strokeRect(x0 + 0.5, y - 7.5, tw + 9, 15);
  c.fillStyle = HAB.ink; c.fillText(text, x0 + 5, y + 0.5);
  return [x0, x0 + tw + 10];
}
/* fine horizontal hatching, as in an engraved sky */
function hatchSky(c, w, top, bottom, col, gap){ c.save(); c.strokeStyle = col; c.lineWidth = 0.7; c.beginPath();
  for (let y = top + 0.5; y < bottom; y += gap){ c.moveTo(0, y); c.lineTo(w, y); } c.stroke(); c.restore(); }
const K = {head: 0, neck: 1, lumbar: 2, pelvis: 3, shL: 4, elL: 5, wrL: 6, haL: 7, shR: 8, elR: 9, wrR: 10, haR: 11,
  hipL: 12, knL: 13, anL: 14, heL: 15, toL: 16, hipR: 17, knR: 18, anR: 19, heR: 20, toR: 21, sW: 22, sP: 23, sT: 24};

class Scene {
  constructor(cv){ this.cv = cv; this.cache = {}; this.tr = null; this.cam = 7; }
  setTrial(tr){ this.tr = tr; this.cam = null; }
  /* far backdrop, cached per planet and size */
  backdrop(planet, w, h, gy){
    const key = planet + '|' + w + '|' + h + '|' + Math.round(gy);
    if (this.cache[key]) return this.cache[key];
    const cv = document.createElement('canvas'); cv.width = w; cv.height = h; const c = cv.getContext('2d'), r = lcg(planet.length * 977 + 13);
    const ridge = (base, amp, col, n, flat) => { c.fillStyle = col; c.beginPath(); c.moveTo(0, h); const ph = [r() * 6, r() * 6, r() * 6], pts = [];
      for (let x = 0; x <= w; x += 4){ let y = Math.sin(x / w * n + ph[0]) + 0.5 * Math.sin(x / w * n * 2.3 + ph[1]) + 0.25 * Math.sin(x / w * n * 5.1 + ph[2]);
        if (flat) y = Math.max(-0.2, Math.min(0.55, y * 1.6)); const yy = base - amp * (0.55 + 0.45 * y); pts.push([x, yy]); c.lineTo(x, yy); } c.lineTo(w, h); c.fill();
      c.strokeStyle = 'rgba(20,40,75,.55)'; c.lineWidth = 0.9; c.beginPath(); pts.forEach(([x, y], i) => i ? c.lineTo(x, y) : c.moveTo(x, y)); c.stroke(); };
    if (planet === 'Earth'){
      c.fillStyle = '#eef3f5'; c.fillRect(0, 0, w, h); hatchSky(c, w, 0, gy * 0.75, 'rgba(31,99,198,.16)', 4);
      c.strokeStyle = '#14284b'; c.lineWidth = 1.1; c.beginPath(); c.arc(w * 0.83, h * 0.17, 15, 0, 7); c.fillStyle = '#f7f9fb'; c.fill(); c.stroke();
      c.beginPath(); for (let k = 0; k < 12; k++){ const a = k * Math.PI / 6; c.moveTo(w * 0.83 + 20 * Math.cos(a), h * 0.17 + 20 * Math.sin(a)); c.lineTo(w * 0.83 + 27 * Math.cos(a), h * 0.17 + 27 * Math.sin(a)); } c.stroke();
      c.strokeStyle = 'rgba(20,40,75,.55)'; c.fillStyle = '#f7f9fb'; c.lineWidth = 0.9;
      for (let i = 0; i < 5; i++){ const x = r() * w, y = h * (0.08 + r() * 0.26), s = 22 + r() * 30; c.beginPath(); for (let k = 0; k < 4; k++) c.ellipse(x + (k - 1.5) * s * 0.55, y + (k % 2) * -4, s * 0.42, s * 0.24, 0, Math.PI, 0); c.fill(); c.stroke(); }
      ridge(gy + 4, 78, '#c9d6c6', 7); ridge(gy + 6, 46, '#a9c0ab', 11); ridge(gy + 8, 20, '#8fae95', 17);
    } else if (planet === 'Moon'){
      c.fillStyle = '#121110'; c.fillRect(0, 0, w, h);
      for (let i = 0; i < 220; i++){ const a = 0.3 + r() * 0.7; c.fillStyle = 'rgba(247,249,251,' + a.toFixed(2) + ')'; const s = r() < 0.08 ? 1.6 : 0.9; c.fillRect(r() * w, r() * gy * 0.95, s, s); }
      const ex = w * 0.2, ey = h * 0.2, er = 24; c.save(); c.beginPath(); c.arc(ex, ey, er, 0, 7); c.clip();
      c.fillStyle = '#1f63c6'; c.fillRect(ex - er, ey - er, 2 * er, 2 * er); hatchSky(c, w, ey - er, ey + er, 'rgba(247,249,251,.35)', 3);
      c.fillStyle = 'rgba(14,20,32,.94)'; c.beginPath(); c.arc(ex + er * 0.62, ey - er * 0.18, er * 1.02, 0, 7); c.fill(); c.restore();
      c.strokeStyle = 'rgba(247,249,251,.85)'; c.lineWidth = 1; c.beginPath(); c.arc(ex, ey, er, 0, 7); c.stroke();
      ridge(gy + 4, 54, '#55575b', 6); ridge(gy + 6, 30, '#6d6f73', 10); ridge(gy + 8, 13, '#85878a', 15);
    } else {
      c.fillStyle = '#efd9bd'; c.fillRect(0, 0, w, h); hatchSky(c, w, 0, gy * 0.8, 'rgba(196,81,28,.14)', 4);
      c.strokeStyle = '#14284b'; c.lineWidth = 1; c.fillStyle = '#f7f9fb'; c.beginPath(); c.arc(w * 0.72, h * 0.22, 9, 0, 7); c.fill(); c.stroke();
      ridge(gy + 4, 70, '#d9a382', 5, true); ridge(gy + 6, 40, '#c98a64', 9, true); ridge(gy + 8, 16, '#b0472a', 14);
    }
    this.cache[key] = cv; return cv;
  }
  /* The habitat floor: a schematic section through the habitat vessel, seen along its axis (the walk goes across
     the vessel, wall to wall). Everything is placed from DATA.structs (span, supports, vessel diameter, floor height,
     shakers); the floor's deflection comes from the trial (deckW), like the deck's. */
  drawHabitat(c, w, h, tr, ST, o){
    const {SX, SY, sc, Kd, follow, lerp, wd, nx} = o, Pp = HAB[tr.planet] || HAB.Earth;
    const L = ST.L, V = ST.vessel, R = V.D / 2, zA = V.z_axis, xc = L / 2, tD = ST.deck_t, tS = 0.14;
    const doorH = 2.1, tP = 0.45;                                   // drawn door height and platform depth (schematic)
    const X = (r, t) => SX(xc + r * Math.cos(t)), Z = (r, t) => SY(zA + r * Math.sin(t));
    // outside the vessel: pale ground tinted by planet, engraved horizontal hatching
    c.fillStyle = Pp.out; c.fillRect(0, 0, w, h); hatchSky(c, w, 0, h, Pp.hatch, 4);
    // inside: the deck space above the floor, the lower deck below it
    const disc = r => { c.beginPath(); c.arc(SX(xc), SY(zA), r * sc, 0, 2 * Math.PI); };
    disc(R); c.fillStyle = HAB.inside; c.fill();
    c.save(); disc(R); c.clip();
    c.fillStyle = HAB.lower; c.fillRect(0, SY(0), w, h);
    c.strokeStyle = 'rgba(20,40,75,.07)'; c.lineWidth = 0.8; c.beginPath();
    for (let yy = SY(0) + 6.5; yy < h; yy += 5){ c.moveTo(0, yy); c.lineTo(w, yy); } c.stroke();
    c.restore();
    // rigid floor before and after the habitat floor: the entry platform (the walk starts on it) and the far landing
    const plat = (x0, x1) => {
      const a = SX(x0), b = SX(x1), top = SY(0), hh = tP * sc;
      if (b < 0 || a > w) return;
      c.fillStyle = HAB.rigid; c.fillRect(a, top, b - a, hh);
      c.save(); c.beginPath(); c.rect(a, top, b - a, hh); c.clip(); c.strokeStyle = 'rgba(20,40,75,.32)'; c.lineWidth = 0.8; c.beginPath();
      for (let xx = a - hh - ((a % 7) + 7) % 7; xx < b; xx += 7){ c.moveTo(xx, top + hh); c.lineTo(xx + hh, top); } c.stroke(); c.restore();
      c.strokeStyle = HAB.ink; c.lineWidth = 1; c.strokeRect(a + 0.5, top + 0.5, b - a - 1, hh);
      c.lineWidth = 2; c.beginPath(); c.moveTo(a, top); c.lineTo(b, top); c.stroke();
    };
    plat(-9, 0); plat(L, L + 9);
    // vessel wall in section (hatched band), open where the platforms pass through it
    const thF = Math.asin(-zA / R), thD = Math.asin(Math.min(0.99, (doorH - zA) / R));
    const arc = (r, t0, t1) => { const n = Math.max(12, Math.ceil(Math.abs(t1 - t0) / 0.015)); c.beginPath();
      for (let k = 0; k <= n; k++){ const t = t0 + (t1 - t0) * k / n; k ? c.lineTo(X(r, t), Z(r, t)) : c.moveTo(X(r, t), Z(r, t)); } };
    const spans = [[thD, Math.PI - thD], [Math.PI - thF, 2 * Math.PI + thF]];      // above the doors; below the floor
    c.save(); c.strokeStyle = 'rgba(20,40,75,.38)'; c.lineWidth = 0.9; c.setLineDash([3, 3]);
    for (const [t0, t1] of [[thF, thD], [Math.PI - thD, Math.PI - thF]]) for (const r of [R, R + tS]){ arc(r, t0, t1); c.stroke(); }
    c.restore();
    for (const [t0, t1] of spans){
      c.lineCap = 'butt'; c.strokeStyle = HAB.shell; c.lineWidth = tS * sc; arc(R + tS / 2, t0, t1); c.stroke();
      c.strokeStyle = HAB.ink; c.lineWidth = 1.1; arc(R, t0, t1); c.stroke(); arc(R + tS, t0, t1); c.stroke();
      c.lineWidth = 2; for (const t of [t0, t1]){ c.beginPath(); c.moveTo(X(R - 0.03, t), Z(R - 0.03, t)); c.lineTo(X(R + tS + 0.03, t), Z(R + tS + 0.03, t)); c.stroke(); }
    }
    // supports: bearings at both walls, the central support down to the bottom of the vessel
    const sup = ST.supports || [0, L];
    for (const xs of sup){
      const xb = Math.min(L - 0.14, Math.max(0.14, xs)), zb = -tD;
      if (xs > 0.01 && xs < L - 0.01){
        const cw = 0.12, zBot = zA - Math.sqrt(Math.max(0, R * R - (xs - xc) * (xs - xc)));
        c.fillStyle = HAB.column; c.fillRect(SX(xs - cw / 2), SY(zb - 0.12), cw * sc, (zb - 0.12 - zBot) * sc);
        c.strokeStyle = HAB.ink; c.lineWidth = 1; c.strokeRect(SX(xs - cw / 2) + 0.5, SY(zb - 0.12) + 0.5, cw * sc - 1, (zb - 0.12 - zBot) * sc);
      }
      c.fillStyle = HAB.ink; c.beginPath(); c.moveTo(SX(xb), SY(zb)); c.lineTo(SX(xb - 0.1), SY(zb - 0.13)); c.lineTo(SX(xb + 0.1), SY(zb - 0.13)); c.closePath(); c.fill();
    }
    // the floor: undeformed line, then the deflected floor (panels on beams, drawn exaggerated)
    if (Kd > 1){ c.strokeStyle = 'rgba(0,0,0,.28)'; c.setLineDash([4, 4]); c.lineWidth = 1; c.beginPath(); c.moveTo(SX(0), SY(0)); c.lineTo(SX(L), SY(0)); c.stroke(); c.setLineDash([]); }
    if (!tr.deckX) return;
    const zf = j => Kd * lerp(tr.A.deckW, nx, j), tPan = 0.037;
    const band = (d0, d1, fill) => { c.fillStyle = fill; c.beginPath();
      for (let j = 0; j < nx; j++){ const x = tr.deckX[j], z = zf(j) - d0; j ? c.lineTo(SX(x), SY(z)) : c.moveTo(SX(x), SY(z)); }
      for (let j = nx - 1; j >= 0; j--) c.lineTo(SX(tr.deckX[j]), SY(zf(j) - d1));
      c.closePath(); c.fill(); };
    band(0, tPan, HAB.panel); band(tPan, tD, HAB.beam);
    c.strokeStyle = HAB.ink; c.lineWidth = 0.8; c.beginPath();
    for (let j = 0; j < nx; j++){ const x = tr.deckX[j], z = zf(j) - tPan; j ? c.lineTo(SX(x), SY(z)) : c.moveTo(SX(x), SY(z)); } c.stroke();
    c.lineWidth = 1.2; c.beginPath();
    for (let j = 0; j < nx; j++){ const x = tr.deckX[j], z = zf(j) - tD; j ? c.lineTo(SX(x), SY(z)) : c.moveTo(SX(x), SY(z)); }
    c.lineTo(SX(L), SY(zf(nx - 1))); c.stroke();
    c.beginPath(); c.moveTo(SX(0), SY(zf(0))); c.lineTo(SX(0), SY(zf(0) - tD)); c.stroke();
    c.lineWidth = 2; c.beginPath();
    for (let j = 0; j < nx; j++){ const x = tr.deckX[j], z = zf(j); j ? c.lineTo(SX(x), SY(z)) : c.moveTo(SX(x), SY(z)); } c.stroke();
    // shakers under the floor, at the positions of the simulation
    if (tr.exc === 'E1') (ST.shakers || []).forEach(([xs], q) => {
      const zt = Kd * wd(xs) - tD, jig = Math.sin(o.t * 2 * Math.PI * (6 + q) + q) * 0.025, bw = 0.32, bh = 0.26;
      c.fillStyle = '#14284b'; c.fillRect(SX(xs - bw / 2), SY(zt + jig), bw * sc, bh * sc); c.fillStyle = o.pcol; c.fillRect(SX(xs - 0.06), SY(zt), 0.12 * sc, (bh + 0.1) * sc * 0.25);
      c.strokeStyle = '#14284b'; c.lineWidth = 1.5; for (const d of [-1, 1]){ c.beginPath(); c.arc(SX(xs + d * 0.26), SY(zt - bh / 2), 0.08 * sc, d > 0 ? -0.9 : Math.PI - 0.9, d > 0 ? 0.9 : Math.PI + 0.9); c.stroke(); }
      if (follow || sc > 44){ c.fillStyle = '#14284b'; c.font = '11px ' + MONO; c.textAlign = 'center'; c.textBaseline = 'top'; c.fillText('shaker', SX(xs), SY(zt - bh) + 4); }
    });
    // labels, each placed where it does not cover the person
    const vis = (x0, x1) => x1 > 0 && x0 < w;
    { const a = Math.max(SX(-9), 4), b = SX(-0.2); if (b - a > 200) boxLabel(c, 'entry platform (rigid floor)', b - 6, SY(-tP / 2), 'right'); else if (b - a > 110) boxLabel(c, 'entry platform', b - 6, SY(-tP / 2), 'right'); }
    { const a = SX(L + 0.2), b = Math.min(SX(L + 9), w - 4); if (b - a > 110) boxLabel(c, 'rigid floor', a + 6, SY(-tP / 2), 'left'); }
    for (const xs of sup) if (xs > 0.01 && xs < L - 0.01 && vis(SX(xs) - 160, SX(xs) + 10)){
      if (follow) boxLabel(c, 'central support', SX(xs - 0.45), SY(-0.32), 'right');
      else boxLabel(c, 'central support', SX(xs) - 8, SY(zA - R * 0.55), 'right');
    }
    if (!follow){
      c.font = '12.5px ' + FONT; c.fillStyle = HAB.ink2; c.textAlign = 'center'; c.textBaseline = 'bottom';
      c.fillText('habitat vessel, Ø ' + fmt(V.D, 2) + ' m, seen along its axis', SX(xc), SY(zA + R + tS) - 6);
      c.textBaseline = 'middle'; c.fillText('lower deck', SX(xc) + 0.5 * R * sc, SY(zA - R * 0.55));
      boxLabel(c, 'habitat floor, ' + meters(L) + ' m', SX(L - 0.3), SY(-0.75), 'right');
    }
  }
  draw(S){
    const cv = this.cv, rect = cv.getBoundingClientRect(), w = Math.round(rect.width), h = Math.round(rect.height), dpr = window.devicePixelRatio || 1;
    if (!w || !h) return;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)){ cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
    const c = cv.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0);
    const tr = this.tr; if (!tr){ c.clearRect(0, 0, w, h); return; }
    const A = tr.A, n = tr.nAnim, nk = tr.kpNames.length, i0 = Math.max(0, Math.min(n - 2, S.i)), a = S.a, i1 = i0 + 1;
    const lerp = (arr, m, k) => arr[i0 * m + k] * (1 - a) + arr[i1 * m + k] * a;
    const standing = tr.activity === 'standing', hasDeck = !!tr.deckX, Kd = S.exag;
    const px = A.refx[i0] * (1 - a) + A.refx[i1] * a;
    const follow = S.view === 'follow';
    const ST = (typeof DATA !== 'undefined' && DATA.structs) ? DATA.structs[tr.structure] : null, hab = !!(ST && ST.vessel);
    let sc, gy, camT;
    if (hab){          // habitat floor: same scale as the deck when following; the whole vessel otherwise
      const R = ST.vessel.D / 2;
      sc = follow ? Math.min(w / 8.5, h / 3.0) : Math.min(w / (ST.L + 6.5), (h - 76) / (2 * R + 0.8));
      gy = follow ? h * 0.76 : h / 2 + 14 + ST.vessel.z_axis * sc;
      camT = follow ? px + (standing ? 0 : 0.4) : ST.L / 2;
    } else {
      sc = follow ? Math.min(w / 8.5, h / 3.0) : w / 22.5; gy = follow ? h * 0.76 : h * 0.70;
      camT = follow ? px + (standing ? 0 : 0.4) : 7.0;
    }
    this.cam = (this.cam === null || !follow) ? camT : this.cam + (camT - this.cam) * 0.25;
    const cx = this.cam, SX = x => w / 2 + (x - cx) * sc, SY = z => gy - z * sc;
    const G = GROUND[tr.planet], pcol = PAL[tr.planet];
    // deflection of the structure along its span (drawn exaggerated); stations deckX from one support to the other
    const nx = hasDeck ? tr.deckX.length : 29, L = hasDeck ? tr.deckX[nx - 1] : 14;
    const wd = x => { if (!hasDeck || x <= 0 || x >= L) return 0; const u = x / L * (nx - 1), j = Math.min(nx - 2, Math.floor(u)), f = u - j; return lerp(A.deckW, nx, j) * (1 - f) + lerp(A.deckW, nx, j + 1) * f; };
    if (hab) this.drawHabitat(c, w, h, tr, ST, {SX, SY, sc, Kd, follow, lerp, wd, nx, t: S.t, pcol});
    else {
    c.drawImage(this.backdrop(tr.planet, w, h, gy), 0, 0, w, h);
    // ---- near ground
    const xl = cx - w / 2 / sc - 1, xr = cx + w / 2 / sc + 1, r = lcg(4242);
    c.fillStyle = G.top; c.beginPath();
    if (hasDeck){ c.moveTo(SX(xl), SY(0)); c.lineTo(SX(-0.05), SY(0)); c.lineTo(SX(-0.05), SY(-0.5)); c.lineTo(SX(0.5), SY(-0.5)); c.lineTo(SX(1.6), SY(-2.3)); c.lineTo(SX(12.4), SY(-2.3)); c.lineTo(SX(13.5), SY(-0.5)); c.lineTo(SX(14.05), SY(-0.5)); c.lineTo(SX(14.05), SY(0)); c.lineTo(SX(xr), SY(0)); }
    else { c.moveTo(SX(xl), SY(0)); c.lineTo(SX(xr), SY(0)); }
    c.lineTo(SX(xr), h); c.lineTo(SX(xl), h); c.closePath(); c.fill();
    c.save(); c.strokeStyle = 'rgba(20,40,75,.13)'; c.lineWidth = 0.8; c.beginPath();
    for (let yy = gy + 6; yy < h; yy += 5){ c.moveTo(0, yy); c.lineTo(w, yy); } c.stroke(); c.restore();
    c.fillStyle = G.line;
    for (let i = 0; i < 70; i++){ const x = -8 + r() * 32, s = 0.03 + r() * 0.07; let z = 0; if (hasDeck && x > 0 && x < 14){ if (x < 1.6 || x > 12.4) continue; z = -2.3; }
      if (x < xl || x > xr) continue; c.beginPath(); c.ellipse(SX(x), SY(z) + 1, s * sc * 1.6, s * sc * 0.8, 0, Math.PI, 0); c.fill(); }
    // foreground detail (moves with the camera): craters, rocks, tufts
    const r2 = lcg(977 + tr.planet.length), deep = h - gy;
    for (let i = 0; i < 46; i++){
      const x = -9 + r2() * 34, dy = 10 + r2() * r2() * (deep - 14), s = (0.10 + r2() * 0.22) * sc * (0.6 + 1.2 * dy / deep), kind = r2();
      if (x < xl - 1 || x > xr + 1) continue; if (hasDeck && x > -0.4 && x < 14.4 && dy < 2.45 * sc + 10) continue;
      const X0 = SX(x), Y0 = gy + dy;
      if (tr.planet === 'Moon'){ c.fillStyle = 'rgba(60,61,64,.55)'; c.beginPath(); c.ellipse(X0, Y0, s * 1.5, s * 0.42, 0, 0, 7); c.fill(); c.strokeStyle = 'rgba(225,226,228,.55)'; c.lineWidth = 1.2; c.beginPath(); c.ellipse(X0, Y0, s * 1.5, s * 0.42, 0, 0.15, Math.PI - 0.15); c.stroke(); }
      else if (tr.planet === 'Mars'){ c.fillStyle = kind < 0.5 ? '#6f351d' : '#834326'; c.beginPath(); c.moveTo(X0 - s * 0.7, Y0); c.lineTo(X0 - s * 0.35, Y0 - s * 0.5); c.lineTo(X0 + s * 0.3, Y0 - s * 0.42); c.lineTo(X0 + s * 0.7, Y0); c.closePath(); c.fill(); c.fillStyle = 'rgba(255,220,190,.22)'; c.beginPath(); c.moveTo(X0 - s * 0.35, Y0 - s * 0.5); c.lineTo(X0 + s * 0.3, Y0 - s * 0.42); c.lineTo(X0, Y0 - s * 0.12); c.closePath(); c.fill(); }
      else { c.fillStyle = kind < 0.6 ? 'rgba(52,110,62,.55)' : 'rgba(120,170,105,.6)'; c.beginPath(); c.ellipse(X0, Y0, s * 1.3, s * 0.38, 0, Math.PI, 0); c.fill(); }
    }
    if (hasDeck){
      // abutment faces and bearings
      c.fillStyle = G.face; c.fillRect(SX(-0.05), SY(0), SX(0.5) - SX(-0.05), SY(-0.5) - SY(0) + 1); c.fillRect(SX(13.5), SY(0), SX(14.05) - SX(13.5), SY(-0.5) - SY(0) + 1);
      c.fillStyle = '#14284b';
      for (const xb of [0.25, 13.75]){ c.beginPath(); c.moveTo(SX(xb), SY(-0.27)); c.lineTo(SX(xb - 0.16), SY(-0.5)); c.lineTo(SX(xb + 0.16), SY(-0.5)); c.fill(); }
      // deck: undeformed outline, then the (exaggerated) deflected slab
      if (Kd > 1){ c.strokeStyle = 'rgba(0,0,0,.28)'; c.setLineDash([4, 4]); c.lineWidth = 1; c.beginPath(); c.moveTo(SX(0), SY(0)); c.lineTo(SX(14), SY(0)); c.stroke(); c.setLineDash([]); }
      const th = 0.27; c.fillStyle = G.deck; c.beginPath();
      for (let j = 0; j <= 28; j++){ const x = tr.deckX[j], z = Kd * lerp(A.deckW, 29, j); j ? c.lineTo(SX(x), SY(z)) : c.moveTo(SX(x), SY(z)); }
      for (let j = 28; j >= 0; j--){ c.lineTo(SX(tr.deckX[j]), SY(Kd * lerp(A.deckW, 29, j) - th)); }
      c.closePath(); c.fill(); c.strokeStyle = G.deckEdge; c.lineWidth = 1.2; c.stroke();
      c.strokeStyle = '#14284b'; c.lineWidth = 2; c.beginPath();
      for (let j = 0; j <= 28; j++){ const x = tr.deckX[j], z = Kd * lerp(A.deckW, 29, j); j ? c.lineTo(SX(x), SY(z)) : c.moveTo(SX(x), SY(z)); } c.stroke();
      // shakers
      if (tr.exc === 'E1') for (const [q, xs] of [3.5, 7.0, 10.5].entries()){
        const zt = Kd * wd(xs) - th, jig = Math.sin(S.t * 2 * Math.PI * (6 + q) + q) * 0.035, bw = 0.42, bh = 0.34;
        c.fillStyle = '#14284b'; c.fillRect(SX(xs - bw / 2), SY(zt + jig), bw * sc, bh * sc); c.fillStyle = pcol; c.fillRect(SX(xs - 0.07), SY(zt), 0.14 * sc, (bh + 0.12) * sc * 0.25);
        c.strokeStyle = '#14284b'; c.lineWidth = 1.5; for (const d of [-1, 1]){ c.beginPath(); c.arc(SX(xs + d * 0.33), SY(zt - bh / 2), 0.09 * sc, d > 0 ? -0.9 : Math.PI - 0.9, d > 0 ? 0.9 : Math.PI + 0.9); c.stroke(); }
        if (follow || sc > 44){ c.fillStyle = tr.planet === 'Moon' ? '#eef2f6' : '#14284b'; c.font = '11px ' + MONO; c.textAlign = 'center'; c.textBaseline = 'top'; c.fillText('shaker', SX(xs), SY(zt - bh) + 4); }
      }
    }
    }
    // ---- body key points (x, z) at this instant
    const P = new Array(nk);
    let fx0 = 0, fz0 = 0;
    if (standing){ for (const k of [K.heL, K.toL, K.heR, K.toR]){ fx0 += lerp(A.dx, nk, k) / 4; fz0 += lerp(A.dz, nk, k) / 4; } }
    const Kb = standing ? Math.min(Kd, tr.kbMax) : 1, ride = standing ? Kd * (wd(px) ) : (Kd - 1) * wd(px);
    for (let k = 0; k < nk; k++){
      if (standing) P[k] = [px + tr.x0[k] + Kb * (lerp(A.dx, nk, k) - fx0), tr.z0[k] + ride + Kb * (lerp(A.dz, nk, k) - fz0)];
      else P[k] = [px + tr.x0[k] + lerp(A.dx, nk, k), tr.z0[k] + lerp(A.dz, nk, k) + ride];
    }
    const limb = (ks, wm, fill) => { const lw = Math.max(2, wm * sc); c.lineCap = 'round'; c.lineJoin = 'round';
      for (const [col, ww] of [[BODY.line, lw + 2.4], [fill, lw]]){ c.strokeStyle = col; c.lineWidth = ww; c.beginPath(); ks.forEach((k, i) => { const p = P[k]; i ? c.lineTo(SX(p[0]), SY(p[1])) : c.moveTo(SX(p[0]), SY(p[1])); }); c.stroke(); } };
    const foot = (an, he, to, fill) => { c.lineJoin = 'round'; c.beginPath(); const hz = 0.075;
      c.moveTo(SX(P[he][0] - 0.03), SY(P[he][1])); c.lineTo(SX(P[to][0] + 0.07), SY(P[to][1])); c.lineTo(SX(P[to][0] + 0.05), SY(P[to][1] + 0.045)); c.lineTo(SX(P[an][0] + 0.03), SY(P[an][1] + 0.02)); c.lineTo(SX(P[an][0] - 0.05), SY(P[an][1] + 0.02)); c.lineTo(SX(P[he][0] - 0.04), SY(P[he][1] + hz)); c.closePath();
      c.fillStyle = fill; c.fill(); c.strokeStyle = BODY.line; c.lineWidth = 1.2; c.stroke(); };
    // far side (right), trunk, near side (left)
    limb([K.shR, K.elR, K.wrR, K.haR], 0.075, BODY.shirtFar); limb([K.hipR, K.knR, K.anR], 0.115, BODY.legsFar); foot(K.anR, K.heR, K.toR, BODY.shoeFar);
    limb([K.pelvis, K.lumbar, K.neck], 0.27, BODY.shirt);
    limb([K.hipL, K.knL, K.anL], 0.125, BODY.legs); foot(K.anL, K.heL, K.toL, BODY.shoe);
    const hx = (P[K.neck][0] + P[K.head][0]) / 2, hz = (P[K.neck][1] + P[K.head][1]) / 2, hr = Math.hypot(P[K.head][0] - P[K.neck][0], P[K.head][1] - P[K.neck][1]) * 0.47;
    c.fillStyle = BODY.skin; c.strokeStyle = BODY.line; c.lineWidth = 1.3; c.beginPath(); c.arc(SX(hx), SY(hz), hr * sc, 0, 7); c.fill(); c.stroke();
    limb([K.shL, K.elL, K.wrL], 0.08, BODY.shirt); limb([K.wrL, K.haL], 0.06, BODY.skin);
    // sensors
    for (const [name, k, lab] of [['wrist', K.sW, 'watch (wrist)'], ['thigh', K.sT, 'phone (thigh pocket)']]){
      const on = S.sensor === name, s = (on ? 6 : 4), x = SX(P[k][0]), y = SY(P[k][1]);
      if (on){ c.strokeStyle = PAL[name]; c.lineWidth = 2; c.globalAlpha = 0.45 + 0.35 * Math.sin(S.wall * 5); c.beginPath(); c.arc(x, y, s + 6, 0, 7); c.stroke(); c.globalAlpha = 1; }
      c.fillStyle = PAL[name]; c.strokeStyle = PAL.ink; c.lineWidth = 1.5; c.beginPath(); c.rect(x - s, y - s, 2 * s, 2 * s); c.fill(); c.stroke();
      if (on){ c.font = '11px ' + MONO; c.textAlign = 'left'; c.textBaseline = 'middle'; const tw = c.measureText(lab).width; c.fillStyle = '#fbfcfd'; c.fillRect(x + 12, y - 9, tw + 10, 18); c.strokeStyle = PAL.ink; c.lineWidth = 1.2; c.strokeRect(x + 12.5, y - 8.5, tw + 9, 17); c.fillStyle = PAL.ink; c.fillText(lab, x + 17, y + 0.5); }
    }
    // foot forces
    for (const [j, he, to] of [[0, K.heL, K.toL], [1, K.heR, K.toR]]){
      const f = lerp(A.fz, 2, j); if (f < 0.03) continue; const x = SX((P[he][0] + P[to][0]) / 2), y0 = SY(Math.min(P[he][1], P[to][1])) + 3, len = f * 0.85 * sc;
      c.strokeStyle = pcol; c.fillStyle = pcol; c.lineWidth = 3; c.beginPath(); c.moveTo(x, y0 + len); c.lineTo(x, y0 + 7); c.stroke(); c.beginPath(); c.moveTo(x, y0); c.lineTo(x - 6, y0 + 9); c.lineTo(x + 6, y0 + 9); c.fill();
    }
    // ---- labels
    const dark = tr.planet === 'Moon' && !hab; c.textBaseline = 'top'; c.textAlign = 'left';
    const halo = (txt, x, y) => { if (!hab) return; c.save(); c.lineWidth = 4; c.lineJoin = 'round'; c.strokeStyle = 'rgba(251,252,253,.92)'; c.strokeText(txt, x, y); c.restore(); };
    if (hab){          // a paper label behind the title, so that the vessel wall never runs through it
      c.font = 'bold 16px ' + FONT; const w1 = c.measureText(tr.planet + '  ·  g = ' + tr.g.toFixed(2) + ' m/s²').width;
      c.font = '12.5px ' + FONT; const w2 = c.measureText(S.caption || '').width;
      c.fillStyle = 'rgba(251,252,253,.86)'; c.fillRect(8, 6, Math.max(w1, w2) + 14, 46); }
    c.font = 'bold 16px ' + FONT; c.fillStyle = dark ? '#eef2f6' : '#14284b'; halo(tr.planet + '  ·  g = ' + tr.g.toFixed(2) + ' m/s²', 14, 12); c.fillText(tr.planet + '  ·  g = ' + tr.g.toFixed(2) + ' m/s²', 14, 12);
    c.font = '12.5px ' + FONT; c.fillStyle = dark ? '#d8d0bd' : '#3d5272'; halo(S.caption || '', 14, 34); c.fillText(S.caption || '', 14, 34);
    const fg = hab ? HAB.ink : '#fff';
    c.textBaseline = 'bottom'; c.fillStyle = fg; c.font = '11px ' + MONO;
    const msg = (hasDeck ? (Kd > 1 ? 'Deflection of the structure drawn ×' + Kd + (standing ? ', body vibration ×' + Math.round(Kb) : '') : (S.trueSize || 'True size (the structure moves by fractions of a millimetre)')) : 'Rigid ground, no structure') + '   ·   arrows: force under each foot';
    if (hab){ c.save(); c.lineWidth = 3; c.strokeStyle = 'rgba(251,252,253,.9)'; c.lineJoin = 'round'; c.strokeText(msg, 14, h - 10); c.restore(); }
    c.fillText(msg, 14, h - 10);
    const bx = w - 24 - sc; c.strokeStyle = fg; c.lineWidth = 2; c.beginPath(); c.moveTo(bx, h - 16); c.lineTo(bx + sc, h - 16); c.moveTo(bx, h - 21); c.lineTo(bx, h - 11); c.moveTo(bx + sc, h - 21); c.lineTo(bx + sc, h - 11); c.stroke();
    c.textAlign = 'center'; c.fillText('1 m', bx + sc / 2, h - 20);
  }
}
