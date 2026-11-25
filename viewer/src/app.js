/* -------------------------------------------------------------------- app */
const DATA = JSON.parse(document.getElementById('bundle').textContent);
const G_E = 9.81;
const SENSOR_NAME = {wrist: 'Watch', thigh: 'Phone'};
const SENSOR_FULL = {wrist: 'Watch (wrist)', thigh: 'Phone (thigh pocket)'};
const COND = {same: {Earth: 'earth_1.00', Mars: 'mars_1.00', Moon: 'moon_1.00'},
              similar: {Earth: 'earth_1.25', Mars: 'mars_0.77_similar', Moon: 'moon_0.51_similar'}};
const COND_LABEL = {'earth_1.25': 'Earth 1.25 m/s', 'earth_1.00': 'Earth 1.0 m/s', 'mars_1.00': 'Mars 1.0 m/s', 'moon_1.00': 'Moon 1.0 m/s',
  'mars_0.77_similar': 'Mars, what-if (Earth 1.25 m/s gait scaled)', 'moon_0.51_similar': 'Moon, what-if (Earth 1.25 m/s gait scaled)',
  'earth3d_1.00': 'Earth 1.0 m/s, 3-D walker', 'mars3d_1.00': 'Mars 1.0 m/s, 3-D walker', 'moon3d_1.00': 'Moon 1.0 m/s, 3-D walker'};
const NOISE_PSD = Math.pow(100e-6 * 9.80665, 2);     // (m/s^2)^2/Hz, 100 micro-g per root hertz
/* The two structures. "S2" and "S1" are the keys of the result files; the reader only ever sees these names.
   The habitat floor is the study's main case and comes first everywhere. */
const STRUCTS = ['S2', 'S1'];
const STRUCT = {
  S2: {name: 'Habitat floor (NASA Common Habitat)', short: 'habitat floor', html: 'Habitat floor <small>(NASA Common Habitat)</small>'},
  S1: {name: 'Reference bridge (the earlier study’s bridge model)', short: 'reference bridge', html: 'Reference bridge <small>(the earlier study’s bridge model)</small>'}};
function SG(k){ return (DATA.structs || {})[k] || null; }                 // geometry and settings of a structure (from the model code)
function isFloor(k){ const g = SG(k); return !!(g && g.vessel); }
function bodyMass(tr){ return tr.weight / tr.g; }
function onPlanet(P){ return P === 'Moon' ? 'the Moon' : P; }
/* the simulated body's mass, and the standing trials' length and lead-in, read from the bundled trials and results */
function bodyKg(){ const t = DATA.trials[0]; return Math.round(t.weight / t.g); }
function standSeconds(k){ const O = k === 'S2' ? DATA.outcomes_S2 : DATA.outcomes, d = O && O.data.stand_E1 ? O.data.stand_E1.Earth : null; return d && d.n_records ? d.seconds / d.n_records : null; }
function standLeadIn(k){ const t = DATA.trials.find(q => q.structure === k && q.activity === 'standing' && q.leadS !== undefined); return t ? t.leadS : null; }
function standTrialsText(k){ const s = standSeconds(k), l = standLeadIn(k); return (s ? fmt(s, 2) + ' s each' : '') + (l ? (s ? ', ' : '') + 'after a ' + fmt(l, 2) + ' s lead-in' : ''); }
function tolText(k){ const O = k === 'S2' ? DATA.outcomes_S2 : DATA.outcomes; return O && O.recovery_rule && O.recovery_rule.tolerance ? O.recovery_rule.tolerance : ''; }
/* a length for the reader: 14 m, 8.20 m */
function meters(v){ return Math.abs(v - Math.round(v)) < 0.005 ? String(Math.round(v)) : v.toFixed(2); }
/* a range of shares, written once when both ends agree: 2.0 %, 5.7–6.5 % */
function pctRange(a, b, d){ const pa = pct(a, d === undefined ? 1 : d), pb = pct(b, d === undefined ? 1 : d); return (pa === pb ? pa : pa.replace(' %', '') + '–' + pb).replace(' %', '\u00a0%'); }
const state = {structure: 'S2', planet: 'Earth', activity: 'walking', exc: 'E1', speed: 'same', sensor: 'thigh', t: 0, playing: true, rate: 1, exag: 100, view: 'three', draw: '3d', walker: '3d', autoWalker: true};

function b64bytes(s){ const bin = atob(s), n = bin.length, u = new Uint8Array(n); for (let i = 0; i < n; i++) u[i] = bin.charCodeAt(i); return u; }
function decI16(o){ const u = b64bytes(o.b), q = new Int16Array(u.buffer, 0, u.length >> 1), out = new Float32Array(q.length); for (let i = 0; i < q.length; i++) out[i] = q[i] * o.s; return out; }
function decF32(o){ const u = b64bytes(o.f); return new Float32Array(u.buffer, 0, u.length >> 2); }
function prep(tr){
  if (tr.A) return tr;
  const A = tr.A = {refx: decF32(tr.refx), dx: decI16(tr.dx), dz: decI16(tr.dz), dy: tr.dy ? decI16(tr.dy) : new Float32Array(tr.nAnim * tr.kpNames.length), fz: decI16(tr.fz), fx: decI16(tr.fx)};
  if (!tr.y0) tr.y0 = tr.kpNames.map(() => 0);
  if (tr.deckW) A.deckW = decI16(tr.deckW);
  const n = tr.nAnim, nk = tr.kpNames.length, nx = tr.deckX ? tr.deckX.length : 0;
  tr.dur = (n - 1) / tr.fsAnim; tr.kbMax = 1; tr.zdeck0 = 0;
  if (tr.activity === 'standing'){
    let m = 1e-9; const feet = [15, 16, 20, 21];
    for (let i = 0; i < n; i++){ let fx = 0, fz = 0; for (const k of feet){ fx += A.dx[i * nk + k] / 4; fz += A.dz[i * nk + k] / 4; }
      for (let k = 0; k < nk; k++){ m = Math.max(m, Math.abs(A.dx[i * nk + k] - fx), Math.abs(A.dz[i * nk + k] - fz)); } }
    tr.kbMax = 0.07 / m;
    // show the vibration only: remove the constant sag of the structure under the standing person
    if (A.deckW){ for (let j = 0; j < nx; j++){ let s = 0; for (let i = 0; i < n; i++) s += A.deckW[i * nx + j]; s /= n; for (let i = 0; i < n; i++) A.deckW[i * nx + j] -= s; } }
  }
  tr.wMax = 0; if (A.deckW) for (let i = 0; i < A.deckW.length; i++) tr.wMax = Math.max(tr.wMax, Math.abs(A.deckW[i]));
  tr.S = {}; for (const k in tr.sig.ch) tr.S[k] = decI16(tr.sig.ch[k]);
  const ns = tr.S.grfL.length; tr.sigT = new Float32Array(ns); for (let i = 0; i < ns; i++) tr.sigT[i] = (tr.sig.t0 - tr.t0) + i / tr.sig.fs;
  tr.P = {}; for (const k in tr.spec.ch) tr.P[k] = decF32(tr.spec.ch[k]);
  tr.PR = {}; if (tr.spec.rigid) for (const k in tr.spec.rigid) tr.PR[k] = decF32(tr.spec.rigid[k]);
  return tr;
}
/* the trial for a set of choices (structure, planet, activity, excitation, speed, walker), or undefined */
function findTrialFor(s){
  const T = DATA.trials.filter(t => t.structure === s.structure);
  if (s.activity === 'standing') return T.find(t => t.activity === 'standing' && t.planet === s.planet);
  if (s.walker === '3d' && s.speed === 'same'){
    const t3 = T.find(t => t.activity === 'walking' && t.cond === s.planet.toLowerCase() + '3d_1.00' && t.exc === s.exc);
    if (t3) return t3;
  }
  const cond = COND[s.speed][s.planet];
  return T.find(t => t.activity === 'walking' && t.cond === cond && t.exc === s.exc);
}
function findTrial(){ return findTrialFor(state); }
function exists(over){ return !!findTrialFor(Object.assign({}, state, over || {})); }
function standingRow(planet, variant){ return DATA.standing ? DATA.standing.rows.find(r => r.gravity === planet && r.variant === (variant || 'nominal')) : null; }

/* ------------------------------------------------------------------- tabs */
const built = {};
function showTab(name){
  document.querySelectorAll('.tab').forEach(b => { b.classList.toggle('on', b.dataset.tab === name); b.setAttribute('aria-selected', b.dataset.tab === name ? 'true' : 'false'); });
  const tb = document.querySelector('.tab[data-tab="' + name + '"]');
  if (tb){ document.getElementById('tbNo').textContent = tb.dataset.no; document.getElementById('tbTitle').textContent = tb.querySelector('span').textContent; }
  ['watch', 'gravity', 'detect', 'model'].forEach(n => { document.getElementById('tab-' + n).hidden = n !== name; });
  if (!built[name]){ built[name] = true; ({gravity: buildGravity, detect: buildDetect, model: buildModel}[name] || (() => {}))(); }
  CHARTS.forEach(ch => ch.draw()); window.scrollTo(0, 0);
}
document.querySelectorAll('.tab').forEach(b => b.addEventListener('click', () => showTab(b.dataset.tab)));

/* ------------------------------------------------------------ 1  WATCH */
const scene = new Scene(document.getElementById('stage'));
let scene3d = null;
function use3d(){ return state.draw === '3d' && scene3d && scene3d.ok; }
function initDrawing(){
  if (webglAvailable()){ try { scene3d = new Scene3D(document.getElementById('stage3d'), {label: document.getElementById('ovLabel')}); } catch (e){ scene3d = null; } }
  if (!scene3d || !scene3d.ok){ state.draw = '2d'; const d = document.getElementById('draw'); d.value = '2d'; d.disabled = true; d.title = 'This browser has no WebGL; showing the flat drawing.'; }
  else scene3d.setView(state.view);
  syncDrawing();
}
function syncDrawing(){
  const three = use3d();
  document.getElementById('stage').hidden = three;
  document.getElementById('stage3d').hidden = !three;
  ['ovTitle', 'ovNote', 'ovHint'].forEach(id => { document.getElementById(id).hidden = !three; });
  if (!three) document.getElementById('ovLabel').hidden = true;
}
let cur = null, ctl = {}, tcharts = {}, specChart = null;
function structTitle(k){
  const g = SG(k); if (!g) return '';
  return isFloor(k) ? 'Main case: a floor of NASA’s Common Habitat mid deck, built like the floor of NASA’s Habitat Demonstration Unit: ' + meters(g.L) + ' m wall to wall with a central support.'
    : 'The ' + meters(g.L) + ' m bridge model of the earlier study, used here to relate the results to the earlier studies.';
}
function buildControls(){
  const box = document.getElementById('controls');
  const dot = p => glyphSVG(p, 15);
  ctl.structure = segControl(box, 'Structure', STRUCTS.filter(k => DATA.trials.some(t => t.structure === k)).map(k => ({v: k, html: STRUCT[k].html, title: structTitle(k)})),
    state.structure, v => { state.structure = v; pick(); });
  ctl.structure.seg.classList.add('struct');
  ctl.planet = segControl(box, 'Where', PLANETS.map(p => ({v: p, html: dot(p) + p + ' <small>' + (p === 'Earth' ? 'full weight' : Math.round(100 * DATA.planets[p].g / G_E) + ' % of Earth weight') + '</small>'})), state.planet, v => { state.planet = v; pick(); });
  ctl.activity = segControl(box, 'The person is', [{v: 'standing', label: 'Standing still'}, {v: 'walking', label: 'Walking across'}], state.activity, v => { state.activity = v; if (v === 'standing') state.exc = 'E1'; ctl.exc.set(state.exc); pick(); });
  ctl.exc = segControl(box, 'What shakes the structure', [
    {v: 'E1', label: 'Shakers (test)', title: 'Three shakers under the structure apply the same random forces on every planet.'},
    {v: 'E2', label: 'Only the footsteps', title: 'Nothing but the walker moves the structure.'},
    {v: 'rigid', label: 'Solid ground', title: 'The same walk on ground that cannot move. Used as the comparison case. (On the habitat floor this comparison is the grey dashed curve of the frequency chart.)'}], state.exc, v => { state.exc = v; pick(); });
  ctl.speed = segControl(box, 'Walking speed', [
    {v: 'same', label: '1.0 m/s on every planet', title: 'The gait was re-tuned for each planet so that speed, step rate and time on the ground follow published reduced-gravity experiments.'},
    {v: 'similar', label: 'Earth gait, scaled (what-if)', title: 'The Earth gait tuned for 1.25 m/s, replayed in slow motion so that speed and gravity keep the same ratio (equal Froude number); the scaled walkers walk at 0.72 m/s on Mars and 0.45 m/s on the Moon. On Earth this shows the 1.25 m/s gait itself. A what-if case, not what people do. The study analyses it on the reference bridge; on the habitat floor the viewer shows one demonstration crossing per planet.'}], state.speed, v => { state.speed = v; pick(); });
  ctl.walker = segControl(box, 'Walker', [
    {v: '3d', label: '3-D', title: 'Full 3-D body with sideways balance and arm swing, tuned on each planet.'},
    {v: '2d', label: 'Side view', title: 'Walker restricted to forward and up-down motion; used for the Earth–Mars–Moon comparison.'}], state.walker, v => { state.walker = v; state.autoWalker = false; pick(); });
  ctl.sensor = segControl(box, 'Device shown', ['wrist', 'thigh'].map(k => ({v: k, html: '<span class="sw" style="background:var(--' + k + ')"></span>' + SENSOR_FULL[k]})), state.sensor, v => { state.sensor = v; refreshTrialViews(); });
}
function buildWatch(){
  buildControls();
  const tc = document.getElementById('timeCards');
  const mk = (key, title, desc, yLabel, h) => { const card = makeCard(tc, {title, desc, height: h || 150, file: 'trial_' + key}); tcharts[key] = new Chart(card, {xLabel: 'time in this trial (s)', yLabel, legend: true, xUnit: 's', onPick: x => { state.t = Math.max(0, Math.min(cur ? cur.dur : 0, x)); }}); return card; };
  mk('sensor', 'Device signal in time', 'Change in the size of the acceleration measured by the worn device.', 'm/s²');
  mk('deck', 'Structure motion under the person', 'Vertical acceleration of the structure at the person’s position.', 'm/s²');
  mk('grf', 'Force under each foot', 'Vertical force under each foot (and, for the 3-D walker, the sideways force), in units of the person’s weight on this planet.', '× body weight');
  const sc = makeCard(document.getElementById('specCard'), {title: 'Frequency content of the device signal', desc: 'Peaks at the dashed lines come from the structure. Peaks elsewhere come from the person’s own movement.', height: 300, file: 'trial_spectrum',
    note: ' '});
  specChart = new Chart(sc, {xLabel: 'frequency (Hz)', yLabel: 'strength, (m/s²)² per Hz', yLog: true, xMin: 0, xMax: 30, xUnit: 'Hz'});
  const cmp = el('input', {type: 'checkbox', id: 'cmpPlanets', onchange: e => { state.compare = e.target.checked; refreshTrialViews(); }});
  sc.opts.appendChild(el('label', {title: 'Overlay the same situation on the other two planets'}, [cmp, ' Compare with the other planets']));
  document.getElementById('play').addEventListener('click', () => { state.playing = !state.playing; syncPlay(); });
  document.getElementById('seek').addEventListener('input', e => { if (cur) state.t = e.target.value / 1000 * cur.dur; });
  document.getElementById('rate').addEventListener('change', e => { state.rate = +e.target.value; });
  document.getElementById('exag').addEventListener('change', e => { state.exag = +e.target.value; });
  document.getElementById('view').addEventListener('change', e => { state.view = e.target.value; scene.cam = null; if (scene3d) scene3d.setView(state.view); });
  document.getElementById('draw').addEventListener('change', e => { state.draw = e.target.value; syncDrawing(); });
  document.getElementById('view').value = state.view;
  initDrawing();
  document.addEventListener('keydown', e => { if (e.code === 'Space' && e.target.tagName !== 'SELECT' && e.target.tagName !== 'BUTTON' && !document.getElementById('tab-watch').hidden){ e.preventDefault(); state.playing = !state.playing; syncPlay(); } });
  pick();
  requestAnimationFrame(frame);
}
function syncPlay(){ document.getElementById('play').textContent = state.playing ? '❚❚ Pause' : '▶ Play'; }
/* default drawing exaggeration: the reference bridge keeps its defaults; the light floor moves more while someone walks on it */
function defaultExag(tr){ return tr.activity === 'standing' ? (isFloor(tr.structure) ? 1000 : 3000) : (isFloor(tr.structure) ? 100 : 300); }
function pick(){
  const s = state;
  if (s.activity === 'standing') s.exc = 'E1';
  // always land on a trial that exists: drop the choices this structure does not have (what-if gaits, other speeds, solid ground)
  if (s.activity === 'walking'){
    if (!exists() && exists({speed: 'same'})) s.speed = 'same';
    if (!exists() && exists({exc: 'E1'})) s.exc = 'E1';
    if (!exists()){ s.speed = 'same'; s.exc = 'E1'; }
  }
  if (!exists()){ s.activity = 'standing'; s.exc = 'E1'; }
  const standing = s.activity === 'standing';
  ctl.activity.set(s.activity); ctl.exc.set(s.exc); ctl.speed.set(s.speed);
  ctl.activity.enable('walking', exists({activity: 'walking', exc: 'E1', speed: 'same'}));
  ['E1', 'E2', 'rigid'].forEach(v => ctl.exc.enable(v, standing ? v === 'E1' : exists({exc: v})));
  ['same', 'similar'].forEach(v => ctl.speed.enable(v, !standing && exists({speed: v})));
  const has3d = DATA.trials.some(t => t.structure === s.structure && t.cond === s.planet.toLowerCase() + '3d_1.00' && t.exc === s.exc);
  const can3d = has3d && !standing && s.speed === 'same';
  ctl.walker.enable('3d', can3d); ctl.walker.enable('2d', !standing);
  if (!can3d && s.walker === '3d'){ s.walker = '2d'; ctl.walker.set('2d'); }
  else if (can3d && s.wantWalker3d !== false && s.walker !== '3d' && s.autoWalker){ s.walker = '3d'; ctl.walker.set('3d'); }
  const tr = findTrial(), msg = document.getElementById('stageMsg');
  if (!tr){ cur = null; scene.setTrial(null); msg.hidden = false; msg.textContent = 'This case has not been simulated.'; document.getElementById('tiles').innerHTML = ''; return; }
  msg.hidden = true; cur = prep(tr); scene.setTrial(cur); if (scene3d && scene3d.ok) scene3d.setTrial(cur); state.t = 0;
  const v = document.getElementById('view'); if (standing && state.view === 'full'){ state.view = 'three'; v.value = 'three'; if (scene3d) scene3d.setView('three'); }
  const key = tr.structure + '|' + s.activity;
  if (state.lastKey !== key){ state.exag = defaultExag(tr); document.getElementById('exag').value = String(state.exag); state.lastKey = key; }
  refreshTrialViews();
}
/* one row of the trial's data schedule: quantity, value, unit, remark */
function tile(k, v, unit, n){ return '<tr><td>' + k + '</td><td class="v">' + v + '</td><td class="u">' + (unit || '') + '</td><td class="r">' + (n || '') + '</td></tr>'; }
function scheduleTables(h){
  const rows = h.match(/<tr>.*?<\/tr>/g) || [], half = Math.ceil(rows.length / 2);
  const tbl = rr => '<table class="t"><thead><tr><th class="l">Quantity</th><th>Value</th><th class="l">Unit</th><th class="l">Remark</th></tr></thead><tbody>' + rr.join('') + '</tbody></table>';
  return tbl(rows.slice(0, half)) + (rows.length > half ? tbl(rows.slice(half)) : '');
}
/* the structure's natural frequencies as chart guides; on the habitat floor also where they sit with a person standing on it */
function modeGuides(struct, planet, fmax){
  const g = SG(struct), out = ((g && g.modes) || []).filter(m => m.f < fmax).map(m => ({x: m.f, label: m.label}));
  const occ = isFloor(struct) && DATA.outcomes_S2 && DATA.outcomes_S2.occupied ? DATA.outcomes_S2.occupied[planet] : null;
  if (occ) for (const lab in occ) if (occ[lab] < fmax) out.push({x: occ[lab], color: PAL.ochre, dash: [2, 2]});
  return out;
}
function specNote(tr){
  const tail = ' The sensor curve combines its three axes. The horizontal line is the noise level of a typical phone or watch accelerometer: anything below it would be hidden in a real device.';
  if (isFloor(tr.structure)) return 'Dashed vertical lines: the habitat floor’s natural frequencies below 30 Hz (B = vertical bending), empty floor. Amber dotted lines: the same modes with a person standing on the floor, predicted before the trials (a light floor shifts when someone stands on it).'
    + (tr.PR && Object.keys(tr.PR).length ? ' Grey dashed: the same walk on solid ground, cut to the floor’s crossing.' : '') + tail;
  return 'Dashed vertical lines: natural frequencies of the structure (B = vertical bending, T = twisting, L = sideways bending).' + tail;
}
function refreshTrialViews(){
  const tr = cur; if (!tr) return;
  const st = tr.stats, sen = state.sensor, pc = PAL[tr.planet], walking = tr.activity === 'walking', g = SG(tr.structure), floor = isFloor(tr.structure);
  // tiles
  let h = '';
  if (g) h += tile('Structure', meters(g.L), 'm', floor ? STRUCT.S2.short + ', wall to wall' : STRUCT.S1.short + ' (the earlier study’s bridge model)');
  h += tile('Gravity', tr.g.toFixed(2), 'm/s²', Math.round(100 * tr.g / G_E) + ' % of Earth') + tile('Body weight', Math.round(tr.weight), 'N', 'same ' + Math.round(bodyMass(tr)) + ' kg body everywhere');
  if (walking){
    h += tile('Walking speed', fmt(st.speed, 3), 'm/s') + tile('Step rate', fmt(st.step_frequency, 3), 'steps/s', 'step length ' + fmt(st.step_length, 2) + ' m')
      + tile('Each foot on the ground', pct(st.duty_factor), '', st.flight_fraction > 0.02 ? 'both feet in the air ' + pct(st.flight_fraction) + ' of the time' : 'always at least one foot down')
      + tile('Peak force under a foot', fmt(st.grf_foot_peak_over_W, 3), '× weight')
      + (st.trunk_lean !== undefined ? tile('Upper body leans forward', fmt(st.trunk_lean * 180 / Math.PI, 2), '°', 'average while walking') : '');
    if (st.step_width !== undefined) h += tile('Step width', fmt(100 * st.step_width, 2), 'cm', 'sideways distance between the feet')
      + tile('Sideways force, peak', fmt(st.grf_lateral_peak_over_W, 2), '× weight', 'body sway ' + fmt(100 * st.com_lateral_excursion, 2) + ' cm side to side');
  } else {
    const row = standingRow(tr.planet);
    if (row) h += tile('Body resonance, standing', fmt(row.apparent_mass_f, 3), 'Hz', 'Earth: ' + fmt(standingRow('Earth').apparent_mass_f, 3) + ' Hz') + tile('Legs and soles compressed by', fmt(row.leg_compression_mm + row.pad_compression_mm, 2), 'mm', 'less weight, less compression');
  }
  if (st.deck_rms_z !== undefined) h += tile('Vibration of the structure under the person', fmt(1000 * st.deck_rms_z, 3), 'mm/s²', 'vertical, typical size');
  h += tile(SENSOR_NAME[sen] + ' vibration', fmt(1000 * st[sen + '_rms'], 3), 'mm/s²', 'all movement, three axes');
  document.getElementById('tiles').innerHTML = scheduleTables(h);
  // time series
  const T = Array.from(tr.sigT), what = floor ? 'floor' : 'bridge';
  tcharts.sensor.card.card.querySelector('.ttl').textContent = SENSOR_NAME[sen] + ' signal in time';
  tcharts.deck.card.card.querySelector('.desc').textContent = 'Vertical acceleration of the ' + (tr.deckX ? (floor ? 'habitat floor' : 'reference bridge') : 'structure') + ' at the person’s position.';
  tcharts.sensor.o.series = [{x: T, y: tr.S[sen], color: PAL[sen], label: SENSOR_NAME[sen] + '', width: 1.6}];
  tcharts.deck.card.card.style.display = tr.S.deck ? '' : 'none';
  tcharts.deck.o.series = tr.S.deck ? [{x: T, y: tr.S.deck, color: PAL.deck, label: (floor ? 'Floor' : 'Bridge') + ' under the person', width: 1.6}] : [];
  tcharts.grf.o.series = [{x: T, y: tr.S.grfL, color: pc, label: 'Left foot', width: 1.8}, {x: T, y: tr.S.grfR, color: pc, label: 'Right foot', dash: [5, 3], width: 1.8}];
  if (tr.S.grfY) tcharts.grf.o.series.push({x: T, y: tr.S.grfY, color: PAL.ink, label: 'Sideways (both feet)', width: 1.1});
  tcharts.grf.o.yMin = 0;
  if (tr.S.grfY){ let mn = 0; for (let i = 0; i < tr.S.grfY.length; i++) mn = Math.min(mn, tr.S.grfY[i]); tcharts.grf.o.yMin = Math.min(-0.2, 1.15 * mn); }
  for (const k of ['sensor', 'deck', 'grf']){ const s = tcharts[k].o.series; if (!s.length) continue; let m = 0; for (const q of s) for (let i = 0; i < q.y.length; i++) m = Math.max(m, Math.abs(q.y[i])); if (k === 'grf'){ tcharts[k].o.yMax = Math.max(1.2, m * 1.1); } else { tcharts[k].o.yMin = -m * 1.1; tcharts[k].o.yMax = m * 1.1; } }
  // spectrum
  const f = tr.spec.f, ser = [{x: f, y: tr.P[sen], color: PAL[sen], label: SENSOR_NAME[sen] + '' + (tr.deckX ? ' on the ' + what : ' on solid ground'), width: 2.2}];
  if (tr.PR[sen]) ser.push({x: f, y: tr.PR[sen], color: PAL.rigid, label: 'Same walk on solid ground', dash: [5, 3], width: 1.6});
  if (tr.P.deck) ser.push({x: f, y: tr.P.deck, color: PAL.ink, label: (floor ? 'Floor' : 'Bridge') + ' under the person', width: 1.1});
  if (state.compare){
    for (const p of PLANETS){
      if (p === tr.planet) continue;
      const o = findTrialFor(Object.assign({}, state, {planet: p}));
      if (!o) continue; prep(o);
      const fo = o.spec.f, yo = Array.from(fo, (_, i) => o.P[sen][i]);
      ser.push({x: fo, y: yo, color: PAL[p], label: SENSOR_NAME[sen] + ', ' + p, width: 1.4, alpha: 0.85});
    }
  }
  const guides = modeGuides(tr.structure, tr.planet, 30);
  let lo = Infinity, hi = 0; for (const s of ser) for (let i = 0; i < f.length; i++){ if (f[i] < 0.5 || f[i] > 30) continue; const v = s.y[i]; if (v > 0){ lo = Math.min(lo, v); hi = Math.max(hi, v); } }
  lo = Math.max(lo, hi * 1e-9, 1e-10);
  specChart.card.note.innerHTML = specNote(tr);
  specChart.set({series: ser, guides, refLines: [{y: NOISE_PSD, label: 'typical phone/watch sensor noise'}], yMin: Math.pow(10, Math.floor(Math.log10(Math.min(lo, NOISE_PSD * 0.5)))), yMax: Math.pow(10, Math.ceil(Math.log10(hi)))});
  explain();
}
/* numbers for the habitat floor's narrative, all read from the bundled result files */
function floorFacts(planet, sensor){
  const O = DATA.outcomes_S2, H = DATA.hsi && DATA.hsi.data ? DATA.hsi.data.S2 : null, d = O && O.data.stand_E1 ? O.data.stand_E1[planet] : null;
  return {occ: O && O.occupied ? O.occupied[planet] : null, occTrials: d ? d.occupied_B1_from_trials : null, trans: d && d[sensor] ? d[sensor].trans : null,
          zEmpty: H ? H.empty.zeta : null, zPerson: H && H.bodies.nominal[planet] ? H.bodies.nominal[planet].zeta : null};
}
function explain(){
  const tr = cur, st = tr.stats, P = tr.planet, sen = SENSOR_NAME[state.sensor].toLowerCase(), walking = tr.activity === 'walking', out = [];
  const g = SG(tr.structure), floor = isFloor(tr.structure), kg = Math.round(bodyMass(tr));
  const ratio = (st.deck_rms_z ? st[state.sensor + '_rms'] / st.deck_rms_z : null);
  const modes = (g ? g.modes : []).filter(m => m.f < 30);
  if (!walking){
    const r = standingRow(P), rE = standingRow('Earth');
    if (floor) out.push('A ' + kg + ' kg person stands still on the habitat floor, ' + fmt(tr.standX !== undefined ? tr.standX : g.stand[0], 3) + ' m from the wall on the floor’s beam line, while three shakers under the floor make it vibrate. The same shaker forces are used on every planet.');
    else out.push('A ' + kg + ' kg person stands still on the reference bridge (the earlier study’s ' + meters(g ? g.L : 14) + ' m bridge model) while three shakers make it vibrate. The same shaker forces are used on every planet.');
    if (r && rE) out.push(P === 'Earth' ? 'On Earth the body “bounces” on its legs and foot soles at about <b>' + fmt(r.apparent_mass_f, 3) + ' Hz</b>. This value was set to agree with published measurements on standing people.'
      : 'On ' + onPlanet(P) + ' the person weighs ' + Math.round(100 * tr.g / G_E) + ' % of Earth weight. Legs and soles are squeezed less, which makes them softer, so the body “bounces” at about <b>' + fmt(r.apparent_mass_f, 3) + ' Hz</b> instead of ' + fmt(rE.apparent_mass_f, 3) + ' Hz on Earth.');
    if (floor){
      const F = floorFacts(P, state.sensor);
      out.push('Vibration of the structure slower than this bounce frequency passes into the body almost unchanged; faster vibration is filtered out. Both of the floor’s natural frequencies below 30 Hz, '
        + modes.map(m => m.label + ' ' + fmt(m.f, 4) + ' Hz').join(' and ') + ', lie above it, so the body passes on only part of them'
        + (F.trans ? ': in the standing trials on ' + onPlanet(P) + ' the ' + sen + ' receives <b>' + F.trans.map((v, i) => fmt(v, 2) + '</b> of ' + (modes[i] ? modes[i].label : '')).join(' and <b>') + ' (three axes, relative to the floor’s vertical motion under the feet).' : '.'));
      if (F.occ && F.occ.B1 && F.zPerson !== null) out.push('The floor is light (' + fmt(g.m_area, 3) + ' kg per m²), so the person becomes part of it: standing here moves B1 from ' + fmt(modes[0].f, 4) + ' Hz to about <b>' + fmt(F.occ.B1, 4) + ' Hz</b> (predicted before the trials'
        + (F.occTrials ? '; ' + fmt(F.occTrials, 4) + ' Hz in these trials' : '') + ') and raises its damping from ' + pct(F.zEmpty, 0) + ' to <b>' + pct(F.zPerson, 1) + '</b>. That is why the floor’s peak sits just above the first dashed line in the frequency chart.');
    } else out.push('Vibration of the structure slower than this bounce frequency passes into the body almost unchanged; faster vibration is filtered out. That is why the lowest frequencies of the structure (B1, B2) show clearly in the sensor curve and the higher ones fade.');
    if (ratio) out.push('Here the ' + sen + ' vibrates about <b>' + fmt(ratio, 2) + ' times</b> as much as the structure under the feet (all frequencies together).');
  } else {
    const what = tr.exc === 'E1' ? 'while three shakers make it vibrate (same forces on every planet)' : tr.exc === 'E2' ? 'and nothing but the footsteps moves it' : 'on solid ground that cannot move (comparison case)';
    if (floor){
      out.push('A ' + kg + ' kg person walks across the habitat floor along its beam line, from wall to wall (' + meters(g.L) + ' m, with a central support at mid-span), ' + what + '.');
      if (tr.xStart !== undefined) out.push('The walk starts on rigid ground ' + fmt(-tr.xStart, 2) + ' m before the floor, so that the gait has settled when the walker steps onto it, and ends on rigid ground past the far wall. The drawing shows this ground as an entry platform and a landing reached through openings in the vessel wall; they are schematic.');
    } else out.push('A ' + kg + ' kg person walks across ' + (tr.exc === 'rigid' ? 'the same line ' : 'the reference bridge (the earlier study’s ' + meters(g ? g.L : 14) + ' m bridge model) ') + what + '.');
    out.push('Walking speed <b>' + fmt(st.speed, 3) + ' m/s</b>, <b>' + fmt(st.step_frequency, 3) + ' steps per second</b>; each foot is on the ground ' + pct(st.duty_factor) + ' of the time and the largest force under a foot is ' + fmt(st.grf_foot_peak_over_W, 3) + ' × the local weight.'
      + (st.flight_fraction > 0.02 ? ' For ' + pct(st.flight_fraction) + ' of the time both feet are in the air: this is a bouncing gait, not a normal walk.' : '')
      + (st.swing_scuffs_per_step > 0.2 ? ' The swinging foot briefly brushes the ground early in each swing (a toe scuff).' : ''));
    if (tr.cond === 'earth_1.25') out.push('This is the Earth gait tuned for 1.25 m/s, the gait from which the what-if gaits for Mars and the Moon are scaled.');
    else if (tr.cond.indexOf('similar') >= 0) out.push('This is the Earth gait replayed in slow motion for the weaker gravity (equal Froude number). It is a what-if case: experiments show that people do not actually walk this way in reduced gravity.'
      + ' The study analyses it on both structures.');
    else if (P !== 'Earth') out.push('The walking pattern was re-tuned for ' + onPlanet(P) + ' so that speed, step rate and time on the ground follow published reduced-gravity experiments. Forces and body motion are then whatever the physics gives.');
    else out.push('The Earth gait was tuned to published numbers for human walking (speed, step rate, time on the ground, force under the foot).');
    if (tr.model3d) out.push('This is the <b>full 3-D walker</b>: it keeps its sideways balance by choosing where to put each foot, so the body sways gently from side to side'
      + (st.step_width !== undefined ? ' (feet ' + fmt(100 * st.step_width, 2) + ' cm apart sideways; largest sideways force ' + fmt(st.grf_lateral_peak_over_W, 2) + ' × weight)' : '')
      + '. It was tuned on each planet (same forward targets as the side-view walker; sideways targets taken from Earth, because no reduced-gravity step-width data exist), so the Earth–Mars–Moon comparison can be made with both walkers.');
    else if (tr.cond === P.toLowerCase() + '_1.00') out.push('This is the side-view walker (no sideways motion). Choose <b>Walker: 3-D</b> above to see the full 3-D walker on this planet.');
    if (tr.exc !== 'rigid') out.push('In the frequency chart, compare the coloured curve with the grey dashed one (same walk on solid ground' + (floor ? ', cut to the floor’s crossing' : '') + '). Where the coloured curve rises above the grey one at a dashed line, the sensor is picking up the structure.');
  }
  out.push('<b>Mars and Moon cases are model extrapolations</b>; they have not been compared with measurements.');
  document.getElementById('explain').innerHTML = '<h3>Notes on this trial</h3><ul>' + out.map(s => '<li>' + s + '</li>').join('') + '</ul>';
}
function captionOf(tr){
  const walking = tr.activity === 'walking';
  return (tr.structure && STRUCT[tr.structure] && tr.exc !== 'rigid' ? STRUCT[tr.structure].short + ' · ' : '')
    + (walking ? 'walking at ' + fmt(tr.stats.speed, 3) + ' m/s' + (tr.exc === 'E1' ? ', shakers on' : tr.exc === 'E2' ? ', footsteps only' : ', solid ground') : 'standing still, shakers on');
}
let lastWall = null;
function frame(now){
  requestAnimationFrame(frame);
  const dt = lastWall === null ? 0 : Math.min(0.1, (now - lastWall) / 1000); lastWall = now;
  if (!cur || document.getElementById('tab-watch').hidden) return;
  if (state.playing){ state.t += dt * state.rate; if (state.t > cur.dur) state.t = 0; }
  const u = state.t * cur.fsAnim, i = Math.floor(u);
  const walking = cur.activity === 'walking', floor = isFloor(cur.structure);
  const caption = captionOf(cur);
  const trueSize = cur.deckX ? 'True size (the ' + (floor ? 'floor' : 'structure') + ' moves by ' + (cur.wMax < 1e-3 ? 'fractions of a millimetre' : 'up to ' + fmt(1000 * cur.wMax, 2) + ' mm') + ' here)' : '';
  if (use3d()){
    const ttl = document.getElementById('ovTitle');
    ttl.innerHTML = cur.planet + ' · g = ' + cur.g.toFixed(2) + ' m/s²<small>' + caption + (cur.model3d ? ' · 3-D walker' : (walking ? ' · side-view walker' : '')) + '</small>';
    ttl.classList.toggle('dark', cur.planet === 'Moon' && !floor);
    const kd = cur.deckX ? state.exag : 1;
    document.getElementById('ovNote').textContent = (cur.deckX ? (kd > 1 ? (floor ? 'Floor motion drawn ×' + kd + ' (only the modelled ' + meters(SG('S2').W) + ' m strip moves)' : 'Structure motion drawn ×' + kd) : trueSize) : 'Solid ground, no structure') + '  ·  arrows: force under each foot';
    document.getElementById('ovNote').classList.toggle('ink', floor);
    scene3d.draw({i, a: u - i, exag: kd, view: state.view, sensor: state.sensor, t: state.t, wall: now / 1000});
  } else
  scene.draw({i, a: u - i, exag: cur.deckX ? state.exag : 1, view: state.view === 'full' ? 'full' : 'follow', sensor: state.sensor, t: state.t, wall: now / 1000,
    caption, trueSize});
  document.getElementById('seek').value = Math.round(1000 * state.t / cur.dur);
  document.getElementById('clock').textContent = state.t.toFixed(2) + ' s';
  const half = walking ? 3 : 1.5, x0 = Math.max(0, Math.min(cur.dur - 2 * half, state.t - half));
  for (const k in tcharts){ const ch = tcharts[k]; if (ch.card.card.style.display === 'none') continue; ch.o.xMin = x0; ch.o.xMax = x0 + 2 * half; ch.o.cursorX = state.t; ch.draw(); }
}

/* a table card with a CSV button */
function makeTable(parent, o){
  const card = el('div', {class: 'card'}), tools = el('div', {class: 'tools'});
  card.appendChild(el('div', {class: 'head'}, [el('div', null, [el('div', {class: 'ttl', text: o.title}), o.desc ? el('div', {class: 'desc', html: o.desc}) : null]), tools]));
  const t = el('table', {class: 't'}), th = el('tr');
  o.columns.forEach((c, i) => th.appendChild(el('th', {html: c, class: (o.left || [0]).includes(i) ? 'l' : ''})));
  t.appendChild(el('thead', null, [th])); const tb = el('tbody');
  o.rows.forEach(r => { const row = el('tr', r.sep ? {class: 'sep'} : null); (r.cells || r).forEach((c, i) => row.appendChild(el('td', {html: c === null || c === undefined ? '–' : String(c), class: (o.left || [0]).includes(i) ? 'l' : ''}))); tb.appendChild(row); });
  t.appendChild(tb); card.appendChild(el('div', {class: 'tscroll'}, [t]));
  if (o.note) card.appendChild(el('div', {class: 'note', html: o.note}));
  tools.appendChild(el('button', {class: 'btn small', text: 'Save data', onclick: () => {
    const q = v => '"' + String(v === null || v === undefined ? '' : v).replace(/<[^>]+>/g, '').replace(/"/g, '""') + '"';
    const csv = [o.columns.map(q).join(',')].concat(o.rows.map(r => (r.cells || r).map(q).join(','))).join('\n');
    download((o.file || 'table') + '.csv', 'data:text/csv;charset=utf-8,' + encodeURIComponent(csv)); }}));
  parent.appendChild(card); return card;
}
function section(parent, title, lead){ parent.appendChild(el('h2', null, [el('span', {text: title})]));   /* underline the words only, not the detail bubble */ if (lead) parent.appendChild(el('p', {class: 'lead', html: lead})); }
function planetSeries(fn){ return PLANETS.map(p => Object.assign({label: p, color: PAL[p]}, fn(p))); }
