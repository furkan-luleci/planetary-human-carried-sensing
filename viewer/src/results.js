/* ------------------------------------------------------ results pages: shared helpers */
const VARIANT = {
  nominal: 'Nominal body (legs stiffen under load)',
  earth_fixed: 'Muscle stiffness kept at Earth level',
  leg_n1: 'Legs that do not stiffen (linear)',
  leg_n2: 'Legs that stiffen strongly',
  soft_damped: 'Softer, more damped body (5.2 Hz, 39 %)',
  stiff_light: 'Stiffer, lightly damped body (6.6 Hz, 22 %)',
  prop_damping: 'Damping that follows the leg force',
  fully_linear: 'Control: nothing in the body follows weight',
  rigid_leg: 'First prototype (rigid legs, not calibrated)'};
function modeCats(modes){ return modes.map(m => m.label + '\n' + fmt(m.f, 3) + ' Hz'); }
function missing(parent, what){ parent.appendChild(el('div', {class: 'caveat', text: what + ' has not been computed in this build of the viewer.'})); }
/* the result files of each structure (the habitat floor is the main case) */
function outcomesOf(k){ return k === 'S2' ? DATA.outcomes_S2 : DATA.outcomes; }
function mountingOf(k){ return k === 'S2' ? DATA.mounting_S2 : DATA.mounting; }
function timingOf(k){ return k === 'S2' ? DATA.step_timing_S2 : DATA.step_timing; }
function clocksOf(k){ return k === 'S2' ? DATA.step_timing_clocks_S2 : DATA.step_timing_clocks; }
function jitterOf(k){ return k === 'S2' ? DATA.step_timing_jitter_S2 : DATA.step_timing_jitter; }
function haveStruct(k){ return !!outcomesOf(k); }
const DECK_ONLY = ' <span class="pill deck">reference bridge only</span>';
/* a structure switch (habitat floor first); items without data are left out */
function structControl(parent, label, value, onChange){
  const c = segControl(parent, label, STRUCTS.filter(haveStruct).map(k => ({v: k, html: STRUCT[k].html, title: structTitle(k)})), value, onChange);
  c.seg.classList.add('struct'); return c;
}
/* the structure's modes below 30 Hz (with the frequencies the result file was analysed with) */
function modesOf(k){ const O = outcomesOf(k); return O ? O.modes : ((SG(k) || {}).modes || []).filter(m => m.f < 30); }
/* blind recovery of one cell, by the rule that applies to the structure: on the habitat floor a found line also counts at the
   mode's occupied frequency (declared before the trials: rec_alt), on the reference bridge at its frequency (rec) */
function recOf(k, d){
  if (!d) return null;
  const n = modesOf(k).length;
  if (k === 'S2'){
    const r = d.rec_alt, c = d.rec_alt_chance; if (!r) return null;
    return {tp: r.tp, fp: r.fp, n, f1: r.f1, recall: r.tp / n, precision: r.tp + r.fp > 0 ? r.tp / (r.tp + r.fp) : 0, chance: c ? c.tp : null, chanceRecall: c ? c.tp / n : null,
            emptyTp: d.rec ? d.rec.tp : null, emptyFp: d.rec ? d.rec.fp : null};
  }
  const r = d.rec, c = d.rec_chance; if (!r) return null;
  return {tp: r.tp, fp: r.fp, n: r.tp + r.fn, f1: r.f1, recall: r.recall, precision: r.precision, chance: c ? c.tp : null, chanceRecall: c ? c.recall : null};
}
/* which line was counted for which mode on the habitat floor (empty or occupied frequency), and the false alarms left over */
function floorFound(d, modes, occ){
  const r = d.rec || {}, ra = d.rec_alt || {}, tol = f => Math.max(0.15, 0.03 * f), used = new Set();
  const rows = modes.map(m => {
    const pr = (r.pairs || []).find(q => Math.abs(q[0] - m.f) < 1e-6);
    if (pr) return {f: pr[1], at: 'empty', ref: m.f};
    if ((ra.matched || []).some(v => Math.abs(v - m.f) < 1e-6)){
      const fo = occ ? occ[m.label] : null, fa = (r.false_alarms || []).find(v => fo && !used.has(v) && Math.abs(v - fo) <= tol(fo));
      if (fa !== undefined){ used.add(fa); return {f: fa, at: 'occupied', ref: fo}; }
      return {f: null, at: 'occupied', ref: fo};
    }
    return null;
  });
  return {rows, falseAlarms: (r.false_alarms || []).filter(v => !used.has(v))};
}
/* natural frequencies drawn as chart guides; on the habitat floor also the occupied frequencies (ochre) */
function guidesOf(k, fmax, planets){
  const out = modesOf(k).filter(m => m.f < (fmax || 30)).map(m => ({x: m.f, label: m.label}));
  const O = DATA.outcomes_S2;
  if (k === 'S2' && O && O.occupied) for (const p of (planets || PLANETS)) for (const lab in (O.occupied[p] || {})) if (O.occupied[p][lab] < (fmax || 30)) out.push({x: O.occupied[p][lab], color: PAL.ochre, dash: [2, 2]});
  return out;
}
const OCC_NOTE = 'Amber dotted lines: the same modes with a person standing on the floor, predicted before the trials.';
/* how far a standing person shifts the reference bridge's first mode (linear model, every body and planet) */
function deckShift(){ const H = DATA.hsi && DATA.hsi.data ? DATA.hsi.data.S1 : null; if (!H) return '';
  let m = 0; for (const v in H.bodies) for (const p in H.bodies[v]) if (H.bodies[v][p]) m = Math.max(m, Math.abs(H.bodies[v][p].f / H.empty.f - 1));
  return ' (a person standing on it shifts the first one by at most ' + fmt(100 * m, 2) + ' %)'; }
/* reader-facing note of a situation; the habitat floor's own geometry for its standing point */
function scenNote(k, sc){
  if (!sc) return '';
  if (k === 'S2' && sc.id === 'stand_E1'){ const g = SG('S2'); return 'The person stands still ' + fmt(g.stand[0], 3) + ' m from the wall, on the floor’s beam line; three shakers drive the floor with the same forces on every planet.'; }
  return (sc.note || '').replace(/^Phase [A-Z]: /, '');
}

/* ------------------------------------------------------ 2  WHAT GRAVITY CHANGES */
function buildGravity(){
  const root = document.getElementById('tab-gravity'), S = DATA.standing, g2s = SG('S2'), g1s = SG('S1');
  root.appendChild(el('p', {class: 'lead', html: 'The same ' + bodyKg() + ' kg body is simulated under Earth, Mars and Moon gravity on two structures, each the same on every planet: the main case, a floor of NASA’s Common Habitat (the <b>habitat floor</b>), and the <b>reference bridge</b>, the bridge model of the earlier study, used to relate the results to the earlier studies. Only the weight changes. This page shows what that does to the body, to the way it walks, to the floor under a standing person, and to how much of the structure’s vibration reaches a worn device.'}));
  /* ---- standing */
  section(root, 'Standing body: vertical resonance on each planet', 'A standing person is a mass on springs (legs, joints, foot soles). The frequency at which this system resonates decides which vibrations of the structure reach the thigh pocket or the wrist.');
  if (!S){ missing(root, 'The standing study'); }
  else {
    const row = (p, v) => S.rows.find(r => r.gravity === p && r.variant === v);
    const nom = PLANETS.map(p => row(p, 'nominal')), vs = Object.keys(VARIANT).filter(v => v !== 'rigid_leg' && v !== 'fully_linear' && row('Moon', v));
    const rng = p => { const a = vs.map(v => row(p, v).apparent_mass_f); return [Math.min(...a), Math.max(...a)]; };
    const rM = rng('Mars'), rL = rng('Moon'), lin = row('Moon', 'fully_linear');
    root.appendChild(el('div', {class: 'key', html: 'On Earth the standing body resonates vertically at <b>' + fmt(nom[0].apparent_mass_f, 3) + ' Hz</b> (set to agree with published measurements). With the same body the model gives <b>' + fmt(nom[1].apparent_mass_f, 3) + ' Hz on Mars</b> and <b>' + fmt(nom[2].apparent_mass_f, 3) + ' Hz on the Moon</b>. '
      + 'Across the modelling assumptions tested, the range is ' + fmt(rM[0], 3) + '–' + fmt(rM[1], 3) + ' Hz on Mars and ' + fmt(rL[0], 3) + '–' + fmt(rL[1], 3) + ' Hz on the Moon.'
      + (lin ? ' In the control case, where nothing in the body can follow the weight, the frequency does not drop (' + fmt(row('Earth', 'fully_linear').apparent_mass_f, 3) + ' → ' + fmt(lin.apparent_mass_f, 3) + ' Hz): the shift comes from legs and soles being softer when they carry less weight.' : '')}));
    const g2 = el('div', {class: 'grid2'}); root.appendChild(g2);
    // transmissibility curves (a property of the body; the dashed lines mark the chosen structure's natural frequencies)
    const st = {sensor: 'thigh', variant: 'nominal', struct: STRUCTS.find(haveStruct) || 'S1'};
    const c1 = makeCard(g2, {title: 'Floor-to-device transmissibility (standing)', desc: 'Device vibration divided by floor vibration, vertical direction. 1 = same as the floor; above 1 = amplified.', height: 320, file: 'standing_transmissibility',
      note: 'A property of the body: small-vibration (linear) analysis of the model, from the vertical motion under the feet to the device. Dashed lines mark the natural frequencies of the structure chosen above the chart.'});
    const ch1 = new Chart(c1, {xLog: true, xMin: 0.5, xMax: 40, yMin: 0, xLabel: 'vibration frequency (Hz)', yLabel: 'device ÷ floor', xUnit: 'Hz', refLines: [{y: 1, label: 'same as floor'}]});
    const up1 = () => ch1.set({guides: ((SG(st.struct) || {}).modes || modesOf(st.struct)).filter(m => m.f < 40).map(m => ({x: m.f, label: m.label})), series: planetSeries(p => ({x: S.f_base, y: decF32(S.curves['T|' + st.variant + '|' + p + '|' + st.sensor + '_L_zz']), width: 2.2}))});
    structControl(c1.opts, null, st.struct, v => { st.struct = v; up1(); });
    segControl(c1.opts, null, ['wrist', 'thigh'].map(k => ({v: k, label: SENSOR_FULL[k]})), st.sensor, v => { st.sensor = v; up1(); });
    const sel = el('select', {onchange: e => { st.variant = e.target.value; up1(); }}); Object.keys(VARIANT).forEach(v => { if (row('Earth', v)) sel.appendChild(el('option', {value: v, text: VARIANT[v]})); });
    c1.opts.appendChild(el('label', null, ['Body assumption ', sel])); up1();
    // resonance versus planet for every assumption
    const c2 = makeCard(g2, {title: 'Body resonance on each planet, for every modelling assumption', desc: 'Each line is one assumption about how legs and muscles behave. All agree on Earth by construction; their spread elsewhere is the uncertainty.', height: 320, file: 'standing_resonance_by_assumption',
      note: 'The resonance is the frequency of the single mass-on-a-spring that best matches the body’s response (apparent mass, 0.5–20 Hz).'});
    const ser = Object.keys(VARIANT).filter(v => row('Earth', v) && v !== 'rigid_leg').map(v => ({x: [0, 1, 2], y: PLANETS.map(p => row(p, v).apparent_mass_f), label: VARIANT[v], endLabel: fmt(row('Moon', v).apparent_mass_f, 3),
      color: v === 'nominal' ? PAL.ink : v === 'fully_linear' ? PAL.Earth : PAL.muted, width: v === 'nominal' ? 3 : 1.6, marker: v === 'nominal' ? 4.5 : 3, dash: v === 'fully_linear' ? [5, 3] : null}));
    new Chart(c2, {series: ser, xCats: PLANETS.map(p => p + '\n'), xMin: -0.25, xMax: 2.45, yLabel: 'body resonance (Hz)', endLabels: true, right: 44, unit: 'Hz', legend: true}).draw();
    makeTable(root, {title: 'Table: equivalent resonance and damping of the standing body', desc: 'Frequency (Hz) and damping (% of critical) of the best-matching single mass-on-a-spring.', file: 'standing_body_resonance',
      columns: ['Body assumption', 'Earth f', 'Earth damping', 'Mars f', 'Mars damping', 'Moon f', 'Moon damping', 'Moon ÷ Earth f'],
      rows: Object.keys(VARIANT).filter(v => row('Earth', v)).map(v => [VARIANT[v]].concat(PLANETS.flatMap(p => [fmt(row(p, v).apparent_mass_f, 3), fmt(100 * row(p, v).apparent_mass_zeta, 2) + ' %']), [fmt(row('Moon', v).apparent_mass_f / row('Earth', v).apparent_mass_f, 3)])),
      note: 'Earth values of the first seven rows are calibration targets taken from published standing-vibration studies; they are inputs, not predictions.'});
  }
  /* ---- the person becomes part of the habitat floor */
  buildOccupant(root);
  /* ---- walking */
  section(root, 'Gaits on each planet', 'The walker is not a replayed motion. A simple balance controller drives the joints and the steps come out of the physics. Its settings were tuned on Earth to published gait numbers, then adjusted for Mars and the Moon. The 1.0 m/s gaits walked on both structures; the 1.25 m/s Earth gait and the what-if gaits were simulated on the reference bridge only.');
  const Gt = DATA.gait || {}, ids = ['earth_1.25', 'earth_1.00', 'mars_1.00', 'moon_1.00', 'mars_0.77_similar', 'moon_0.51_similar'].filter(k => Gt[k] && Gt[k].metrics);
  const deckOnlyGait = k => k === 'earth_1.25' || Gt[k].kind === 'similarity';
  if (!ids.length) missing(root, 'The walking study');
  else {
    const gm = k => (Gt[k] || {}).metrics || {};
    const same = ['earth_1.00', 'mars_1.00', 'moon_1.00'].filter(k => Gt[k]);
    if (same.length === 3) root.appendChild(el('div', {class: 'key', html: 'At the same speed (1.0 m/s) the tuned walker takes ' +
      same.map(k => fmt(gm(k).step_frequency, 3)).join(' / ') + ' steps per second on Earth / Mars / Moon, with each foot on the ground ' +
      same.map(k => pct(gm(k).duty_factor)).join(' / ') + ' of the time. Relative to the local body weight, the peak force under a foot is ' +
      same.map(k => fmt(gm(k).grf_foot_peak_over_W, 3)).join(' / ') + '; in newtons that is ' +
      same.map(k => Math.round(gm(k).grf_foot_peak_over_W * bodyKg() * Gt[k].g)).join(' / ') + ' N. ' +
      (same.some(k => gm(k).flight_fraction > 0.02) ? 'Some cases include short phases with both feet in the air.' : 'No case has a phase with both feet in the air.')}));
    const tgt = (k, name) => { const t = (Gt[k].targets || {})[name]; return t ? ' <span style="color:var(--muted)">(' + (t[2] === 'max' ? '≤ ' : '') + t[0] + ')</span>' : ''; };
    makeTable(root, {title: 'Table: walking pattern in every case', desc: 'Simulated value, with the target it was tuned to in brackets where one was used. Forces are in units of local body weight.', file: 'gait_table',
      columns: ['Case', 'Speed (m/s)', 'Step rate (1/s)', 'Step length (m)', 'Foot on ground', 'Both feet in air', 'Peak force, one foot', 'Peak force, both feet', 'Force swing at step rate', 'Up-down travel of body (cm)', 'Upper body leans forward (°)', 'Toe scuffs per step'],
      rows: ids.map(k => { const m = Gt[k].metrics; return [COND_LABEL[k] + (Gt[k].kind === 'similarity' ? ' <span class="pill">what-if</span>' : '') + (deckOnlyGait(k) ? DECK_ONLY : ''), fmt(m.speed, 3) + tgt(k, 'speed'), fmt(m.step_frequency, 3) + tgt(k, 'step_frequency'), fmt(m.step_length, 2), pct(m.duty_factor) + tgt(k, 'duty_factor'), pct(m.flight_fraction, 1), fmt(m.grf_foot_peak_over_W, 3) + tgt(k, 'grf_foot_peak_over_W'), fmt(m.grf_peak_over_W, 3) + tgt(k, 'grf_peak_over_W'), fmt((m.dlf || [])[0], 2) + tgt(k, 'dlf1'), fmt(100 * m.com_vertical_excursion, 2), (m.trunk_lean === undefined ? '–' : fmt(m.trunk_lean * 180 / Math.PI, 2) + ((Gt[k].targets || {}).trunk_lean ? ' <span style="color:var(--muted)">(≤ ' + fmt(Gt[k].targets.trunk_lean[0] * 180 / Math.PI, 2) + ')</span>' : '')), fmt(m.swing_scuffs_per_step || 0, 2)]; }),
      note: 'The gaits were tuned on solid ground; these numbers do not depend on the structure. “Force swing at step rate” is the first dynamic load factor used in the design of footbridges and floors (human walking on Earth: about 0.3–0.4 at this pace). “What-if” rows replay the Earth controller in slow motion (equal Froude number); published harness experiments show people do not walk like that in low gravity. Rows marked “reference bridge only” were walked on the reference bridge only.'});
    const Pf = DATA.profiles;
    if (Pf){
      const g3 = el('div', {class: 'grid2'}); root.appendChild(g3);
      const mkp = (keys, title, file) => { const c = makeCard(g3, {title, desc: 'Vertical force under one foot from touch-down to lift-off, averaged over steps, in units of local body weight.', height: 280, file});
        new Chart(c, {series: keys.filter(k => Pf[k]).map(k => ({x: Pf.pct, y: Pf[k].fz, label: COND_LABEL[k], color: PAL[Pf[k].planet], dash: k === 'earth_1.25' && keys.includes('earth_1.00') ? [5, 3] : null, width: 2.2})), xLabel: 'time on the ground (% of contact)', yLabel: '× body weight', yMin: 0, unit: '× weight', xUnit: '%'}).draw(); };
      mkp(['earth_1.00', 'mars_1.00', 'moon_1.00'], 'Force under one foot: same speed on every planet (both structures)', 'gait_force_same_speed');
      mkp(['earth_1.25', 'mars_0.77_similar', 'moon_0.51_similar'], 'Force under one foot: the Earth 1.25 m/s gait scaled for gravity (what-if)', 'gait_force_similar');
    }
  }
  /* ---- how strongly the walker shakes the structure */
  if (outcomesOf('S2') || outcomesOf('S1')){
    const st = {struct: STRUCTS.find(haveStruct)};
    const IDS = [['walk_same_E2', 'Walking 1.0 m/s\n(side-view walker)'], ['walk3d_E2', '3-D walker\n1.0 m/s'], ['walk_similar_E2', 'Earth gait scaled\n(what-if)']];
    const c = makeCard(root, {title: 'Structure vibration caused by the walker', desc: 'Footsteps only (no shakers): typical size of the structure’s vertical vibration under the walker, in mm/s².', height: 260, file: 'structure_vibration_footsteps', note: ' '});
    const ch = new Chart(c, {type: 'bar', yLabel: 'mm/s², RMS', valueFmt: v => fmt(v, 3), unit: 'mm/s²'});
    const up = () => { const O = outcomesOf(st.struct), ids = IDS.filter(q => O && O.data[q[0]]);
      c.note.innerHTML = 'Mean over all crossings. The walking table above gives the step rates and foot forces behind these numbers.' + (st.struct === 'S2' ? ' The habitat floor is light (' + fmt(g2s.m_area, 3) + ' kg per m²; the reference bridge: ' + fmt(g1s.m_area, 3) + ' kg per m²), so the walker’s own steps shake it far more than the bridge.' : ' The what-if gait was walked on the reference bridge only.');
      ch.set({cats: ids.map(q => q[1]), series: planetSeries(p => ({values: ids.map(q => { const d = (O.data[q[0]] || {})[p]; return d && d.deck_rms_under ? 1000 * d.deck_rms_under.mean : null; })}))}); };
    structControl(c.opts, null, st.struct, v => { st.struct = v; up(); }); up();
  }
  /* ---- transmission per mode */
  section(root, 'Transmission from the structure to the devices', 'For every natural frequency of the structure: device vibration divided by the vibration of the structure under the person, while the person stands still. Above 1 the body amplifies it, below 1 it filters it out.');
  if (outcomesOf('S2') || S){
    const st = {sensor: 'thigh', struct: STRUCTS.find(k => k === 'S2' ? !!DATA.outcomes_S2 : !!(S || DATA.outcomes)), how: DATA.outcomes ? 'trials' : 'lin'};
    const c = makeCard(root, {title: 'Share of the structure’s vibration that reaches the device, by natural frequency (standing)', desc: 'Bars above 1: amplified by the body. Bars below 1: filtered out.', height: 300, file: 'transmission_by_mode', note: ' '});
    const ch = new Chart(c, {type: 'bar', yLabel: 'device ÷ structure', refLines: [{y: 1, label: 'same as structure'}], valueFmt: v => fmt(v, 2)});
    const keyT = el('div', {class: 'key'}); c.card.parentNode.insertBefore(keyT, c.card);
    let howCtl = null;
    const up = () => {
      const floor = st.struct === 'S2', O = outcomesOf(st.struct);
      if (howCtl) howCtl.seg.style.display = floor ? 'none' : '';
      let modes, series;
      if (floor){
        modes = O.modes;
        series = planetSeries(p => ({values: modes.map((m, i) => { const d = ((O.data.stand_E1 || {})[p] || {})[st.sensor]; return d && d.trans ? d.trans[i] : null; })}));
        c.note.innerHTML = 'Habitat floor: estimated from the standing trials (shakers on, ' + ((O.data.stand_E1 || {}).Earth || {}).n_records + ' records of ' + standTrialsText('S2') + ') with the known shaker forces as reference: the part of each signal that follows the shakers. The device’s three axes combined, relative to the vertical motion of the floor under the feet. The habitat floor has no small-vibration (linearized) estimate per mode. <b>Why no walking bars?</b> While walking, the part of the device signal caused by the structure is smaller than the walker’s own step-to-step variability, so it cannot be separated reliably (see page 3 for what can still be detected).';
      } else if (st.how === 'lin' && S){
        modes = S.deck_modes.filter(m => m.f < 30);
        series = planetSeries(p => ({values: modes.map(m => { const r = S.rows.find(q => q.gravity === p && q.variant === 'nominal'), d = r.deck_modes.find(q => q.mode === m.label); return d ? d[st.sensor + '_L_3axis_over_floor_z'] : null; })}));
        c.note.innerHTML = 'Reference bridge only: exact small-vibration (linearized) analysis of the person standing on the bridge. The device’s three axes combined, relative to the vertical motion of the bridge under the feet. The same quantity from the standing trials agrees within a few percent (choose “from the standing trials”).';
      } else {
        modes = O.modes;
        series = planetSeries(p => ({values: modes.map((m, i) => { const d = ((O.data.stand_E1 || {})[p] || {})[st.sensor]; return d && d.trans ? d.trans[i] : null; })}));
        c.note.innerHTML = 'Reference bridge: estimated from the standing trials (' + standTrialsText('S1') + ') using the known shaker forces as reference. The device’s three axes combined, relative to the vertical motion of the bridge under the feet.';
      }
      ch.set({cats: modeCats(modes), series});
      const tr = p => (((outcomesOf('S2') || {data: {}}).data.stand_E1 || {})[p] || {})[st.sensor];
      if (DATA.outcomes_S2 && tr('Earth') && tr('Moon')){
        const fall = DATA.outcomes_S2.modes.map((m, i) => 1 - tr('Moon').trans[i] / tr('Earth').trans[i]);
        keyT.innerHTML = '<b>Habitat floor.</b> Both of the floor’s modes lie above the body’s resonance on every planet, so both reach the ' + SENSOR_NAME[st.sensor].toLowerCase() + ' only in part: ' +
          DATA.outcomes_S2.modes.map((m, i) => m.label + ' ' + PLANETS.map(p => fmt(tr(p).trans[i], 2)).join(' / ')).join(', ') + ' on Earth / Mars / Moon. From Earth to the Moon the transmission falls by ' + fmt(100 * Math.min(...fall), 2) + '–' + fmt(100 * Math.max(...fall), 2) + ' %.';
      } else keyT.hidden = true;
    };
    structControl(c.opts, null, st.struct, v => { st.struct = v; up(); });
    if (DATA.outcomes && S) howCtl = segControl(c.opts, null, [{v: 'trials', label: 'from the standing trials'}, {v: 'lin', label: 'exact (linearized) analysis'}], st.how, v => { st.how = v; up(); });
    segControl(c.opts, null, ['wrist', 'thigh'].map(k => ({v: k, label: SENSOR_FULL[k]})), st.sensor, v => { st.sensor = v; up(); }); up();
  }
  root.appendChild(el('div', {class: 'caveat', html: '<b>Read with care.</b> Earth results rest on calibration to published human data. Mars and Moon results are extrapolations of the same model. No measurement in reduced gravity was used or reproduced. The person is unsuited, as inside a pressurized habitat.'}));
}

/* the person standing on the habitat floor shifts and damps its first mode (linear model and standing trials) */
function buildOccupant(root){
  const HS = DATA.hsi, OC = DATA.occupied_curves, O2 = DATA.outcomes_S2, K2 = DATA.phaseK ? DATA.phaseK.structures.S2 : null;
  if (!HS || !HS.data || !HS.data.S2) return;
  const H2 = HS.data.S2, H1 = HS.data.S1, g2 = SG('S2'), g1 = SG('S1');
  section(root, 'The standing occupant as part of the habitat floor', 'The habitat floor is light (' + fmt(g2.m_area, 3) + ' kg per m²; the reference bridge: ' + fmt(g1.m_area, 3) + ' kg per m²). A person standing on it moves with the floor’s first mode, shifts it and damps it. How much depends on how stiff the legs are, and that depends on the weight they carry.');
  const cal = Object.keys(H2.bodies).filter(v => v !== 'fully_linear' && v !== 'rigid_leg');
  const zr = (H, p) => { const a = cal.map(v => H.bodies[v][p].zeta); return [Math.min(...a), Math.max(...a)]; };
  const nomZ = p => H2.bodies.nominal[p].zeta, ctl = H2.bodies.fully_linear;
  const z1 = PLANETS.flatMap(p => (H1 ? cal.map(v => H1.bodies[v][p].zeta) : [])), f1 = PLANETS.flatMap(p => (H1 ? cal.map(v => H1.bodies[v][p].f) : []));
  const fr = (a, b) => fmt(a, 3) === fmt(b, 3) ? fmt(a, 3) : fmt(a, 3) + '–' + fmt(b, 3);
  root.appendChild(el('div', {class: 'key', html: 'With a person standing ' + meters(H2.standing_point[0]) + ' m from the wall, the floor’s first mode (' + fmt(H2.empty.f, 4) + ' Hz empty) moves to <b>' + PLANETS.map(p => fmt(H2.bodies.nominal[p].f, 4)).join(' / ') + ' Hz</b> on Earth / Mars / Moon, and its damping rises from ' + pct(H2.empty.zeta, 0) + ' to <b>' + PLANETS.map(p => pct(nomZ(p), 1)).join(' / ') + '</b> (nominal body; across the calibrated bodies '
    + PLANETS.map(p => { const r = zr(H2, p); return pctRange(r[0], r[1]); }).join(', ') + '). The added damping falls with gravity. It does not fall in the control body, in which nothing follows the load (' + PLANETS.map(p => pct(ctl[p].zeta, 1)).join(' / ') + ').'
    + (H1 ? ' On the reference bridge nothing of this happens: its first mode stays at ' + fr(Math.min(...f1), Math.max(...f1)) + ' Hz with ' + pctRange(Math.min(...z1), Math.max(...z1)) + ' damping on every planet and for every body.' : '')}));
  const g = el('div', {class: 'grid2'}); root.appendChild(g);
  // damping of the first mode, every body assumption
  const st = {struct: 'S2'};
  const cd = makeCard(g, {title: 'Damping of the first mode with a person standing', desc: 'Linear model, person at the standing point. Each line is one assumption about how legs and muscles behave.', height: 320, file: 'occupant_damping', note: ' '});
  const chd = new Chart(cd, {xCats: PLANETS.map(p => p + '\n'), xMin: -0.25, xMax: 2.45, yLabel: 'first-mode damping (%)', endLabels: true, right: 44, unit: '%', legend: true, yMin: 0});
  const upd = () => { const H = HS.data[st.struct]; if (!H){ chd.set({series: []}); return; }
    const ser = Object.keys(H.bodies).filter(v => v !== 'rigid_leg' && VARIANT[v]).map(v => ({x: [0, 1, 2], y: PLANETS.map(p => 100 * H.bodies[v][p].zeta), label: VARIANT[v], endLabel: fmt(100 * H.bodies[v].Moon.zeta, 2),
      color: v === 'nominal' ? PAL.ink : v === 'fully_linear' ? PAL.Earth : PAL.muted, width: v === 'nominal' ? 3 : 1.6, marker: v === 'nominal' ? 4.5 : 3, dash: v === 'fully_linear' ? [5, 3] : null}));
    let ymax = 0; ser.forEach(s => s.y.forEach(v => { ymax = Math.max(ymax, v); }));
    cd.note.innerHTML = (st.struct === 'S2' ? 'Habitat floor' : 'Reference bridge') + ': ' + pct(H.empty.zeta, 0) + ' damping when empty (amber line). Peak of the vertical response under the feet nearest the empty first mode; damping from the half-power width.';
    chd.set({series: ser, endLabels: st.struct === 'S2', refLines: [{y: 100 * H.empty.zeta, label: 'empty structure'}], yMax: Math.max(1.15 * ymax, 100 * H.empty.zeta * 1.6)}); };
  structControl(cd.opts, null, st.struct, v => { st.struct = v; upd(); upd2(); });
  // accelerance under the feet, empty and occupied
  const ca = makeCard(g, {title: 'The floor under a standing person (linear model)', desc: 'Vertical acceleration under the feet per newton of vertical force there, empty (dotted) and with the person standing (nominal body).', height: 320, file: 'occupant_accelerance', note: ' '});
  const cha = new Chart(ca, {xLabel: 'frequency (Hz)', yLabel: '(m/s²) per N', yLog: true, xUnit: 'Hz', tipFmt: v => fmt(v, 3)});
  const upd2 = () => { const C = OC && OC.linear ? OC.linear[st.struct] : null; if (!C){ cha.set({series: []}); return; }
    const f = Array.from(decF32(C.f)), y = k => Array.from(decF32(C[k]));
    const ser = [{x: f, y: y('empty'), color: PAL.ink, label: 'empty', dash: [1.5, 2.5], width: 1.4}].concat(PLANETS.map(p => ({x: f, y: y(p), color: PAL[p], label: p, width: 2})));
    if (C.Moon_control) ser.push({x: f, y: y('Moon_control'), color: PAL.ochre, label: 'Moon, control body', dash: [6, 3], width: 1.5});
    const fmax = st.struct === 'S2' ? 20 : 10;
    ca.note.innerHTML = (st.struct === 'S2' ? 'Habitat floor: the person shifts and damps B1, less so as gravity falls; in the control body (amber, Moon) nothing follows the load.' : 'Reference bridge: the empty and occupied curves overlap; a person changes nothing on a heavy concrete span.') + ' Dashed lines: the structure’s natural frequencies (empty).';
    cha.set({series: ser, xMin: f[0], xMax: Math.min(fmax, f[f.length - 1]), guides: Object.entries(C.modes).filter(([, v]) => v < fmax).map(([k, v]) => ({x: v, label: k})), yMin: undefined, yMax: undefined}); };
  upd(); upd2();
  // the standing trials: the floor's vibration under the feet, and the occupied frequency against its prediction
  const g2b = el('div', {class: 'grid2'}); root.appendChild(g2b);
  if (OC && OC.trials){
    const cp = makeCard(g2b, {title: 'Standing trials: the floor’s vibration under the feet', desc: 'Spectrum of the floor’s vertical acceleration under the feet, all standing trials combined (shakers on).', height: 300, file: 'occupant_trials',
      note: 'Dashed: the floor’s natural frequencies, empty. ' + OCC_NOTE + ' The peak near B1 is where the occupied floor actually rings in the trials.'});
    const fs = PLANETS.map(p => Array.from(OC.trials[p].f)), ys = PLANETS.map(p => Array.from(decF32(OC.trials[p].psd)));
    new Chart(cp, {xLabel: 'frequency (Hz)', yLabel: '(m/s²)² per Hz', yLog: true, xMin: fs[0][0], xMax: Math.min(20, fs[0][fs[0].length - 1]), xUnit: 'Hz', guides: guidesOf('S2', 20),
      series: PLANETS.map((p, i) => ({x: fs[i], y: ys[i], color: PAL[p], label: p + ' (' + OC.trials[p].n_records + ' records)', width: 1.8}))}).draw();
  }
  if (K2 && K2.c4_model){
    const c4 = K2.c4_model.cells;
    makeTable(g2b, {title: 'Table: the floor’s first mode with a person standing', desc: 'Frequency with the person standing: predicted before the trials (linear model, nominal body) and found in the standing trials; damping from the linear model.', file: 'occupant_first_mode',
      columns: ['Planet', 'Empty (Hz)', 'Predicted (Hz)', 'Trials (Hz)', 'Trials − predicted', 'Damping, nominal body', 'Damping, calibrated bodies', 'Damping, control body'],
      rows: PLANETS.filter(p => c4[p]).map(p => { const r = zr(H2, p); return [p, fmt(H2.empty.f, 4), fmt(c4[p].predicted, 4), fmt(c4[p].trials, 4), (c4[p].diff_rel >= 0 ? '+' : '') + fmt(100 * c4[p].diff_rel, 2) + ' %', pct(nomZ(p), 1), pctRange(r[0], r[1]), pct(ctl[p].zeta, 1)]; }),
      note: 'Declared before the trials: the frequency from the trials agrees with the prediction on every planet within the declared tolerance (' + (K2.c4_model.holds ? 'met' : 'not met') + '). Empty damping ' + pct(H2.empty.zeta, 0) + ' (assumed).'});
  }
}

/* ------------------------------------------------------ 3  CAN THE STRUCTURE BE DETECTED */
function buildDetect(){
  const root = document.getElementById('tab-detect');
  root.appendChild(el('p', {class: 'lead', html: 'Three different questions are kept apart. <b>Transmission</b>: does the vibration reach the device? (previous page) <b>Visibility</b>: does a natural frequency of the structure stand out in the device signal, next to the person’s own movement and the sensor’s noise? <b>Recovery</b>: can someone who does not know the structure read its frequencies from the worn device alone? The habitat floor is the main case; the reference bridge relates the results to the earlier studies.'}));
  const first = STRUCTS.find(haveStruct);
  if (!first){ missing(root, 'The detection study'); return; }
  const st = {struct: first, sensor: 'thigh', scen: outcomesOf(first).scenarios[0].id};
  const bar = el('div', {class: 'controls'}); root.appendChild(bar);
  const charts = [], refresh = () => charts.forEach(f => f());
  const O = () => outcomesOf(st.struct), D = p => ((O().data[st.scen] || {})[p] || {});
  const scenBox = el('div', {class: 'ctl'});
  const buildScen = () => { scenBox.innerHTML = ''; const sc = O().scenarios;
    if (!sc.some(s => s.id === st.scen)) st.scen = sc[0].id;
    scenBox.appendChild(el('span', {text: 'Situation'}));
    const c = segControl(scenBox, null, sc.map(s => ({v: s.id, label: s.label})), st.scen, v => { st.scen = v; refresh(); }); };
  structControl(bar, 'Structure', st.struct, v => { st.struct = v; buildScen(); refresh(); });
  bar.appendChild(scenBox); buildScen();
  segControl(bar, 'Device', ['wrist', 'thigh'].map(k => ({v: k, html: '<span class="sw" style="background:var(--' + k + ')"></span>' + SENSOR_FULL[k]})), st.sensor, v => { st.sensor = v; refresh(); });
  const keybox = el('div', {class: 'key'}); root.appendChild(keybox);
  /* ensemble spectra and visibility */
  section(root, 'Visibility of the natural frequencies');
  const g1 = el('div', {class: 'grid2'}); root.appendChild(g1);
  const cs = makeCard(g1, {title: 'Sensor signal on the structure, all trials combined', desc: 'One curve per planet, including simulated sensor noise. Dashed lines: natural frequencies of the structure.', height: 320, file: 'detect_spectra', note: ' '});
  const chs = new Chart(cs, {xLabel: 'frequency (Hz)', yLabel: 'strength, (m/s²)² per Hz', yLog: true, xMin: 0, xMax: 30, xUnit: 'Hz'});
  charts.push(() => { cs.note.innerHTML = 'Sensor noise ' + O().noise_density_ug + ' µg/√Hz (typical phone or watch, as in the earlier study) was added before the analysis. The three axes of the left and right device combined.' + (st.struct === 'S2' ? ' ' + OCC_NOTE : '');
    chs.set({guides: guidesOf(st.struct, 30), series: planetSeries(p => { const d = D(p).spec; return d ? {x: d.f, y: d[st.sensor], width: 2} : {x: [], y: []}; })}); });
  const cv = makeCard(g1, {title: 'Height of each natural frequency above its surroundings', desc: 'Height of the peak above the same device’s background (same activity on solid ground, with noise), in decibels. 10 dB = ten times stronger.', height: 320, file: 'detect_visibility',
    note: 'Bars near 0 dB mean the frequency is hidden. For walking, the background is the same walk on solid ground; for standing it is the sensor noise. Band: ' + tolText('S1') + ' either side of each natural frequency of the empty structure.'});
  const chv = new Chart(cv, {type: 'bar', yLabel: 'peak above background (dB)', valueFmt: v => fmt(v, 2), unit: 'dB', refLines: [{y: 6, label: 'clearly visible (6 dB)'}]});
  charts.push(() => chv.set({cats: modeCats(O().modes), series: planetSeries(p => ({values: (D(p)[st.sensor] || {}).vis_db || O().modes.map(() => null)}))}));
  /* recovery */
  section(root, 'Blind recovery of the natural frequencies');
  const g2 = el('div', {class: 'grid2'}); root.appendChild(g2);
  const cr = makeCard(g2, {title: 'Blind recovery of the structure’s frequencies', desc: 'A peak-picking rule, tuned on the reference bridge’s Earth trials only and then frozen, is applied to the worn device alone.', height: 300, file: 'detect_recovery', note: ' '});
  const chr = new Chart(cr, {type: 'bar', cats: ['Found\n(recall)', 'Correct\n(precision)', 'Overall score\n(F1)', 'Found by chance\n(no structure)'], yLabel: 'share', yMax: 1.08, valueFmt: v => pct(v)});
  charts.push(() => { cr.note.innerHTML = '“Found” = share of the true natural frequencies below 30 Hz that were identified (within ' + tolText(st.struct) + '). “Correct” = share of the reported frequencies that are real natural frequencies of the structure. “Found by chance” = the same rule applied where there is no structure at all (same activity on solid ground, or noise only): a result on the structure is only meaningful if it beats this.'
      + (st.struct === 'S2' ? ' <b>Habitat floor:</b> a reported line also counts for a mode at the mode’s frequency with the person on the floor (the occupied frequency, declared before the trials); the rule itself was not re-tuned for the floor.' : '');
    chr.set({series: planetSeries(p => { const r = recOf(st.struct, D(p)[st.sensor]); return {values: r ? [r.recall, r.precision, r.f1, r.chanceRecall] : [null, null, null, null]}; })}); });
  const tb = el('div'); g2.appendChild(tb);
  charts.push(() => { tb.innerHTML = ''; const modes = O().modes, floor = st.struct === 'S2';
    const occ = floor ? DATA.outcomes_S2.occupied : null;
    if (floor){
      const F = PLANETS.map(p => { const d = D(p)[st.sensor]; return d ? floorFound(d, modes, occ[p]) : null; });
      makeTable(tb, {title: 'Table: frequencies read from the worn device', desc: 'True natural frequency and what the blind rule reported, per planet; “occupied” = counted at the frequency with the person on the floor. “–” = not found.', file: 'detect_recovery_table',
        columns: ['Natural frequency'].concat(PLANETS.flatMap(p => [p + ' found (Hz)', p + ' counted at'])),
        rows: modes.map((m, i) => [m.label + ' · ' + fmt(m.f, 4) + ' Hz'].concat(PLANETS.flatMap((p, j) => { const q = F[j] ? F[j].rows[i] : null; return q ? [q.f === null ? 'yes' : fmt(q.f, 4), q.at + (q.at === 'occupied' ? ' (' + fmt(q.ref, 4) + ')' : '')] : ['–', '–']; })))
          .concat([{cells: ['<b>False alarms</b> (reported, but not a natural frequency of the floor)'].concat(PLANETS.flatMap((p, j) => [F[j] && F[j].falseAlarms.length ? F[j].falseAlarms.map(v => fmt(v, 3)).join(', ') : 'none', '']))},
                   {cells: ['Counted at the empty frequency only: found / false'].concat(PLANETS.flatMap(p => { const r = recOf('S2', D(p)[st.sensor]); return [r ? r.emptyTp + ' / ' + r.emptyFp : '–', '']; }))}]),
        note: 'Match tolerance: ' + tolText('S2') + '. Occupied frequencies (B1 / B2) were predicted before the trials with the linear model: ' + PLANETS.map(p => p + ' ' + modes.map(m => fmt(occ[p][m.label], 4)).join(' / ') + ' Hz').join(', ') + '.'});
    } else makeTable(tb, {title: 'Table: frequencies read from the worn device', desc: 'True natural frequency and what the blind rule reported, per planet. “–” = not found.', file: 'detect_recovery_table',
      columns: ['Natural frequency'].concat(PLANETS.flatMap(p => [p + ' found (Hz)', p + ' error'])),
      rows: modes.map((m, i) => [m.label + ' · ' + fmt(m.f, 4) + ' Hz'].concat(PLANETS.flatMap(p => { const r = (D(p)[st.sensor] || {}).rec; const q = r ? r.pairs.find(a => Math.abs(a[0] - (r.truth ? r.truth[i] : m.f)) < 1e-6) : null; return q ? [fmt(q[1], 4), (q[2] >= 0 ? '+' : '') + fmt(100 * q[2] / q[0], 2) + ' %'] : ['–', '–']; })))
        .concat([{sep: false, cells: ['<b>False alarms</b> (reported, but not a natural frequency of the structure)'].concat(PLANETS.flatMap(p => { const r = (D(p)[st.sensor] || {}).rec; return [r && r.false_alarms.length ? r.false_alarms.map(v => fmt(v, 3)).join(', ') : 'none', '']; }))}]),
      note: 'Reference bridge. True frequencies: natural frequencies of the empty structure' + deckShift() + '. Match tolerance: ' + tolText('S1') + '.'}); });
  const g3 = el('div', {class: 'grid2'}); root.appendChild(g3);
  const f1Of = (r, k) => st.struct === 'S2' ? (r[k].f1_alt !== undefined ? r[k].f1_alt : r[k].f1) : r[k].f1;
  const cn = makeCard(g3, {title: 'Recovery and sensor quality', desc: 'Overall recovery score (F1) with a better or worse accelerometer. Same frozen rule.', height: 240, file: 'detect_recovery_vs_noise',
    note: '30 µg/√Hz: good inertial sensor; 75: typical phone or watch, as in the earlier study (used everywhere else); 300: low-cost device.'});
  const chn = new Chart(cn, {xLabel: 'accelerometer noise (µg/√Hz)', yLabel: 'F1 score', yMin: 0, yMax: 1.05, xLog: true, xMin: 25, xMax: 360, xUnit: 'µg/√Hz'});
  charts.push(() => chn.set({series: planetSeries(p => { const r = (D(p)[st.sensor] || {}).rec_vs_noise || {}; const ks = Object.keys(r).map(Number).sort((a, b) => a - b); return {x: ks, y: ks.map(k => f1Of(r, String(k))), marker: 3.5, width: 2}; })}));
  const cq = makeCard(g3, {title: 'Recovery and amount of data', desc: 'Overall recovery score when the analyst pools 1, 2, … crossings (or standing records).', height: 240, file: 'detect_recovery_vs_records',
    note: 'More data averages out the person’s own movement and the noise. The x-axis counts records' + (standSeconds(st.struct) ? ' (standing records are ' + fmt(standSeconds(st.struct), 2) + ' s each)' : '') + '.'});
  const chq = new Chart(cq, {xLabel: 'records pooled', yLabel: 'F1 score', yMin: 0, yMax: 1.05, xMin: 0.8});
  charts.push(() => chq.set({series: planetSeries(p => { const r = (D(p)[st.sensor] || {}).rec_vs_n || {}; const ks = Object.keys(r).map(Number).sort((a, b) => a - b); return {x: ks, y: ks.map(k => f1Of(r, String(k))), marker: 3.5, width: 2}; }), xMax: Math.max(2, ...PLANETS.map(p => Object.keys((D(p)[st.sensor] || {}).rec_vs_n || {}).length)) + 0.2}));
  charts.push(() => { const sc = O().scenarios.find(s => s.id === st.scen), recs = PLANETS.map(p => [p, recOf(st.struct, D(p)[st.sensor])]).filter(q => q[1]);
    const parts = recs.map(([p, r]) => '<b>' + p + '</b> ' + r.tp + ' of ' + r.n + ' found, ' + r.fp + ' false'), ch = recs.filter(q => q[1].chance !== null).map(([p, r]) => p + ' ' + r.chance);
    keybox.innerHTML = '<b>' + STRUCT[st.struct].name + ' · ' + sc.label + ', ' + SENSOR_NAME[st.sensor].toLowerCase() + '.</b> Natural frequencies recovered blind: ' + parts.join(' · ') + '. ' +
      (ch.length ? 'For comparison, the same rule “finds” ' + ch.join(' / ') + ' of them where there is no structure at all (chance level). ' : '') + (st.struct === 'S2' ? 'A line counts at the empty or the declared occupied frequency. ' : '') + scenNote(st.struct, sc); });
  /* the earlier study's index: computed on its own structure, the reference bridge */
  const LO = DATA.legacy, DEV = {wrist: 'watch', thigh: 'phone'};
  if (LO){
    section(root, 'The earlier study’s Observability Index (reference bridge only)', 'The earlier simulation study rated each natural frequency of its bridge model with an index from 0 to 100. It is computed here on that study’s own structure, the reference bridge, with its own processing; it serves as a link to the earlier study, not as a result for the habitat floor.');
    const gL = el('div', {class: 'grid2'}); root.appendChild(gL);
    const loCats = LO.references.table3.refs.map(r => r[0]), ts = LO.threshold_sensitivity;
    const co = makeCard(gL, {title: 'Reference bridge: Observability Index of the earlier Earth study', desc: 'The earlier study’s index (0–100) for each of its reference modes, computed with the earlier study’s own processing: 20 records, matched solid-ground backgrounds, its reference frequencies and bands.', height: 300, file: 'detect_legacy_oi',
      note: 'Classes of the earlier study: below 27.5 unobservable, 27.5–92.3 marginal, from 92.3 observable. They were calibrated on Earth for the earlier study’s body model; for this body model, and on Mars and the Moon, they are uncalibrated.' + (ts ? ' Moving both thresholds 25 % down or up changes the class of ' + ts.changed.lower25 + ' or ' + ts.changed.higher25 + ' of ' + ts.cells + ' mode results.' : '') + ' The situation and device are those chosen at the top of this page (on the reference bridge).'});
    const cho = new Chart(co, {type: 'bar', cats: loCats, yLabel: 'Observability Index', yMax: 105, refLines: [{y: 27.5, label: 'marginal from 27.5'}, {y: 92.3, label: 'observable from 92.3'}], valueFmt: v => fmt(v, 3)});
    charts.push(() => cho.set({series: planetSeries(p => { const r = (LO.data[st.scen] || {})[p]; const d = r ? r.table3[DEV[st.sensor]] : null; return {values: loCats.map(m => d && d.pmode[m] ? d.pmode[m].oi : null)}; })}));
    const TC = DATA.threshold_check, side = el('div'); gL.appendChild(side);
    if (TC) side.appendChild(el('div', {class: 'card', html: '<div class="ttl">Transfer of the index to this body model (reference bridge)</div><p style="margin:6px 0 0">Criteria declared before the analysis: (1) B1 observable for both devices in both activities: <b>' + (TC.criterion1.holds ? 'met' : 'not met') + '</b>; (2) no mode found blind at 10 dB or more above its background called unobservable, and none not found and below 3 dB called observable: <b>' + (TC.criterion2.holds ? 'met' : 'not met') + '</b>'
      + (TC.criterion2.holds ? '' : ' (' + TC.criterion2.contradictions.length + ' of ' + TC.criterion2.cells + ' results' + (TC.criterion2.contradictions_by_activity ? ', ' + TC.criterion2.contradictions_by_activity.walking + ' of them while walking' : '') + ')') + '. Verdict: ' + TC.verdict + '.</p>'
      + (TC.post_hoc_summary && TC.post_hoc_summary.contradiction_cells_after_removal_db ? '<p class="note">Why (checked after the results): the earlier study removes the gait’s harmonics using the exact simulated step timing. After that removal, the walking results it calls observable carry ' + fmt(TC.post_hoc_summary.contradiction_cells_after_removal_db[0], 3) + '–' + fmt(TC.post_hoc_summary.contradiction_cells_after_removal_db[1], 3) + ' dB more than their solid-ground twins: real content from the structure, hidden from an analysis that does not know the step timing.</p>' : '')}));
    if (LO.earlier_baseline && (LO.data.baseline_walk || LO.data.baseline_stand)){
      const SH = {'observable': 'O', 'marginally observable': 'M', 'unobservable': 'U'};
      const rows = [];
      [['baseline_walk', 'walking', 'Walking 1.5 m/s'], ['baseline_stand', 'standing', 'Standing at mid-span']].forEach(([sid, act, lab]) => {
        const r = (LO.data[sid] || {}).Earth; if (!r) return;
        ['watch', 'phone'].forEach(dev => { const d = r.table3[dev], ref = (LO.earlier_baseline[act] || {})[dev] || {};
          rows.push([lab + ', ' + dev].concat(loCats.map(m => '<b>' + SH[d.pmode[m].band] + '</b> / ' + ((ref.classes || {})[m] || '–'))).concat([fmt(d.oi, 3) + ' / ' + (ref.device !== undefined ? fmt(ref.device, 4) : '–')])); }); });
      makeTable(root, {title: 'Table: the earlier study’s baseline on this body model (reference bridge, Earth)', desc: 'Class here (bold) / class in the earlier study, for each reference mode, and the device-level index. Same structure, shakers, sensor noise, number of records and processing; different body model.', file: 'earlier_baseline_check',
        columns: ['Case'].concat(loCats).concat(['Device index here / earlier study']), rows: rows,
        note: 'O observable, M marginal, U unobservable. The earlier study’s person is a lumped model with prescribed gait; here the body is articulated and its gait is produced by control.'});
    }
  }
  charts.forEach(f => f());
  /* removal of the gait's harmonics on the device's own step timing */
  buildTiming(root, st, charts);
  /* how the sensor is worn */
  buildMounting(root, st, charts);
  /* the two structures side by side */
  if (DATA.structures){
    const K = DATA.structures;
    const th = K.header.map(h => '<th>' + h + '</th>').join('');
    const tr = K.rows.map(r => '<tr>' + r.map((c, i) => '<td' + (i ? ' class="mono"' : '') + '>' + c + '</td>').join('') + '</tr>').join('');
    section(root, 'The habitat floor against the reference bridge');
    const g2s = SG('S2'), g1s = SG('S1');
    root.appendChild(el('div', {class: 'card', html: '<div class="ttl">The same tests on both structures</div>' +
      '<p style="margin:6px 0 0">The main case is a floor of NASA’s Common Habitat (a habitat designed for the Moon and Mars), built like the floor of NASA’s Habitat Demonstration Unit and sized to its documented loads: ' + meters(g2s.L) + ' m wall to wall on a central support, ' + fmt(g2s.m_area, 3) + ' kg per m² against the reference bridge’s ' + fmt(g1s.m_area, 3) + ' kg per m². The reference bridge is the earlier study’s ' + meters(g1s.L) + ' m bridge model. Every test, rule and planet is the same for both, and the rules for the floor were written down before its trials. (A lighter version of the floor, sized for lunar weight only, is a sensitivity check in the paper.)</p>' +
      '<div style="overflow-x:auto;margin-top:8px"><table class="t"><tr>' + th + '</tr>' + tr + '</table></div>' +
      '<p class="note">On the light floor the person becomes part of the structure: the floor’s first mode shifts and is damped by the person, and the damping the person adds falls from Earth to the Moon (page 2). On the reference bridge nothing of this happens.</p>'}));
  }
  root.appendChild(el('div', {class: 'caveat', html: '<b>Read with care.</b> These are properties of simulated signals. The peak-picking rule and the index thresholds were set using Earth cases of the reference bridge only and were not changed for Mars, the Moon or the habitat floor. A different excitation level, device or structure would move these numbers. The habitat floor is one beam line of one floor, with assumed damping and one person on it.'}));
}

/* the gait's harmonics removed by following the gait phase, on the device's own step timing */
function buildTiming(root, st, charts){
  if (!timingOf('S2') && !timingOf('S1')) return;
  section(root, 'Gait removal and step timing', 'The earlier study removes the gait’s own harmonics from the device signal by following the gait phase, which it takes from the simulated walker’s exact footfalls. A watch or a phone has to time its own steps. Here each device times them from its own signal (left and right separately, on every walk), and the same processing is run again. Walking situations only; the structure, situation and device are those chosen at the top of this page.');
  const jst = {planet: 'Earth'}, jcharts = [], DEVJ = {wrist: 'watch', thigh: 'phone'};
  const jbar = el('div', {class: 'controls'}); root.appendChild(jbar);
  segControl(jbar, 'Planet, this section', PLANETS.map(p => ({v: p, html: glyphSVG(p, 15) + p})), jst.planet, v => { jst.planet = v; jcharts.forEach(f => f()); });
  const jkey = el('div', {class: 'key'}); root.appendChild(jkey);
  const g4 = el('div', {class: 'grid2'}); root.appendChild(g4);
  const cs4 = makeCard(g4, {title: 'Device signal after the gait’s harmonics are removed', desc: 'All crossings combined. Solid: on the structure; dotted: the same walks on solid ground. Ink: removal on the exact footfall times; pigment: removal on the device’s own step timing.', height: 300, file: 'timing_spectra',
    note: 'Where a solid curve rises above its dotted twin, the structure shows through. With the device’s own timing the gait lines are removed less completely.'});
  const ch4 = new Chart(cs4, {xLabel: 'frequency (Hz)', yLabel: 'strength, (m/s²)² per Hz', yLog: true, xMin: 0, xMax: 30, xUnit: 'Hz'});
  const cb4 = makeCard(g4, {title: 'Natural frequencies found blind, with and without the removal', desc: 'The frozen peak-picking rule (its harmonic rejection switched off, the removal taking its place) on the structure and, for chance, on solid ground.', height: 300, file: 'timing_recovery',
    note: 'A result means something only where the bar for the structure clearly beats the chance bar. The declared test: more than chance + 1 in a walking case counts as recovered.'});
  const chb4 = new Chart(cb4, {type: 'bar', cats: ['no removal', 'removal, device’s\nown timing', 'removal, exact\nfootfall times'], yLabel: 'natural frequencies found', yMin: 0, valueFmt: v => (Number.isInteger(v) ? String(v) : fmt(v, 2))});
  const tSid = () => { const SJ = timingOf(st.struct); return SJ && SJ.data[st.scen] ? st.scen : 'walk_same_E1'; };
  jcharts.push(() => {
    const SJ = timingOf(st.struct), r = SJ ? (SJ.data[tSid()] || {})[jst.planet] : null, dev = DEVJ[st.sensor], n = modesOf(st.struct).length;
    ch4.o.guides = guidesOf(st.struct, 30, [jst.planet]);
    chb4.o.yLabel = 'natural frequencies found (of ' + n + ')'; chb4.o.yMax = n + 0.6;
    if (!r){ ch4.set({series: []}); chb4.set({series: []}); return; }
    const m2 = r.m2, e = m2.exact[dev], d = m2.device[dev], pc = PAL[jst.planet];
    ch4.set({series: [{x: e.f, y: e.S, color: PAL.ink, label: 'exact timing, on the structure', width: 1.5, dash: []},
                      {x: e.f, y: e.B, color: PAL.ink, label: 'exact timing, solid ground', width: 1.1, dash: [1.5, 2.5]},
                      {x: d.f, y: d.S, color: pc, label: 'device timing, on the structure', width: 1.5, dash: []},
                      {x: d.f, y: d.B, color: pc, label: 'device timing, solid ground', width: 1.1, dash: [1.5, 2.5]}]});
    const g = k => m2[k][dev].no_rejection;
    chb4.set({series: [{label: 'on the structure', color: pc, values: ['none', 'device', 'exact'].map(k => g(k).tp)},
                       {label: 'by chance, solid ground', color: PAL.rigid, values: ['none', 'device', 'exact'].map(k => g(k).chance_tp)}]});
  });
  jcharts.push(() => {
    const SJ = timingOf(st.struct), walking = !!(SJ && SJ.data[st.scen]), ks = (clocksOf(st.struct) || {}).summary;
    const shown = (outcomesOf(st.struct).scenarios.find(q => q.id === tSid()) || {}).label || '';
    let txt = (walking ? '' : '<i>The curves show “' + shown + '”; choose a walking situation at the top of this page to see another.</i><br>') + '<b>' + STRUCT[st.struct].name + '.</b> ' +
      (ks ? 'The devices time their steps to ' + fmt(ks.total_ms_median, 2) + ' ms (median), and the part of that error that does not repeat every stride is ' + fmt(ks.random_ms_median, 2) + ' ms. ' : '');
    const KC = DATA.phaseK && DATA.phaseK.structures[st.struct] ? DATA.phaseK.structures[st.struct].c3_device_clock : null;
    if (st.struct === 'S2'){
      if (KC) txt += '<b>Blind recovery after the removal:</b> above chance + 1 in <b>' + KC.above_chance_plus_one.device + ' of ' + KC.n + '</b> walking cases with the device’s own timing (' + KC.above_chance_plus_one.exact + ' with the exact footfalls, ' + KC.above_chance_plus_one.none + ' without removal). So a device that removes its own gait harmonics this way can find the floor’s modes in some walks, where blind peak picking without the removal finds them in none.';
    } else {
      const SJC = DATA.step_timing_check;
      if (SJC){ const c1 = SJC.criterion1, c3 = SJC.criterion3, m2s = SJC.m2_summary;
        txt += '<b>The earlier study’s index (reference bridge only):</b> with this timing ' + c1.same_class + ' of ' + c1.cells + ' walking results keep their class, and ' + c1.observable_kept + ' of ' + c1.observable_exact + ' “observable” results stay observable; declared test (90 % each) ' + (c1.holds ? 'met' : '<b>not met</b>') + '. ' +
          '<b>Blind recovery after the removal:</b> above chance + 1 in ' + m2s.device_no_rejection.cells_above_chance_plus_one + ' of ' + c3.cells + ' walking cases with the device’s own timing (' + m2s.exact_no_rejection.cells_above_chance_plus_one + ' with the exact footfalls, ' + m2s.none_no_rejection.cells_above_chance_plus_one + ' without removal). So a device that removes its own gait harmonics this way can find the structure’s lowest frequencies in some walks; the earlier study’s walking classes, however, need timing finer than a device has.'; }
    }
    jkey.innerHTML = txt;
  });
  const g5 = el('div', {class: 'grid2'}); root.appendChild(g5);
  const cj = makeCard(g5, {title: 'Required precision of the step timing', desc: ' ', height: 270, file: 'timing_precision', note: ' '});
  const chj = new Chart(cj, {xLabel: 'random footfall timing error added (ms; 0.03 = exact)', yLabel: 'above background (dB)', xLog: true, xMin: 0.025, xMax: 4.5, unit: 'dB', xUnit: 'ms', endLabels: true});
  jcharts.push(() => {
    const SJX = jitterOf(st.struct); cj.card.style.display = SJX ? '' : 'none'; if (!SJX) return;
    const floor = st.struct === 'S2', dv = [];
    Object.values(SJX.device_clock_error_ms).forEach(d => Object.values(d).forEach(v => dv.push(v.random_ms_median)));
    const cases = floor ? ['Earth', 'Moon'].filter(p => SJX.results.some(r => r.case === 'walk_same_E1' && r.planet === p)) : ['Earth'];
    const labs = floor ? modesOf('S2').map(m => m.label) : ['B1', 'B2', 'T1'], dashes = [[], [8, 2.5], [1.6, 2.4]], ser = [];
    cases.forEach(p => { const rr = SJX.results.filter(r => r.case === 'walk_same_E1' && r.planet === p).sort((a, b) => a.sigma_ms - b.sigma_ms), xs = rr.map(r => Math.max(r.sigma_ms, 0.03));
      labs.forEach((m, i) => ser.push({x: xs, y: rr.map(r => r.devices.watch.excess_db[m]), color: floor ? PAL[p] : PAL.ink, dash: dashes[i], label: m + (floor ? ', ' + p : ''), endLabel: m, marker: 3, width: 1.5})); });
    cj.head.querySelector('.desc').innerHTML = 'Exact footfall times with random errors added (' + cases.join(' and ') + ', 1.0 m/s, shakers on, watch). Device signal above its solid-ground background after the removal, in ' + (floor ? 'bands around each of the floor’s modes (empty and occupied frequency)' : 'the earlier study’s reference bands') + '.';
    cj.note.innerHTML = 'Shaded: the random part of the error of the devices’ own clocks in these walks. Checked after the main results; not part of the declared test.';
    chj.set({series: ser, bands: [{x0: Math.min(...dv), x1: Math.max(...dv), color: PAL.band}]});
  });
  const tbj = el('div'); g5.appendChild(tbj);
  jcharts.push(() => {
    tbj.innerHTML = ''; const dev = DEVJ[st.sensor], SJ = timingOf(st.struct), byp = SJ ? SJ.data[tSid()] : null;
    if (!byp) return;
    const pls = PLANETS.filter(p => byp[p]);
    if (st.struct === 'S2'){
      const side = st.sensor;
      makeTable(tbj, {title: 'Habitat floor: found / chance, every planet', desc: 'Natural frequencies found blind (of ' + modesOf('S2').length + ') on the floor / on solid ground, for the walking situation and device chosen at the top.', file: 'timing_floor_cells',
        columns: ['Planet', 'No removal', 'Device’s own timing', 'Exact footfalls', 'Device clock random error, left / right (ms)'],
        rows: pls.map(p => { const m2 = byp[p].m2, g = k => m2[k][dev].no_rejection, tm = byp[p].timing || {};
          return [p].concat(['none', 'device', 'exact'].map(k => { const v = g(k); return (v.tp > v.chance_tp + 1 ? '<b>' : '') + v.tp + ' / ' + v.chance_tp + (v.tp > v.chance_tp + 1 ? '</b>' : ''); }))
            .concat([['L', 'R'].map(s => tm[side + '_' + s] ? fmt(tm[side + '_' + s].random_ms_median, 2) : '–').join(' / ')]); }),
        note: 'Bold: more than chance + 1. A line counts at the empty or the declared occupied frequency. The earlier study’s index is not computed on the floor: its reference modes and thresholds belong to the reference bridge.'});
    } else {
      const modes = Object.keys(byp[pls[0]].exact[dev].pmode);
      makeTable(tbj, {title: 'Index classes: exact footfalls → device’s own timing (reference bridge only)', desc: 'The earlier study’s classes (O observable, M marginal, U unobservable) for the situation and device chosen at the top.', file: 'timing_classes',
        columns: ['Mode'].concat(pls), rows: modes.map(m => [m].concat(pls.map(p => { const a = byp[p].exact[dev].pmode[m].cls, b = byp[p].device[dev].pmode[m].cls; return a === b ? a : '<b>' + a + ' → ' + b + '</b>'; }))),
        note: 'Bold: the class changes when the device times its own steps.'});
    }
  });
  jcharts.forEach(f => f());
  charts.push(() => jcharts.forEach(f => f()));
}

/* how the device is attached to the body */
function buildMounting(root, st, charts){
  if (!mountingOf('S2') && !mountingOf('S1')) return;
  section(root, 'Attachment of the devices', 'In the simulation the watch and the phone are fixed rigidly to the arm and the thigh. A real watch sits on a strap and soft skin; a phone moves a little inside a pocket. Here that attachment is a small spring and damper between body and device. The device then shakes more than the body near the attachment’s own frequency and less above it. The structure is the one chosen at the top of this page.');
  const gm = el('div', {class: 'grid2'}); root.appendChild(gm);
  const cmt = makeCard(gm, {title: 'Body motion reaching the device', desc: 'Device motion ÷ body-segment motion, in decibels (0 dB = rigid). Dashed lines: natural frequencies of the structure.', height: 280, file: 'mounting_transmissibility',
    note: '22.8 and 10.7 Hz: the range measured for a strap-mounted accelerometer at the lower back. 5 Hz: an assumed very loose attachment (no measurement found). Damping 0.3 and 0.1: assumed.'});
  const cols = {22.8: PAL.ink, 10.7: PAL.thigh, 5: PAL.ochre};
  const ff = Array.from({length: 300}, (_, i) => 0.5 + i * 0.1);
  const Hdb = (f, fm, z) => { const w = 2 * Math.PI * fm, s = 2 * Math.PI * f; const nr = w * w, ni = 2 * z * w * s, dr = w * w - s * s, di = 2 * z * w * s; return 10 * Math.log10((nr * nr + ni * ni) / (dr * dr + di * di)); };
  const chm = new Chart(cmt, {xLabel: 'frequency (Hz)', yLabel: 'device ÷ body (dB)', xMin: 0, xMax: 30, xUnit: 'Hz', unit: 'dB', legend: true});
  const right = el('div'); gm.appendChild(right);
  const bar2 = el('div', {class: 'controls'}); right.appendChild(bar2);
  const mt = el('div'); right.appendChild(mt);
  const st2 = {sensor: 'thigh'};
  const drawT = () => {
    const MN = mountingOf(st.struct); mt.innerHTML = ''; if (!MN){ chm.set({series: []}); return; }
    chm.set({guides: guidesOf(st.struct, 30), series: [{x: [0.5, 30], y: [0, 0], color: PAL.rigid || '#898781', label: 'rigid', width: 1.2}].concat(MN.mounts.filter(m => m.f).map(m => ({x: ff, y: ff.map(f => Hdb(f, m.f, m.zeta)), color: cols[m.f], dash: m.zeta < 0.2 ? [5, 3] : null, label: m.f + ' Hz, damping ' + m.zeta, width: 1.6})))});
    const k = st2.sensor, n = (MN.modes || modesOf(st.struct)).length;
    const cell = (mid, sid, p, chance) => { const d = ((MN.data[mid] || {})[sid] || {})[p]; if (!d || !d[k]) return '–'; const r = d[k]; return r.tp + ' found' + (r.fp ? ', ' + r.fp + ' false' : '') + (chance ? ' <span style="color:var(--muted)">(chance ' + r.chance_tp + ')</span>' : ''); };
    makeTable(mt, {title: 'Natural frequencies recovered blind, ' + SENSOR_NAME[k].toLowerCase() + ' (' + STRUCT[st.struct].short + ')', desc: 'Shakers on. Same frozen rule and sensor noise as above; only the attachment changes.', file: 'mounting_recovery_' + k,
      columns: ['Attachment', 'Standing, Earth', 'Standing, Moon', 'Walking 1.0 m/s, Earth', 'Walking 1.0 m/s, Moon', '3-D walker, Earth'], left: [0],
      rows: MN.mounts.map(m => [m.label, cell(m.id, 'stand_E1', 'Earth', false), cell(m.id, 'stand_E1', 'Moon', false), cell(m.id, 'walk_same_E1', 'Earth', true), cell(m.id, 'walk_same_E1', 'Moon', true), cell(m.id, 'walk3d_E1', 'Earth', true)]),
      note: (n === 1 ? 'One natural frequency' : (n === 2 ? 'Two' : n === 7 ? 'Seven' : n) + ' natural frequencies') + ' below 30 Hz' + (st.struct === 'S2' ? ', counted at the empty or the declared occupied frequency' : '') + '. “Chance” = the same rule applied to the same walk on solid ground, where there is no structure.'}); };
  segControl(bar2, 'Device', ['wrist', 'thigh'].map(k => ({v: k, html: '<span class="sw" style="background:var(--' + k + ')"></span>' + SENSOR_FULL[k]})), st2.sensor, v => { st2.sensor = v; drawT(); });
  drawT();
  charts.push(drawT);
}

/* ------------------------------------------------------ 4  MODEL AND CHECKS */
function buildModel(){
  const root = document.getElementById('tab-model'), V = DATA.verification, g2 = SG('S2'), g1 = SG('S1');
  root.appendChild(el('p', {class: 'lead', html: 'The setting the study has in mind, what is inside the simulation, how it was checked, and what it does not show.'}));
  const floorText = g2 ? 'a floor of NASA’s Common Habitat mid deck, built like the floor of NASA’s Habitat Demonstration Unit (the main case: ' + meters(g2.L) + ' m wall to wall with a central support)' : 'a floor of NASA’s Common Habitat';
  const deckText = g1 ? 'the reference bridge, the ' + meters(g1.L) + ' m bridge model of the earlier Earth study' : 'the reference bridge, the bridge model of the earlier Earth study';
  if (DATA.habitat_png) root.appendChild(el('div', {class: 'card', html: '<div class="ttl">The setting: inside a habitat</div>' +
    '<img class="habitat" alt="The study at a glance: people inside a habitat on the Moon or Mars, where local gravity still acts; an occupant whose watch on the wrist and phone in a trouser pocket pick up the vibration of the floor through the body; and the question whether the devices alone can recover the natural frequencies of the floor." src="data:image/png;base64,' + DATA.habitat_png + '">' +
    '<p style="margin:6px 0 0">The study has in mind people inside pressurized habitats on the Moon and Mars. Indoors they wear no spacesuit: they stand at workstations and walk on floors, mezzanines, platforms and walkways, with a phone in a trouser pocket and a watch on the wrist. The drawing is a <b>schematic</b>. Two structures were simulated, each the same on every planet: ' + floorText + ', and ' + deckText + ', used to relate the results to the earlier studies.</p>'}));
  const two = el('div', {class: 'two'}); root.appendChild(two);
  const a = el('div', {class: 'card'}); two.appendChild(a);
  const gs = PLANETS.map(p => fmt(DATA.planets[p].g, 3)).join(' · ');
  a.innerHTML = '<div class="ttl">The model</div>' +
  '<svg class="schem" viewBox="0 0 760 330" role="img" aria-label="Sketch of the model">' +
  '<defs><pattern id="hdeck" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="6" stroke-width="0.8" style="stroke:var(--ink)"/></pattern>' +
  '<pattern id="hgnd" width="7" height="7" patternUnits="userSpaceOnUse"><path d="M0 7 L7 0 M0 0 L7 7" stroke-width="0.6" style="stroke:var(--ink2)"/></pattern>' +
  '<marker id="ah" markerWidth="8" markerHeight="8" refX="4" refY="7" orient="auto"><path d="M0 0 L4 8 L8 0z" style="fill:var(--ink)"/></marker></defs>' +
  '<rect x="0" y="240" width="60" height="10" fill="url(#hgnd)" stroke-width="0.6" style="stroke:var(--ink2)"/><rect x="700" y="240" width="60" height="10" fill="url(#hgnd)" stroke-width="0.6" style="stroke:var(--ink2)"/>' +
  '<path d="M0 240 H60 M700 240 H760" stroke-width="2" style="stroke:var(--ink)"/>' +
  '<rect x="60" y="232" width="640" height="16" fill="url(#hdeck)" stroke-width="1.2" style="stroke:var(--ink)"/><rect x="60" y="232" width="640" height="16" fill-opacity="0.55" style="fill:var(--sheet)"/>' +
  '<path d="M60 248 l-14 26 h28 z M700 248 l-14 26 h28 z" style="fill:var(--ink)"/>' +
  '<path d="M60 240 Q380 262 700 240" fill="none" stroke-dasharray="1.5 3" stroke-width="1.2" style="stroke:var(--muted)"/>' +
  '<g stroke-width="11" stroke-linecap="round" fill="none" style="stroke:var(--ink)"><path d="M330 98 L332 160"/><path d="M332 160 L318 196 L326 228"/><path d="M332 160 L350 196 L344 228"/><path d="M330 108 L312 150"/><path d="M330 108 L352 148"/></g>' +
  '<g stroke-width="6.5" stroke-linecap="round" fill="none" style="stroke:var(--sheet)"><path d="M330 98 L332 160"/><path d="M332 160 L318 196 L326 228"/><path d="M332 160 L350 196 L344 228"/><path d="M330 108 L312 150"/><path d="M330 108 L352 148"/></g>' +
  '<circle cx="329" cy="78" r="15" stroke-width="2" style="fill:var(--sheet);stroke:var(--ink)"/>' +
  '<g fill="none" stroke-width="1.8" style="stroke:var(--earth)"><path d="M318 200 l-5 4 l10 4 l-10 4 l10 4 l-5 4"/><path d="M350 200 l-5 4 l10 4 l-10 4 l10 4 l-5 4"/></g>' +
  '<rect x="347" y="143" width="11" height="11" stroke-width="1.2" style="fill:var(--wrist);stroke:var(--ink)"/><rect x="336" y="176" width="11" height="11" stroke-width="1.2" style="fill:var(--thigh);stroke:var(--ink)"/>' +
  '<g style="fill:var(--ink)"><rect x="208" y="250" width="24" height="18"/><rect x="368" y="250" width="24" height="18"/><rect x="528" y="250" width="24" height="18"/></g>' +
  '<path d="M420 60 v52" stroke-width="1.6" marker-end="url(#ah)" style="stroke:var(--ink)"/>' +
  '<g font-family="B612,DejaVu Sans,sans-serif" font-size="16" style="fill:var(--ink)"><text x="432" y="80">gravity acts on every body part</text><text x="432" y="98" font-size="14" style="fill:var(--ink2)">' + gs + ' m/s²</text>' +
  '<text x="372" y="152" style="fill:var(--wrist)">wrist (watch)</text><text x="354" y="188" style="fill:var(--thigh)">thigh pocket (phone)</text>' +
  '<text x="150" y="208" style="fill:var(--earth)">springy legs and soles;</text><text x="150" y="224" style="fill:var(--earth)">feet can lift off</text>' +
  '<text x="452" y="290">shakers: same forces on every planet</text><text x="70" y="300" style="fill:var(--ink2)">floor or bridge, vibrating in its natural modes</text><text x="70" y="318" font-size="12" style="fill:var(--ink2)">person and structure push on each other with equal and opposite forces, solved together</text></g></svg>' +
  '<ul style="margin:6px 0 0;padding-left:18px"><li><b>Body.</b> 15 rigid segments with published masses and lengths (' + bodyKg() + ' kg, 1.75 m), joints driven by muscle-like torques. Standing uses the full 3-D body. Walking is simulated twice on every planet: with a side-view (planar) walker and with a full 3-D walker with sideways balance (page 3, “3-D walker” situations). In the 3-D drawing the body shape and clothing are for display only; the motion is the simulated one.</li>' +
  '<li><b>Legs and soles.</b> Springs and dampers whose stiffness on Earth was set so that the standing body resonates as measured in published experiments (' + (standingRow('Earth') ? 'about ' + fmt(standingRow('Earth').apparent_mass_f, 2) + ' Hz' : '') + ').</li>' +
  '<li><b>Structures.</b> ' + (g2 ? 'The main case is a habitat floor derived from NASA designs: a floor of NASA’s Common Habitat mid deck (a vessel of ' + fmt(g2.vessel.D, 2) + ' m diameter), built like the floor of NASA’s Habitat Demonstration Unit, ' + meters(g2.L) + ' m wall to wall with a central support, modelled as one beam line with the ' + meters(g2.W) + ' m of floor that moves with it (' + g2.modes.filter(m => m.f < 30).map(m => m.label + ' ' + fmt(m.f, 4) + ' Hz').join(', ') + ' below 30 Hz; a lighter version sized for lunar weight is a sensitivity check). ' : '') +
    (g1 ? 'The reference bridge is the ' + meters(g1.L) + ' m bridge model of the earlier Earth study, reproduced exactly, vibrating in its first ' + g1.n_modes + ' natural shapes. ' : '') + 'Each is the same on every planet.</li>' +
  '<li><b>Contact.</b> Feet press on the moving surface of the structure, can slide and can lift off; they are never glued down.</li>' +
  '<li><b>Sensors.</b> Virtual accelerometers on the wrist (like a watch) and in a trouser pocket (like a phone) report what a real device would: movement plus the pull of local gravity.</li>' +
  '<li><b>Gravity.</b> Changed only in the equations of motion. Mass, structure and sensors stay the same.</li></ul>';
  const b = el('div'); two.appendChild(b);
  b.appendChild(el('div', {class: 'card', html: '<div class="ttl">Limits of this study</div><ul style="margin:6px 0 0;padding-left:18px">' +
    '<li>It is <b>not validated</b> on the Moon or Mars. No reduced-gravity measurement was reproduced.</li><li>The person is <b>unsuited</b>, as inside a pressurized habitat. Suited work outdoors is not covered: a suit adds mass and stiff joints, and the devices would sit on the suit.</li><li>One <b>habitat floor</b> was simulated, derived from NASA designs, as one beam line of the floor (its plate modes are not modelled), with assumed damping and one person on it. A crew of several would change a light floor more.</li>' +
    '<li>The walker is a <b>control hypothesis</b>, tuned to published gait numbers. It is not a model of how people decide to move.</li><li>One body (' + bodyKg() + ' kg), two structures (one habitat floor and the reference bridge), one sensor quality. Other people, floors and structures will differ.</li>' +
    '<li>How human joints and tissue stiffen in low gravity is not known. The spread across “body assumptions” on page 2 shows how much that matters.</li></ul>'}));
  b.appendChild(el('div', {class: 'card', html: '<div class="ttl">Citing numbers from this viewer</div><p style="margin:6px 0 0">Every chart has <b>Save image</b> (PNG) and <b>Save data</b> (CSV) buttons. The numbers come from the files in <span class="mono">outputs/tables</span> of the project folder, and each is produced by a script listed in the project’s README.</p>'}));
  /* natural vibration shapes of each structure */
  const SHS = DATA.shapes, keys = STRUCTS.filter(k => SHS && SHS[k]);
  if (keys.length){
    section(root, 'Natural vibration shapes of the two structures', 'Every chart labels the natural frequencies B (vertical bending), T (twisting) and L (sideways bending). Pick one to see how the structure moves in it (motion greatly exaggerated).');
    const card = el('div', {class: 'card'}), cv = el('canvas'); cv.style.height = '260px';
    const bar = el('div', {class: 'opts'}), modeBox = el('div'); card.appendChild(bar); card.appendChild(cv);
    const note = el('div', {class: 'note'}); card.appendChild(note);
    root.appendChild(card);
    const sv = {struct: keys[0], mode: 0};
    const fillModes = () => { modeBox.innerHTML = ''; sv.mode = 0; const SH = SHS[sv.struct], g = SG(sv.struct);
      segControl(modeBox, null, SH.modes.map((m, i) => ({v: i, html: '<b>' + m.label + '</b> <small>' + fmt(m.f, 3) + ' Hz</small>'})), 0, v => { sv.mode = +v; });
      note.innerHTML = g && g.vessel ? 'Habitat floor: blue line, the beam line, along which the person stands (dot, ' + meters(g.stand[0]) + ' m from the wall) and walks; triangles, the supports at both walls and in the middle. The floor’s twist is restrained (it is one beam line of a continuous floor), so it has only bending modes; its plate modes are outside the model.'
        : 'Reference bridge: blue line, the walking line used in this study (' + fmt(g ? g.line_y : 0.575, 3) + ' m from the centre); dot, the standing point. A sensor on a person standing there feels twisting modes as up-and-down motion plus rocking.'; };
    if (keys.length > 1) structControl(bar, null, sv.struct, v => { sv.struct = v; fillModes(); });
    bar.appendChild(modeBox); fillModes();
    const draw = now => {
      requestAnimationFrame(draw);
      if (document.getElementById('tab-model').hidden) return;
      const r = cv.getBoundingClientRect(), w = r.width, h = r.height, dpr = window.devicePixelRatio || 1;
      if (!w) return;
      if (cv.width !== Math.round(w * dpr)){ cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
      const SH = SHS[sv.struct], g = SG(sv.struct) || {}, c = cv.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, w, h); c.fillStyle = PAL.surface; c.fillRect(0, 0, w, h);
      const m = SH.modes[sv.mode] || SH.modes[0], amp = Math.sin(now / 1000 * 2 * Math.PI * 0.7);
      const s = Math.min((w - 80) / (SH.L + 0.9 * SH.W), (h - 60) / (0.55 * SH.W + 2.2));
      const P = (x, y, z) => [40 + (x + 0.9 * (y + SH.W / 2)) * s, h / 2 + 30 - (0.45 * (y + SH.W / 2) + z) * s];
      const pt = (i, j) => { const k = i * SH.ny + j, x = SH.L * i / (SH.nx - 1), y = -SH.W / 2 + SH.W * j / (SH.ny - 1);
        return P(x, y + 0.9 * amp * m.y[k], 0.9 * amp * m.z[k]); };
      // undeformed outline
      c.strokeStyle = PAL.grid; c.lineWidth = 1; c.setLineDash([4, 4]); c.beginPath();
      for (const [a, b] of [[[0, -SH.W / 2], [SH.L, -SH.W / 2]], [[SH.L, -SH.W / 2], [SH.L, SH.W / 2]], [[SH.L, SH.W / 2], [0, SH.W / 2]], [[0, SH.W / 2], [0, -SH.W / 2]]]){
        const p0 = P(a[0], a[1], 0), p1 = P(b[0], b[1], 0); c.moveTo(p0[0], p0[1]); c.lineTo(p1[0], p1[1]); }
      c.stroke(); c.setLineDash([]);
      // surface as quads, far edge first
      for (let j = SH.ny - 2; j >= 0; j--) for (let i = 0; i < SH.nx - 1; i++){
        const a = pt(i, j), b = pt(i + 1, j), d = pt(i + 1, j + 1), e = pt(i, j + 1);
        c.fillStyle = j ? PAL.surface2 : PAL.grid; c.strokeStyle = PAL.muted; c.lineWidth = 0.6;
        c.beginPath(); c.moveTo(a[0], a[1]); c.lineTo(b[0], b[1]); c.lineTo(d[0], d[1]); c.lineTo(e[0], e[1]); c.closePath(); c.fill(); c.stroke(); }
      // walking line (interpolated between the centre row and an edge row) and the standing point
      const ly = g.line_y !== undefined ? g.line_y : 0.575, t = Math.abs(ly) / (SH.W / 2), jr = ly >= 0 ? 2 : 0;
      const at = i => { const a = pt(i, 1), b = pt(i, jr); return [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]; };
      c.strokeStyle = PAL.Earth; c.lineWidth = 2; c.beginPath();
      for (let i = 0; i < SH.nx; i++){ const q = at(i); i ? c.lineTo(q[0], q[1]) : c.moveTo(q[0], q[1]); }
      c.stroke();
      if (g.stand){ const u = g.stand[0] / SH.L * (SH.nx - 1), i0 = Math.min(SH.nx - 2, Math.floor(u)), f = u - i0, q0 = at(i0), q1 = at(i0 + 1);
        c.fillStyle = PAL.ochre; c.strokeStyle = PAL.ink; c.lineWidth = 1.2; c.beginPath(); c.arc(q0[0] + f * (q1[0] - q0[0]), q0[1] + f * (q1[1] - q0[1]), 5, 0, 7); c.fill(); c.stroke(); }
      for (const xb of (g.supports || [0, SH.L])){ const p0 = P(xb, 0, -0.15); c.fillStyle = PAL.ink; c.beginPath(); c.moveTo(p0[0], p0[1]); c.lineTo(p0[0] - 8, p0[1] + 14); c.lineTo(p0[0] + 8, p0[1] + 14); c.fill(); }
      c.fillStyle = PAL.ink; c.font = '600 14px ' + FONT; c.textAlign = 'left'; c.textBaseline = 'top';
      const kind = {B: 'vertical bending', T: 'twisting', L: 'sideways bending', A: 'lengthwise'}[m.label[0]] || '';
      c.fillText(STRUCT[sv.struct].short.replace(/^./, ch => ch.toUpperCase()) + ' · ' + m.label + ' · ' + fmt(m.f, 4) + ' Hz · ' + kind, 12, 10);
    };
    requestAnimationFrame(draw);
  }
  section(root, 'Verification checks', 'Each check has a pass limit that was written down before the check was run.');
  if (!V){ missing(root, 'The verification summary'); return; }
  makeTable(root, {title: 'Table: verification checks', file: 'verification_checks', columns: ['What was checked', 'Result', 'Limit', 'Verdict'], left: [0, 1, 2],
    rows: V.checks.map(c => [c.what, c.result, c.limit, c.pass ? '<span class="pass">pass</span>' : '<span class="fail">FAIL</span>']),
    note: V.note || ''});
}

/* ------------------------------------------------------------------- start */
/* print colours: whiteprint or blueprint; the charts recolour in place */
function setPrint(name){
  if (!PALS[name]) name = 'white';
  const map = {};
  for (const k in PALS[name]) if (typeof PAL[k] === 'string' && PAL[k] !== PALS[name][k]) map[PAL[k]] = PALS[name][k];
  Object.assign(PAL, PALS[name]);
  document.documentElement.setAttribute('data-print', name);
  for (const k in PATTERNS) delete PATTERNS[k];
  for (const ch of CHARTS) for (const key of ['series', 'guides', 'bands', 'refLines']) (ch.o[key] || []).forEach(q => { if (q && map[q.color]) q.color = map[q.color]; });
  document.querySelectorAll('.printsel button').forEach(b => { b.classList.toggle('on', b.dataset.print === name); b.setAttribute('aria-checked', b.dataset.print === name ? 'true' : 'false'); });
  try { localStorage.setItem('ps-print', name); } catch (e) { /* storage unavailable: the choice lasts this visit */ }
  CHARTS.forEach(ch => ch.draw());
}
(function(){
  let pr = null;
  try { pr = localStorage.getItem('ps-print'); } catch (e) { pr = null; }
  if (!pr) pr = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'blue' : 'white';
  setPrint(pr);
  document.querySelectorAll('.printsel button').forEach(b => b.addEventListener('click', () => setPrint(b.dataset.print)));
})();
document.getElementById('tbTrials').textContent = DATA.meta.n_records;
document.getElementById('tbBody').textContent = bodyKg() + ' kg, 1.75 m';
document.getElementById('footMeta').textContent = 'Built ' + DATA.meta.built + ' · project version ' + (DATA.meta.commit || '?') + ' · ' + DATA.meta.n_records + ' simulated trials';
buildWatch();
