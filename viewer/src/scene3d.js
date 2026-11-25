/* ---------------------------------------------------------------- 3-D scene
   A shaded 3-D rendering of the simulated person on the structure (three.js, inlined): the reference bridge (the
   earlier study's bridge model) over a ravine in a landscape, or the habitat floor inside a cut-away habitat vessel.
   World axes as in the simulation: x along the span, y to the walker's left, z up.
   The person is drawn from the simulated body landmarks; segment shapes and
   clothing are for display only. */
const S3 = {
  skin: 0xc68e6d, hair: 0x2e2119, shirt: 0x2f6f8f, sleeve: 0x2f6f8f, pants: 0x283246, shoe: 0x23262b, sole: 0xe9e6df,
  concrete: 0xb3afa6, concreteDark: 0x7d7a73, rail: 0x6f7378, shaker: 0x2f3237, steel: 0x3a3d42};
const PLANET3 = {
  Earth: {skyTop: '#7fb5ec', skyLow: '#dcecf8', fog: 0xcfe2f1, fogNear: 60, fogFar: 260, hemiSky: 0xdfefff, hemiGround: 0x5d7a4a, hemi: 0.7,
          sun: 0xfff4e0, sunI: 2.6, sunDir: [-0.5, -0.9, 1.0], ground: ['#5a8749', '#527d43', '#62904f', '#4a733d'], groundEdge: '#8a7a5e'},
  Mars:  {skyTop: '#c79670', skyLow: '#efd4b0', fog: 0xe3c4a0, fogNear: 40, fogFar: 220, hemiSky: 0xf4d9bd, hemiGround: 0x8a4b2c, hemi: 0.8,
          sun: 0xfff0dc, sunI: 2.0, sunDir: [-0.4, -0.7, 0.9], ground: ['#8f5234', '#844b2f', '#9a5b3a', '#7b452b'], groundEdge: '#7f4024'},
  Moon:  {skyTop: '#03040a', skyLow: '#0d1018', fog: null, hemiSky: 0x5a6070, hemiGround: 0x2a2a2c, hemi: 0.35,
          sun: 0xffffff, sunI: 2.4, sunDir: [-0.6, -0.5, 0.45], ground: ['#7d7e80', '#727375', '#88898b', '#6a6b6d'], groundEdge: '#77787b'}};

function webglAvailable(){
  try { const c = document.createElement('canvas'); return !!(window.WebGLRenderingContext && (c.getContext('webgl') || c.getContext('experimental-webgl'))) && typeof THREE !== 'undefined'; }
  catch (e){ return false; }
}

class Scene3D {
  constructor(cv, overlay){
    this.cv = cv; this.ov = overlay; this.tr = null; this.planet = null; this.ok = false;
    try {
      this.r = new THREE.WebGLRenderer({canvas: cv, antialias: true});
    } catch (e){ return; }
    this.ok = true;
    const r = this.r;
    if (THREE.ColorManagement) THREE.ColorManagement.legacyMode = false;   // treat colours as sRGB, light in linear space
    r.toneMapping = THREE.ACESFilmicToneMapping; r.toneMappingExposure = 1.0;
    r.outputEncoding = THREE.sRGBEncoding; r.shadowMap.enabled = true; r.shadowMap.type = THREE.PCFSoftShadowMap;
    this.scene = new THREE.Scene();
    this.cam = new THREE.PerspectiveCamera(38, 2, 0.05, 900); this.cam.up.set(0, 0, 1);
    this.orbit = {az: -Math.PI / 2, el: 0.12, dist: 5.2}; this.target = new THREE.Vector3(7, 0.6, 1.0); this.camX = null;
    this.hemi = new THREE.HemisphereLight(0xffffff, 0x444444, 0.8); this.hemi.position.set(0, 0, 1); this.scene.add(this.hemi);
    this.sun = new THREE.DirectionalLight(0xffffff, 2); this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048); const sc = this.sun.shadow.camera; sc.left = -5; sc.right = 5; sc.top = 5; sc.bottom = -5; sc.near = 1; sc.far = 60;
    this.sun.shadow.bias = -0.0004; this.scene.add(this.sun); this.scene.add(this.sun.target);
    this._buildWorld(); this._buildBridge(); this._buildPerson();
    this._interaction();
  }
  /* ------------------------------------------------------------ textures */
  _noiseTex(cols, n, seed, size, extra){
    const c = document.createElement('canvas'); c.width = c.height = size; const g = c.getContext('2d'), rnd = lcg(seed);
    g.fillStyle = cols[0]; g.fillRect(0, 0, size, size);
    for (let i = 0; i < n; i++){ g.fillStyle = cols[1 + (i % (cols.length - 1))]; g.globalAlpha = 0.18 + 0.3 * rnd();
      const x = rnd() * size, y = rnd() * size, s = 2 + rnd() * size / 40; g.beginPath(); g.ellipse(x, y, s, s * (0.5 + rnd()), rnd() * 3, 0, 7); g.fill(); }
    g.globalAlpha = 1; if (extra) extra(g, rnd, size);
    const t = new THREE.CanvasTexture(c); t.wrapS = t.wrapT = THREE.RepeatWrapping; t.encoding = THREE.sRGBEncoding; t.anisotropy = 4; return t;
  }
  _skyTex(planet){
    const P = PLANET3[planet], W = 4096, H = 1024, hz = 760, c = document.createElement('canvas'); c.width = W; c.height = H;
    const g = c.getContext('2d'), rnd = lcg(planet.length * 131 + 7);
    let gr = g.createLinearGradient(0, 0, 0, hz); gr.addColorStop(0, P.skyTop); gr.addColorStop(1, P.skyLow); g.fillStyle = gr; g.fillRect(0, 0, W, H);
    if (planet === 'Moon'){
      for (let i = 0; i < 1400; i++){ g.fillStyle = 'rgba(255,255,255,' + (0.3 + 0.7 * rnd()).toFixed(2) + ')'; const s = rnd() < 0.06 ? 2.2 : 1.1; g.fillRect(rnd() * W, rnd() * hz * 0.98, s, s); }
      const ex = W * 0.30, ey = 210, er = 46; g.save(); g.beginPath(); g.arc(ex, ey, er, 0, 7); g.clip(); g.fillStyle = '#2f73c9'; g.fillRect(ex - er, ey - er, 2 * er, 2 * er);
      g.fillStyle = 'rgba(255,255,255,.85)'; for (let i = 0; i < 12; i++){ g.beginPath(); g.ellipse(ex + (rnd() - .5) * 2 * er, ey + (rnd() - .5) * 2 * er, 8 + rnd() * 18, 3 + rnd() * 6, rnd() * 3, 0, 7); g.fill(); }
      g.fillStyle = 'rgba(70,140,80,.55)'; for (let i = 0; i < 5; i++){ g.beginPath(); g.ellipse(ex + (rnd() - .5) * 1.4 * er, ey + (rnd() - .5) * 1.4 * er, 8 + rnd() * 10, 6 + rnd() * 8, rnd() * 3, 0, 7); g.fill(); }
      g.fillStyle = 'rgba(3,4,8,.93)'; g.beginPath(); g.arc(ex + er * 0.62, ey - er * 0.18, er * 1.02, 0, 7); g.fill(); g.restore();
    } else {
      const sx = planet === 'Earth' ? W * 0.62 : W * 0.55, sy = planet === 'Earth' ? 170 : 230;
      gr = g.createRadialGradient(sx, sy, 3, sx, sy, planet === 'Earth' ? 150 : 110);
      gr.addColorStop(0, 'rgba(255,252,235,1)'); gr.addColorStop(0.2, planet === 'Earth' ? 'rgba(255,246,205,.8)' : 'rgba(230,238,255,.75)'); gr.addColorStop(1, 'rgba(255,246,205,0)');
      g.fillStyle = gr; g.fillRect(0, 0, W, H);
      if (planet === 'Earth'){ g.fillStyle = 'rgba(255,255,255,.85)';
        for (let i = 0; i < 26; i++){ const x = rnd() * W, y = 60 + rnd() * 380, s = 30 + rnd() * 50; for (let k = 0; k < 6; k++){ g.beginPath(); g.ellipse(x + (k - 2.5) * s * 0.55, y + (rnd() - .5) * 10, s * (0.55 + rnd() * .35), s * 0.26, 0, 0, 7); g.fill(); } } }
    }
    // periodic ridges so the panorama has no seam
    const ridge = (base, amp, col, n1, n2, flat) => { g.fillStyle = col; g.beginPath(); g.moveTo(0, H); const p1 = rnd() * 6, p2 = rnd() * 6, p3 = rnd() * 6;
      for (let x = 0; x <= W; x += 4){ const u = x / W * 2 * Math.PI; let y = Math.sin(n1 * u + p1) + 0.5 * Math.sin(n2 * u + p2) + 0.25 * Math.sin((n1 + n2) * u + p3);
        if (flat) y = Math.max(-0.2, Math.min(0.55, y * 1.6)); g.lineTo(x, base - amp * (0.55 + 0.45 * y)); } g.lineTo(W, H); g.fill(); };
    const R = planet === 'Earth' ? ['#a9c9b5', '#86b48e', '#6da574'] : planet === 'Mars' ? ['#c99572', '#b97651', '#a8613c'] : ['#55575b', '#6c6e72', '#828487'];
    ridge(hz + 2, 120, R[0], 7, 13, planet === 'Mars'); ridge(hz + 4, 70, R[1], 11, 19, planet === 'Mars'); ridge(hz + 6, 32, R[2], 17, 29, false);
    g.fillStyle = R[2]; g.fillRect(0, hz + 6, W, H - hz);
    const t = new THREE.CanvasTexture(c); t.encoding = THREE.sRGBEncoding; t.wrapS = THREE.RepeatWrapping; return t;
  }
  /* --------------------------------------------------------------- world */
  _buildWorld(){
    const S = this.scene;
    this.skyMat = new THREE.MeshBasicMaterial({side: THREE.BackSide, fog: false, depthWrite: false});
    const sky = new THREE.Mesh(new THREE.CylinderGeometry(300, 300, 160, 96, 1, true), this.skyMat);
    sky.rotation.x = Math.PI / 2; sky.position.set(7, 0, 56); sky.renderOrder = -1; S.add(sky); this.sky = sky;
    // ground with a ravine under the bridge (heightfield)
    const nx = 160, ny = 70, X0 = -46, X1 = 60, Y0 = -30, Y1 = 30;
    const geo = new THREE.PlaneGeometry(X1 - X0, Y1 - Y0, nx, ny), pos = geo.attributes.position, rnd = lcg(99);
    for (let i = 0; i < pos.count; i++){
      const x = pos.getX(i) + (X0 + X1) / 2, y = pos.getY(i);
      let z = 0;
      const inX = Math.min(Math.max((x - 0.3) / 1.4, 0), 1) * Math.min(Math.max((13.7 - x) / 1.4, 0), 1);
      const inY = 1 / (1 + Math.exp((Math.abs(y) - 14) / 2.5));
      z = -2.6 * inX * inY;
      if (Math.abs(y) > 2.6 || x < -1.2 || x > 15.2) z += (rnd() - 0.5) * 0.06 + 0.12 * Math.sin(x * 0.21) * Math.sin(y * 0.17);
      pos.setXYZ(i, x, y, z);
    }
    geo.computeVertexNormals();
    this.groundMat = new THREE.MeshStandardMaterial({color: 0xffffff, roughness: 0.95, metalness: 0});
    this.ravineGround = new THREE.Mesh(geo, this.groundMat); this.ravineGround.receiveShadow = true; S.add(this.ravineGround);
    const flat = new THREE.Mesh(new THREE.PlaneGeometry(X1 - X0, Y1 - Y0, 1, 1), this.groundMat);
    flat.position.set((X0 + X1) / 2, 0, 0); flat.receiveShadow = true; S.add(flat); this.flatGround = flat;
    const far = new THREE.Mesh(new THREE.RingGeometry(30, 320, 64, 1), new THREE.MeshStandardMaterial({color: 0xffffff, roughness: 1}));
    far.position.set(7, 0, -0.02); S.add(far); this.farGround = far;
  }
  _buildBridge(){
    const B = new THREE.Group(); this.bridge = B; this.scene.add(B);
    const concrete = new THREE.MeshStandardMaterial({color: S3.concrete, roughness: 0.85});
    // deck: 29 stations x 2 edges x (top, bottom)
    const n = 29; this.deckN = n;
    const pos = new Float32Array(n * 4 * 3), idx = [];
    for (let i = 0; i < n - 1; i++){
      const a = i * 4, b = (i + 1) * 4;          // per station: 0 top-left, 1 top-right, 2 bottom-left, 3 bottom-right
      idx.push(a, a + 1, b, b, a + 1, b + 1);             // top, normal +z
      idx.push(a + 2, b + 2, a + 3, b + 2, b + 3, a + 3); // bottom, normal -z
      idx.push(a, b, a + 2, b, b + 2, a + 2);             // left side (y = +1.5), normal +y
      idx.push(a + 1, a + 3, b + 1, b + 1, a + 3, b + 3); // right side (y = -1.5), normal -y
    }
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(pos, 3)); g.setIndex(idx);
    this.deckGeo = g; const deck = new THREE.Mesh(g, concrete); deck.castShadow = true; deck.receiveShadow = true; B.add(deck);
    // abutments and bearings
    const abut = new THREE.MeshStandardMaterial({color: S3.concreteDark, roughness: 0.9});
    for (const x of [-0.8, 14.8]){ const m = new THREE.Mesh(new THREE.BoxGeometry(1.6, 4.2, 3.0), abut); m.position.set(x, 0, -1.55); m.castShadow = m.receiveShadow = true; B.add(m); }
    // railings: posts every metre on both edges, two rails following the deck
    this.posts = []; this.rails = [];
    const railMat = new THREE.MeshStandardMaterial({color: S3.rail, roughness: 0.45, metalness: 0.6, transparent: true, opacity: 0.55});
    for (const yy of [-1.45, 1.45]){
      for (let k = 0; k <= 14; k++){ const p = new THREE.Mesh(new THREE.CylinderGeometry(0.018, 0.018, 1.0, 8), railMat); p.rotation.x = Math.PI / 2; p.castShadow = true; B.add(p); this.posts.push({m: p, x: k, y: yy}); }
      for (const hz of [0.5, 1.0]) for (let k = 0; k < 14; k++){ const r = new THREE.Mesh(new THREE.CylinderGeometry(0.013, 0.013, 1, 6), railMat); r.castShadow = true; B.add(r); this.rails.push({m: r, x0: k, x1: k + 1, y: yy, h: hz}); }
    }
    // shakers under the deck (L/4, L/2, 3L/4 at -90 %, 0, +90 % of the half-width)
    this.shakers = [];
    const shMat = new THREE.MeshStandardMaterial({color: S3.shaker, roughness: 0.6, metalness: 0.3});
    for (const [x, y] of [[3.5, -1.35], [7.0, 0.0], [10.5, 1.35]]){ const m = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.42, 0.34), shMat); m.castShadow = true; B.add(m); this.shakers.push({m, x, y}); }
  }
  /* ------------------------------------------------------ habitat floor
     A cut-away of the habitat vessel (a horizontal cylinder, its axis across the walk), the floor's modelled strip
     (it deflects with the trial's deckW), the rest of the floor (drawn still), the supports at the walls and the
     central support, the shakers, and rigid ground before and after the floor (the walk starts on it). Sizes come
     from DATA.structs (span, strip width, vessel diameter, floor height, supports, shakers); the vessel's length
     and the openings in its wall are drawn only to make the scene readable. */
  _panelTex(base, seam){
    const c = document.createElement('canvas'); c.width = c.height = 128; const g = c.getContext('2d');
    g.fillStyle = base; g.fillRect(0, 0, 128, 128); g.strokeStyle = seam; g.lineWidth = 2; g.strokeRect(1, 1, 126, 126);
    g.globalAlpha = 0.35; g.lineWidth = 1; g.beginPath(); g.moveTo(0, 64); g.lineTo(128, 64); g.stroke();
    const t = new THREE.CanvasTexture(c); t.wrapS = t.wrapT = THREE.RepeatWrapping; t.encoding = THREE.sRGBEncoding; t.anisotropy = 4; return t;
  }
  _shellTex(){
    const c = document.createElement('canvas'); c.width = 64; c.height = 256; const g = c.getContext('2d');
    g.fillStyle = '#d9d2c3'; g.fillRect(0, 0, 64, 256); g.fillStyle = '#a99f8c'; g.fillRect(0, 0, 64, 12); g.fillStyle = 'rgba(80,70,55,.16)'; g.fillRect(0, 128, 64, 2);
    const t = new THREE.CanvasTexture(c); t.wrapS = t.wrapT = THREE.RepeatWrapping; t.encoding = THREE.sRGBEncoding; return t;
  }
  _ensureHabitat(st){
    if (this.hab) return;
    const L = st.L, W = st.W, R = st.vessel.D / 2, zA = st.vessel.z_axis, xc = L / 2, tD = st.deck_t;
    const Ly = 7.2, tS = 0.16, doorHW = 0.75, doorH = 2.1;          // drawn vessel length, wall thickness and openings (schematic)
    const G = new THREE.Group(); this.habitat = G; this.scene.add(G);
    const H = this.hab = {L, W, R, zA, xc, tD, Ly, halves: []};
    const inDoor = (y, z) => Math.abs(y) < doorHW && z > -0.05 && z < doorH;
    const shellMat = new THREE.MeshStandardMaterial({color: 0xffffff, map: this._shellTex(), roughness: 0.92, metalness: 0, side: THREE.BackSide});
    const capMat = new THREE.MeshStandardMaterial({color: 0xcfc7b7, roughness: 0.92, metalness: 0, side: THREE.BackSide});
    const darkMat = new THREE.MeshStandardMaterial({color: 0x5f594f, roughness: 0.8, side: THREE.DoubleSide});
    const restTex = this._panelTex('#e7e3da', '#b9b1a1'); restTex.repeat.set(1 / 1.22, 1 / 0.508);
    const restMat = new THREE.MeshStandardMaterial({color: 0xffffff, map: restTex, roughness: 0.85});
    const add = (grp, geo, mat, x, y, z, shadow) => { const m = new THREE.Mesh(geo, mat); m.position.set(x, y, z); m.receiveShadow = true; m.castShadow = !!shadow; grp.add(m); return m; };
    for (const sgn of [-1, 1]){                                      // the two halves of the vessel, either side of the walking line
      const half = new THREE.Group(); G.add(half); H.halves.push(half);
      const nT = 176, ny = 12, row = nT + 1, pos = [], uv = [], idx = [];
      for (let j = 0; j <= ny; j++) for (let i = 0; i <= nT; i++){
        const th = 2 * Math.PI * i / nT, y = sgn * (Ly / 2) * j / ny;
        pos.push(xc + R * Math.cos(th), y, zA + R * Math.sin(th)); uv.push(i / nT * 3, Math.abs(y) / 1.2);
      }
      for (let j = 0; j < ny; j++) for (let i = 0; i < nT; i++){
        const thc = 2 * Math.PI * (i + 0.5) / nT, yc = sgn * (Ly / 2) * (j + 0.5) / ny;
        if (inDoor(yc, zA + R * Math.sin(thc))) continue;
        const a = j * row + i, b = a + 1, c = a + row, d = c + 1;
        if (sgn > 0) idx.push(a, c, b, b, c, d); else idx.push(a, b, c, b, d, c);      // outward normals
      }
      const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); geo.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
      geo.setIndex(idx); geo.computeVertexNormals();
      add(half, geo, shellMat, 0, 0, 0);
      // end wall with the hatch on the vessel axis
      const cap = add(half, new THREE.CircleGeometry(R, 96), capMat, xc, sgn * Ly / 2, zA); cap.rotation.x = sgn > 0 ? -Math.PI / 2 : Math.PI / 2;
      const hatch = add(half, new THREE.RingGeometry(0.43 * st.vessel.hatch, 0.53 * st.vessel.hatch, 48), darkMat, xc, sgn * (Ly / 2 - 0.02), zA); hatch.rotation.x = Math.PI / 2;
      // the rest of the floor beyond the modelled strip, drawn still
      const wRest = Ly / 2 - W / 2 - 0.03;
      const rest = add(half, new THREE.BoxGeometry(L, wRest, tD), restMat, xc, sgn * (W / 2 + 0.03 + wRest / 2), -tD / 2 - 0.004);
      const ruv = rest.geometry.attributes.uv; for (let k = 0; k < ruv.count; k++) ruv.setXY(k, ruv.getX(k) * L, ruv.getY(k) * wRest);
    }
    // the vessel wall cut through on the walking line (section)
    { const nT = 176, pos = [], idx = [];
      for (let i = 0; i <= nT; i++){ const th = 2 * Math.PI * i / nT; pos.push(xc + R * Math.cos(th), 0, zA + R * Math.sin(th), xc + (R + tS) * Math.cos(th), 0, zA + (R + tS) * Math.sin(th)); }
      for (let i = 0; i < nT; i++){ const thc = 2 * Math.PI * (i + 0.5) / nT; if (inDoor(0, zA + R * Math.sin(thc))) continue; const a = 2 * i; idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2); }
      const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); geo.setIndex(idx); geo.computeVertexNormals();
      add(G, geo, new THREE.MeshStandardMaterial({color: 0x6d675d, roughness: 0.85, side: THREE.DoubleSide}), 0, 0, 0); }
    // the modelled strip of floor: nx stations x 4 vertices (as the deck), deflected each frame
    const nx = 29, sp = new Float32Array(nx * 4 * 3), suv = new Float32Array(nx * 4 * 2), sidx = [];
    for (let i = 0; i < nx - 1; i++){
      const a = i * 4, b = (i + 1) * 4;
      sidx.push(a, a + 1, b, b, a + 1, b + 1); sidx.push(a + 2, b + 2, a + 3, b + 2, b + 3, a + 3);
      sidx.push(a, b, a + 2, b, b + 2, a + 2); sidx.push(a + 1, a + 3, b + 1, b + 1, a + 3, b + 3);
    }
    const sg = new THREE.BufferGeometry(); sg.setAttribute('position', new THREE.BufferAttribute(sp, 3)); sg.setAttribute('uv', new THREE.BufferAttribute(suv, 2)); sg.setIndex(sidx);
    const stripTex = this._panelTex('#c4a872', '#7a6440'); stripTex.repeat.set(1 / 1.22, 1 / 0.508);
    H.strip = add(G, sg, new THREE.MeshStandardMaterial({color: 0xffffff, map: stripTex, roughness: 0.8}), 0, 0, 0); H.stripGeo = sg; H.nx = nx;
    // supports: a ledger at each wall and the central support (beam across the strip and a column down to the vessel)
    const supMat = new THREE.MeshStandardMaterial({color: 0x6f6a62, roughness: 0.6, metalness: 0.3});
    for (const xs of (st.supports || [0, L])){
      if (xs > 0.01 && xs < L - 0.01){
        add(G, new THREE.BoxGeometry(0.14, W, 0.12), supMat, xs, 0, -tD - 0.06);
        const zBot = zA - Math.sqrt(Math.max(0, R * R - (xs - xc) * (xs - xc))), hc = -tD - 0.12 - zBot;
        add(G, new THREE.BoxGeometry(0.14, 0.14, hc), supMat, xs, 0, zBot + hc / 2);
      } else add(G, new THREE.BoxGeometry(0.16, W, 0.1), supMat, Math.min(L - 0.08, Math.max(0.08, xs)), 0, -tD - 0.05);
    }
    // shakers under the strip, at the simulated positions
    H.shakers = (st.shakers || []).map(([x, y]) => ({x, y, m: add(G, new THREE.BoxGeometry(0.3, 0.3, 0.26), new THREE.MeshStandardMaterial({color: S3.shaker, roughness: 0.6, metalness: 0.3}), x, y, -tD - 0.15, true)}));
    // rigid ground before (the walk starts on it) and after the floor, through openings in the vessel wall
    const platTex = this._panelTex('#9e9a92', '#6c6860'); platTex.repeat.set(1 / 1.5, 1 / 1.5);
    const platMat = new THREE.MeshStandardMaterial({color: 0xffffff, map: platTex, roughness: 0.9});
    for (const [x0, x1] of [[-7, 0], [L, L + 5]]){
      const pm = add(G, new THREE.BoxGeometry(x1 - x0, 2 * doorHW, 0.6), platMat, (x0 + x1) / 2, 0, -0.3);
      const puv = pm.geometry.attributes.uv; for (let k = 0; k < puv.count; k++) puv.setXY(k, puv.getX(k) * (x1 - x0), puv.getY(k) * 2 * doorHW);
    }
    G.visible = false;
  }
  _habPlanet(planet){
    if (this.planet === planet) return; this.planet = planet;
    const bg = {Earth: '#d9e3e4', Mars: '#e9d9c8', Moon: '#d8d8db'}[planet] || '#dddddd';
    this.scene.background = new THREE.Color(bg); this.scene.fog = null;
    this.hemi.color.set(0xfffbf2); this.hemi.groundColor.set(0x7a7164); this.hemi.intensity = 0.8;
    this.sun.color.set(0xffffff); this.sun.intensity = 2.0; this.sunElev = 0.95;
    this.arrows.forEach(a => a.setColor(new THREE.Color(PAL[planet])));
  }
  _drawHabitat(tr, L, wd, Kd, S, nx){
    const H = this.hab; if (!H) return;
    const hasDeck = !!tr.deckX, pos = H.stripGeo.attributes.position, uv = H.stripGeo.attributes.uv, n = Math.min(H.nx, hasDeck ? nx : H.nx), w2 = H.W / 2;
    for (let j = 0; j < n; j++){
      const x = hasDeck ? tr.deckX[j] : H.L * j / (n - 1), z = hasDeck ? Kd * L(tr.A.deckW, nx, j) : 0;
      pos.setXYZ(j * 4, x, w2, z); pos.setXYZ(j * 4 + 1, x, -w2, z); pos.setXYZ(j * 4 + 2, x, w2, z - H.tD); pos.setXYZ(j * 4 + 3, x, -w2, z - H.tD);
      uv.setXY(j * 4, x, w2); uv.setXY(j * 4 + 1, x, -w2); uv.setXY(j * 4 + 2, x, w2); uv.setXY(j * 4 + 3, x, -w2);
    }
    pos.needsUpdate = true; uv.needsUpdate = true; H.stripGeo.computeVertexNormals(); H.stripGeo.computeBoundingSphere();
    const vib = tr.exc === 'E1';
    H.shakers.forEach((s, q) => { s.m.visible = vib; const jig = vib ? 0.01 * Math.sin(S.t * 2 * Math.PI * (6 + q) + q) : 0; s.m.position.set(s.x, s.y, Kd * wd(s.x) - H.tD - 0.15 + jig); });
  }
  _mat(c){ return new THREE.MeshStandardMaterial({color: c, roughness: 0.75, metalness: 0.0}); }
  _buildPerson(){
    const P = new THREE.Group(); this.person = P; this.scene.add(P);
    const skin = this._mat(S3.skin), shirt = this._mat(S3.shirt), pants = this._mat(S3.pants), shoe = this._mat(S3.shoe), sole = this._mat(S3.sole), hair = this._mat(S3.hair);
    const cyl = (r1, r2, mat) => { const m = new THREE.Mesh(new THREE.CylinderGeometry(r1, r2, 1, 20, 1), mat); m.castShadow = true; P.add(m); return m; };
    const sph = (mat, seg) => { const m = new THREE.Mesh(new THREE.SphereGeometry(1, seg || 24, (seg || 24) * 2 / 3), mat); m.castShadow = true; P.add(m); return m; };
    const cap = (r, len, mat) => { const m = new THREE.Mesh(new THREE.CapsuleGeometry(r, len, 6, 16), mat); m.castShadow = true; P.add(m); return m; };
    this.parts = {
      pelvis: sph(pants), abdomen: sph(shirt), chest: sph(shirt), neck: cyl(0.045, 0.05, skin), head: sph(skin, 32), hair: null,
      shL: sph(shirt), shR: sph(shirt), uaL: cyl(0.05, 0.042, shirt), uaR: cyl(0.05, 0.042, shirt), elL: sph(skin), elR: sph(skin),
      faL: cyl(0.038, 0.03, skin), faR: cyl(0.038, 0.03, skin), haL: sph(skin), haR: sph(skin),
      hipL: sph(pants), hipR: sph(pants), thL: cyl(0.085, 0.056, pants), thR: cyl(0.085, 0.056, pants), knL: sph(pants), knR: sph(pants),
      shkL: cyl(0.056, 0.042, pants), shkR: cyl(0.056, 0.042, pants),
      shoeL: cap(0.045, 0.18, shoe), shoeR: cap(0.045, 0.18, shoe), soleL: cap(0.05, 0.19, sole), soleR: cap(0.05, 0.19, sole)};
    const hc = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 16, 0, 2 * Math.PI, 0, Math.PI * 0.56), hair); hc.castShadow = true; P.add(hc); this.parts.hair = hc;
    this.parts.nose = sph(skin, 12); this.parts.earL = sph(skin, 12); this.parts.earR = sph(skin, 12);
    const eyeMat = new THREE.MeshStandardMaterial({color: 0x2b211c, roughness: 0.35});
    this.parts.eyeL = sph(eyeMat, 10); this.parts.eyeR = sph(eyeMat, 10);
    this.parts.browL = sph(hair, 8); this.parts.browR = sph(hair, 8);
    const belt = new THREE.Mesh(new THREE.CylinderGeometry(1, 1, 1, 32, 1, true), this._mat(0x3a2c22)); belt.material.side = THREE.DoubleSide; P.add(belt); this.parts.belt = belt;
    // worn devices
    this.watch = new THREE.Mesh(new THREE.BoxGeometry(0.045, 0.045, 0.014), new THREE.MeshStandardMaterial({color: PAL.wrist, roughness: 0.4, emissive: 0x000000}));
    this.phone = new THREE.Mesh(new THREE.BoxGeometry(0.075, 0.15, 0.011), new THREE.MeshStandardMaterial({color: PAL.thigh, roughness: 0.35, emissive: 0x000000}));
    P.add(this.watch); P.add(this.phone);
    // foot-force arrows
    this.arrows = [0, 1].map(() => { const a = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(), 1, 0x2a78d6, 0.12, 0.08); this.scene.add(a); return a; });
  }
  /* ------------------------------------------------------- interaction */
  _interaction(){
    const cv = this.cv; let drag = null;
    cv.addEventListener('pointerdown', e => { drag = {x: e.clientX, y: e.clientY, az: this.orbit.az, el: this.orbit.el}; cv.setPointerCapture(e.pointerId); cv.style.cursor = 'grabbing'; });
    cv.addEventListener('pointermove', e => { if (!drag) return; this.orbit.az = drag.az - (e.clientX - drag.x) * 0.006; this.orbit.el = Math.max(-0.05, Math.min(1.35, drag.el + (e.clientY - drag.y) * 0.005)); this.userMoved = true; });
    cv.addEventListener('pointerup', () => { drag = null; cv.style.cursor = 'grab'; });
    cv.addEventListener('wheel', e => { e.preventDefault(); this.orbit.dist = Math.max(1.5, Math.min(60, this.orbit.dist * Math.exp(e.deltaY * 0.001))); this.userMoved = true; }, {passive: false});
    cv.style.cursor = 'grab';
  }
  setView(v){
    this.view = v; this.userMoved = false;
    if (v === 'side') this.orbit = {az: -Math.PI / 2, el: 0.10, dist: 5.0};
    else if (v === 'three') this.orbit = {az: Math.PI / 2 - 0.75, el: 0.17, dist: 4.4};   // front-left: face, watch and phone in view
    else this.orbit = this.env === 'hab' ? {az: -Math.PI / 2 - 0.35, el: 0.3, dist: 17} : {az: -Math.PI / 2 - 0.35, el: 0.28, dist: 23};
    this.camX = null;
  }
  setPlanet(planet){
    if (this.planet === planet) return; this.planet = planet; const P = PLANET3[planet];
    this.skyMat.map = this._skyTex(planet); this.skyMat.needsUpdate = true;
    this.scene.background = new THREE.Color(P.skyTop);
    this.scene.fog = P.fog ? new THREE.Fog(P.fog, P.fogNear, P.fogFar) : null;
    const tex = this._noiseTex(P.ground, 2600, planet.length * 17 + 3, 1024, planet === 'Moon' ? (g, rnd, s) => {
      for (let i = 0; i < 70; i++){ const x = rnd() * s, y = rnd() * s, r = 4 + rnd() * 22; g.strokeStyle = 'rgba(230,230,232,.35)'; g.lineWidth = 1.5; g.beginPath(); g.arc(x, y, r, 0.2, 3.0); g.stroke(); g.fillStyle = 'rgba(60,60,64,.25)'; g.beginPath(); g.arc(x, y, r * 0.9, 0, 7); g.fill(); } } :
      planet === 'Mars' ? (g, rnd, s) => { for (let i = 0; i < 160; i++){ g.fillStyle = rnd() < 0.5 ? 'rgba(70,32,18,.45)' : 'rgba(255,215,180,.18)'; const x = rnd() * s, y = rnd() * s, r = 1.5 + rnd() * 5; g.beginPath(); g.ellipse(x, y, r, r * 0.7, rnd() * 3, 0, 7); g.fill(); } } :
      (g, rnd, s) => { g.strokeStyle = 'rgba(40,90,40,.35)'; g.lineWidth = 1; for (let i = 0; i < 900; i++){ const x = rnd() * s, y = rnd() * s; g.beginPath(); g.moveTo(x, y); g.lineTo(x + (rnd() - .5) * 3, y - 3 - rnd() * 5); g.stroke(); } });
    tex.repeat.set(16, 9); this.groundMat.map = tex; this.groundMat.needsUpdate = true;
    const far = tex.clone(); far.needsUpdate = true; far.repeat.set(140, 140); this.farGround.material.map = far; this.farGround.material.needsUpdate = true;
    this.hemi.color.set(P.hemiSky); this.hemi.groundColor.set(P.hemiGround); this.hemi.intensity = P.hemi;
    this.sun.color.set(P.sun); this.sun.intensity = P.sunI; const sdn = new THREE.Vector3(...P.sunDir).normalize(); this.sunElev = Math.asin(Math.max(0.15, sdn.z));
    this.arrows.forEach(a => a.setColor(new THREE.Color(PAL[planet])));
  }
  setTrial(tr){
    this.tr = tr;
    if (tr){
      const st = (typeof DATA !== 'undefined' && DATA.structs) ? DATA.structs[tr.structure] : null, env = st && st.vessel ? 'hab' : 'land';
      if (env !== this.env){
        this.env = env; this.planet = null;
        if (env === 'hab') this._ensureHabitat(st);
        this.sky.visible = env === 'land'; this.farGround.visible = env === 'land'; if (this.habitat) this.habitat.visible = env === 'hab';
        if (this.view === 'full' && !this.userMoved) this.setView('full');
      }
      if (env === 'hab') this._habPlanet(tr.planet); else this.setPlanet(tr.planet);
    }
    this.camX = null;
  }
  /* ------------------------------------------------------------- frame */
  _placeLimb(m, A, B){ const d = new THREE.Vector3().subVectors(A, B), L = d.length(); m.position.copy(A).add(B).multiplyScalar(0.5); m.scale.set(1, Math.max(L, 1e-4), 1); m.quaternion.setFromUnitVectors(Y_UP, d.multiplyScalar(1 / Math.max(L, 1e-9))); }
  _placeBasis(m, c, fwd, lat, up, rf, rl, ru){ const M = new THREE.Matrix4().makeBasis(fwd, lat, up); m.quaternion.setFromRotationMatrix(M); m.position.copy(c); m.scale.set(rf, rl, ru); }
  _frame(f, lat0){ // orthonormal (fwd, lat, up) from an up vector and an approximate lateral vector
    const up = f.clone().normalize(); const lat = lat0.clone().sub(up.clone().multiplyScalar(lat0.dot(up))).normalize(); const fwd = new THREE.Vector3().crossVectors(lat, up).normalize(); return [fwd, lat, up]; }
  draw(S){
    const cv = this.cv, rect = cv.getBoundingClientRect(), w = Math.round(rect.width), h = Math.round(rect.height); if (!w || !h || !this.tr) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    if (this._w !== w || this._h !== h){ this._w = w; this._h = h; this.r.setPixelRatio(dpr); this.r.setSize(w, h, false); this.cam.aspect = w / h; this.cam.updateProjectionMatrix(); }
    const tr = this.tr, A = tr.A, n = tr.nAnim, nk = tr.kpNames.length, i0 = Math.max(0, Math.min(n - 2, S.i)), a = S.a, i1 = i0 + 1;
    const L = (arr, m, k) => arr[i0 * m + k] * (1 - a) + arr[i1 * m + k] * a;
    const standing = tr.activity === 'standing', hasDeck = !!tr.deckX, Kd = S.exag;
    const px = A.refx[i0] * (1 - a) + A.refx[i1] * a;
    // bridge (landscape) or habitat floor (vessel)
    const land = this.env !== 'hab';
    this.bridge.visible = hasDeck && land; this.ravineGround.visible = hasDeck && land; this.flatGround.visible = !hasDeck && land;
    const nx = hasDeck ? tr.deckX.length : 29, Ls = hasDeck ? tr.deckX[nx - 1] : 14;
    const wd = x => { if (!hasDeck || x <= 0 || x >= Ls) return 0; const u = x / Ls * (nx - 1), j = Math.min(nx - 2, Math.floor(u)), f = u - j; return L(A.deckW, nx, j) * (1 - f) + L(A.deckW, nx, j + 1) * f; };
    if (!land) this._drawHabitat(tr, L, wd, Kd, S, nx);
    if (hasDeck && land){
      const pos = this.deckGeo.attributes.position, th = 0.27;
      for (let j = 0; j < 29; j++){ const x = tr.deckX[j], z = Kd * L(A.deckW, 29, j);
        pos.setXYZ(j * 4, x, 1.5, z); pos.setXYZ(j * 4 + 1, x, -1.5, z); pos.setXYZ(j * 4 + 2, x, 1.5, z - th); pos.setXYZ(j * 4 + 3, x, -1.5, z - th); }
      pos.needsUpdate = true; this.deckGeo.computeVertexNormals(); this.deckGeo.computeBoundingSphere();
      for (const p of this.posts){ const z = Kd * wd(p.x); p.m.position.set(p.x, p.y, z + 0.5); }
      for (const r of this.rails){ const A1 = new THREE.Vector3(r.x0, r.y, Kd * wd(r.x0) + r.h), B1 = new THREE.Vector3(r.x1, r.y, Kd * wd(r.x1) + r.h); this._placeLimb(r.m, A1, B1); }
      const vib = tr.exc === 'E1';
      this.shakers.forEach((s, q) => { s.m.visible = vib; const jig = vib ? 0.012 * Math.sin(S.t * 2 * Math.PI * (6 + q) + q) : 0; s.m.position.set(s.x, s.y, Kd * wd(s.x) - 0.27 - 0.2 + jig); });
    }
    // body landmarks (with the same display exaggeration rules as the flat view)
    let fx0 = 0, fy0 = 0, fz0 = 0;
    if (standing){ for (const k of [K.heL, K.toL, K.heR, K.toR]){ fx0 += L(A.dx, nk, k) / 4; fy0 += L(A.dy, nk, k) / 4; fz0 += L(A.dz, nk, k) / 4; } }
    const Kb = standing ? Math.min(Kd, tr.kbMax) : 1, ride = standing ? Kd * wd(px) : (Kd - 1) * wd(px);
    const P = new Array(nk);
    for (let k = 0; k < nk; k++){
      if (standing) P[k] = new THREE.Vector3(px + tr.x0[k] + Kb * (L(A.dx, nk, k) - fx0), tr.y0[k] + Kb * (L(A.dy, nk, k) - fy0), tr.z0[k] + ride + Kb * (L(A.dz, nk, k) - fz0));
      else P[k] = new THREE.Vector3(px + tr.x0[k] + L(A.dx, nk, k), tr.y0[k] + L(A.dy, nk, k), tr.z0[k] + L(A.dz, nk, k) + ride);
    }
    this._pose(P, S);
    // force arrows (ground reaction, in units of body weight)
    [[0, K.heL, K.toL], [1, K.heR, K.toR]].forEach(([j, he, to]) => { const f = L(A.fz, 2, j), ar = this.arrows[j];
      if (f < 0.03){ ar.visible = false; return; } ar.visible = true; const o = P[he].clone().add(P[to]).multiplyScalar(0.5); const len = 0.55 * f;
      ar.position.set(o.x, o.y, Math.min(P[he].z, P[to].z) - len - 0.01); ar.setLength(len, Math.min(0.12, 0.4 * len), 0.07); });
    // camera
    const follow = this.view !== 'full';
    const tx = follow ? px + (standing ? 0 : 0.3) : (land ? 7.0 : Ls / 2);
    this.camX = (this.camX === null || !follow) ? tx : this.camX + (tx - this.camX) * 0.2;
    const ty = follow ? (tr.y0[K.pelvis] || 0) : (land ? 0.3 : 0), tz = follow ? 0.95 : (land ? 0.4 : 0.9);
    this.target.set(this.camX, ty, tz);
    const o = this.orbit, ce = Math.cos(o.el);
    this.cam.position.set(this.target.x + o.dist * ce * Math.cos(o.az), this.target.y + o.dist * ce * Math.sin(o.az), this.target.z + o.dist * Math.sin(o.el));
    this.cam.lookAt(this.target);
    // light follows the person so that shadows stay sharp
    // the sun keeps each planet's elevation but turns with the camera (35 degrees to its left),
    // so that the side of the person facing the viewer is lit and shadows fall away from the viewer
    const se = this.sunElev || 0.7, saz = this.orbit.az + 0.6, sc = Math.cos(se);
    this.sun.target.position.set(this.camX, ty, 0);
    this.sun.position.set(this.camX + 20 * sc * Math.cos(saz), ty + 20 * sc * Math.sin(saz), 20 * Math.sin(se));
    this.sky.position.x = this.camX; this.farGround.position.x = this.camX;
    if (!land && this.hab){        // cut-away: the half of the vessel (and of the rest of the floor) nearer the camera is not drawn
      const nearPos = this.cam.position.y >= 0;
      this.hab.halves[1].visible = !nearPos; this.hab.halves[0].visible = nearPos;
    }
    this.r.render(this.scene, this.cam);
    // overlays: selected sensor label
    const dev = S.sensor === 'wrist' ? this.watch : this.phone;
    const v = dev.position.clone().project(this.cam);
    const lab = this.ov.label; if (v.z < 1){ lab.hidden = false; lab.textContent = S.sensor === 'wrist' ? 'watch (wrist)' : 'phone (thigh pocket)'; lab.style.borderColor = PAL[S.sensor];
      const sx = (v.x + 1) / 2 * w, lw = lab.offsetWidth || 0; lab.style.left = (sx + 14 + lw > w - 4 ? Math.max(4, sx - 14 - lw) : sx + 14) + 'px'; lab.style.top = ((1 - v.y) / 2 * h - 10) + 'px'; }
    else lab.hidden = true;
    const pulse = 0.35 + 0.35 * Math.sin(S.wall * 5);
    this.watch.material.emissive.set(S.sensor === 'wrist' ? PAL.wrist : 0x000000); this.watch.material.emissiveIntensity = S.sensor === 'wrist' ? pulse : 0;
    this.phone.material.emissive.set(S.sensor === 'thigh' ? PAL.thigh : 0x000000); this.phone.material.emissiveIntensity = S.sensor === 'thigh' ? pulse : 0;
  }
  _pose(P, S){
    const p = this.parts, V = () => new THREE.Vector3();
    // trunk frame
    const latT = V().subVectors(P[K.shL], P[K.shR]), upT = V().subVectors(P[K.neck], P[K.lumbar]);
    const [fT, lT, uT] = this._frame(upT, latT);
    const latP = V().subVectors(P[K.hipL], P[K.hipR]), upP = V().subVectors(P[K.lumbar], P[K.pelvis]);
    const [fP, lP, uP] = this._frame(upP, latP);
    const at = (base, u, du, f, df) => { const v = base.clone().add(u.clone().multiplyScalar(du)); if (f) v.add(f.clone().multiplyScalar(df || 0)); return v; };
    this._placeBasis(p.pelvis, at(P[K.pelvis], uP, 0.04, fP, -0.005), fP, lP, uP, 0.115, 0.175, 0.13);
    this._placeBasis(p.abdomen, at(P[K.lumbar], uT, 0.10, fT, -0.004), fT, lT, uT, 0.097, 0.148, 0.15);
    this._placeBasis(p.chest, at(P[K.lumbar], uT, 0.30, fT, 0.0), fT, lT, uT, 0.125, 0.195, 0.19);
    this._placeLimb(p.neck, at(P[K.neck], uT, 0.08), at(P[K.neck], uT, -0.03));
    // head
    const upH = V().subVectors(P[K.head], P[K.neck]); const [fH, lH, uH] = this._frame(upH, lT);
    const hc = at(P[K.neck], uH, 0.135, fH, 0.012);
    this._placeBasis(p.head, hc, fH, lH, uH, 0.098, 0.082, 0.115);
    this._placeBasis(p.hair, at(hc, uH, 0.012, fH, -0.012), fH, lH, uH, 0.104, 0.088, 0.118);
    this._placeBasis(p.nose, at(hc, uH, -0.005, fH, 0.095), fH, lH, uH, 0.022, 0.014, 0.024);
    this._placeBasis(p.earL, at(hc, uH, 0.0, lH, 0.082), fH, lH, uH, 0.018, 0.008, 0.028);
    for (const [e, b, sgn] of [[p.eyeL, p.browL, 1], [p.eyeR, p.browR, -1]]){
      this._placeBasis(e, at(hc, uH, 0.018, fH, 0.086).add(lH.clone().multiplyScalar(0.031 * sgn)), fH, lH, uH, 0.007, 0.0085, 0.0085);
      this._placeBasis(b, at(hc, uH, 0.040, fH, 0.084).add(lH.clone().multiplyScalar(0.032 * sgn)), fH, lH, uH, 0.006, 0.017, 0.0045);
    }
    this._placeBasis(p.earR, at(hc, uH, 0.0, lH, -0.082), fH, lH, uH, 0.018, 0.008, 0.028);
    // belt line at the top of the trousers
    const bc = at(P[K.lumbar], uP, -0.005);
    const Mb = new THREE.Matrix4().makeBasis(fP, uP, new THREE.Vector3().crossVectors(fP, uP));
    p.belt.quaternion.setFromRotationMatrix(Mb); p.belt.position.copy(bc); p.belt.scale.set(0.118, 0.035, 0.162);
    p.hair.quaternion.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI / 2));
    p.hair.scale.set(0.104, 0.118, 0.088);         // geometry x -> forward, y (cap axis) -> up, z -> sideways
    // arms
    for (const [s, sh, el, wr, ha] of [['L', K.shL, K.elL, K.wrL, K.haL], ['R', K.shR, K.elR, K.wrR, K.haR]]){
      p['sh' + s].position.copy(P[sh]).add(lT.clone().multiplyScalar(s === 'L' ? -0.012 : 0.012)); p['sh' + s].scale.setScalar(0.05);
      this._placeLimb(p['ua' + s], P[sh], P[el]); p['el' + s].position.copy(P[el]); p['el' + s].scale.setScalar(0.04);
      this._placeLimb(p['fa' + s], P[el], P[wr]);
      const hd = V().subVectors(P[ha], P[wr]), hl = hd.length(); const hcn = P[wr].clone().add(hd.clone().multiplyScalar(0.5));
      p['ha' + s].position.copy(hcn); p['ha' + s].quaternion.setFromUnitVectors(Y_UP, hd.normalize()); p['ha' + s].scale.set(0.028, Math.max(0.05, 0.6 * hl), 0.045);
    }
    // legs and shoes
    for (const [s, hp, kn, an, he, to] of [['L', K.hipL, K.knL, K.anL, K.heL, K.toL], ['R', K.hipR, K.knR, K.anR, K.heR, K.toR]]){
      p['hip' + s].position.copy(P[hp]); p['hip' + s].scale.setScalar(0.088);
      this._placeLimb(p['th' + s], P[hp], P[kn]); p['kn' + s].position.copy(P[kn]); p['kn' + s].scale.setScalar(0.058);
      this._placeLimb(p['shk' + s], P[kn], P[an]);
      const fdir = V().subVectors(P[to], P[he]), flen = fdir.length(); fdir.normalize();
      const [fu, fl, ff] = this._frame(fdir, lP);          // here 'up' slot carries the foot axis
      const c = P[he].clone().add(fdir.clone().multiplyScalar(0.5 * flen + 0.01)); c.add(new THREE.Vector3(0, 0, 0.035));
      const M = new THREE.Matrix4().makeBasis(fl, fdir, new THREE.Vector3().crossVectors(fl, fdir).normalize());
      const len = (flen + 0.13) / 0.27;
      p['shoe' + s].quaternion.setFromRotationMatrix(M); p['shoe' + s].position.copy(c); p['shoe' + s].scale.set(1.0, len, 0.78);
      p['sole' + s].quaternion.setFromRotationMatrix(M); p['sole' + s].position.copy(c).add(new THREE.Vector3(0, 0, -0.028)); p['sole' + s].scale.set(1.0, len * 1.0, 0.28);
    }
    // worn devices: watch on the left wrist, phone in the left trouser pocket
    const wD = V().subVectors(P[K.wrL], P[K.elL]).normalize();
    this.watch.position.copy(P[K.sW]); this.watch.lookAt(P[K.sW].clone().add(lT));   // face outward
    const tD = V().subVectors(P[K.hipL], P[K.knL]).normalize();
    this.phone.position.copy(P[K.sT]).add(fP.clone().multiplyScalar(0.012));
    this.phone.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(lP.clone().negate(), tD, fP));
  }
}
const Y_UP = new THREE.Vector3(0, 1, 0);
