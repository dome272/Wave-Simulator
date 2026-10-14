import { N, DX, EXT, PRESETS, generate, roughen, brush, raycast, sampleZ, encodeBeach, decodeBeach } from './terrain.js';
import { createRenderer } from './render.js';
import { solve, swellSpectrum } from './waves.js';

const $ = (id) => document.getElementById(id);
const canvas = $('view');
const R = createRenderer(canvas);

// --- state -------------------------------------------------------------------
const beach = { z: null, rock: null, waterLevel: 0 };
const swell = { height: 1.8, period: 13, direction: 15, distanceKm: 2000 };
let tool = 'view';
let paused = false;
let speed = 1;
let last = null; // last solver result
const undoStack = [];

const store = {
  get(k) { try { return JSON.parse(localStorage.getItem(k)); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); return true; } catch { return false; } },
};

// --- solver plumbing -----------------------------------------------------------
let worker = null;
try {
  worker = new Worker(new URL('./solver-worker.js', import.meta.url), { type: 'module' });
  worker.onerror = () => { worker = null; };
} catch { worker = null; }
let busy = false, pending = false, reqId = 0;
const statusEl = $('status');

function requestSolve() {
  if (busy) { pending = true; return; }
  busy = true; pending = false;
  statusEl.hidden = false;
  const msg = { id: ++reqId, N, dx: DX, z: beach.z.slice(), wl: beach.waterLevel, swell: { ...swell } };
  if (worker) {
    worker.onmessage = (e) => onSolved(e.data.res);
    worker.postMessage(msg, [msg.z.buffer]);
  } else {
    setTimeout(() => onSolved(solve(msg)), 0);
  }
}
function onSolved(res) {
  busy = false;
  last = res;
  R.setWaves(res);
  updateVerdict(res.summary);
  if (pending) requestSolve(); else statusEl.hidden = true;
}

// --- swell controls ------------------------------------------------------------
const distKm = (s) => Math.round(50 * Math.pow(6000 / 50, s));
const fmtKm = (km) => (km >= 1000 ? `${(km / 1000).toFixed(1)}k km` : `${km} km`);
function dirText(d) {
  if (Math.abs(d) < 2) return 'straight in';
  return `${Math.abs(d)}° from ${d > 0 ? 'left' : 'right'}`;
}
function readSwell() {
  swell.height = +$('height').value;
  swell.period = +$('period').value;
  swell.direction = +$('direction').value;
  swell.distanceKm = distKm(+$('distance').value);
  $('o-height').value = `${swell.height.toFixed(1)} m`;
  $('o-period').value = `${swell.period.toFixed(1)} s`;
  $('o-direction').value = dirText(swell.direction);
  $('o-distance').value = fmtKm(swell.distanceKm);
  const L0 = (9.81 * swell.period ** 2) / (2 * Math.PI);
  $('s-len').textContent = `${Math.round(L0)} m`;
  $('s-speed').textContent = `${Math.round((L0 / swell.period) * 3.6)} km/h`;
  $('s-set').textContent = `every ${swellSpectrum(swell).wavesPerSet} waves`;
  R.setSwellArrow(swell.direction);
}
let swellTimer = 0;
for (const id of ['height', 'period', 'direction', 'distance']) {
  $(id).addEventListener('input', () => {
    readSwell();
    clearTimeout(swellTimer);
    swellTimer = setTimeout(requestSolve, 120);
    saveSettings();
  });
}

function updateVerdict(s) {
  const el = $('verdict');
  if (!s.breakDepth) {
    el.innerHTML = 'These waves reach the sand without breaking. Try a bigger swell or a shallower bar.';
    return;
  }
  const desc = {
    Spilling: 'crests crumble gently down the face, a long rolling whitewater ride',
    Plunging: 'the lip throws forward and pitches over a hollow face',
    Surging: 'waves surge up the steep bottom with little whitewater',
  }[s.type];
  el.innerHTML = `<strong>${s.type}</strong> · ${desc}. The biggest set waves start breaking about <b>${Math.round(s.breakDist)} m</b> out, in <b>${s.breakDepth.toFixed(1)} m</b> of water.`;
}

// --- beach -----------------------------------------------------------------------
function setBeach(b, keepWl) {
  beach.z = b.z; beach.rock = b.rock;
  if (!keepWl && b.waterLevel !== undefined) { beach.waterLevel = b.waterLevel; $('tide').value = b.waterLevel; }
  readTide();
  R.setHeight(beach);
  R.resetFoam();
  requestSolve();
  autosave();
}
function readTide() {
  beach.waterLevel = +$('tide').value;
  const v = beach.waterLevel;
  $('o-tide').value = `${v > 0 ? '+' : ''}${v.toFixed(2)} m`;
  R.setWaterLevel(v);
}
$('tide').addEventListener('input', () => {
  readTide();
  clearTimeout(swellTimer);
  swellTimer = setTimeout(requestSolve, 120);
  autosave();
});

function pushUndo() {
  undoStack.push({ z: beach.z.slice(), rock: beach.rock.slice() });
  if (undoStack.length > 25) undoStack.shift();
  $('undo').disabled = false;
}
function undo() {
  const s = undoStack.pop();
  if (!s) return;
  beach.z = s.z; beach.rock = s.rock;
  R.setHeight(beach);
  requestSolve();
  autosave();
  $('undo').disabled = undoStack.length === 0;
}
$('undo').addEventListener('click', undo);
$('roughen').addEventListener('click', () => {
  pushUndo();
  roughen(beach, 0.6, (Math.random() * 1e5) | 0);
  R.setHeight(beach);
  requestSolve();
  autosave();
});

let saveTimer = 0;
function autosave() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => store.set('shorebreak.current', encodeBeach(beach)), 800);
}
function saveSettings() {
  store.set('shorebreak.swell', { ...swell, distance: +$('distance').value });
}

// Saved beaches (this browser only).
function renderSaved() {
  const list = store.get('shorebreak.saved') || [];
  const el = $('savedList');
  el.replaceChildren();
  list.forEach((s, k) => {
    const row = document.createElement('div');
    row.className = 'item';
    const load = document.createElement('button');
    load.textContent = s.name;
    load.title = 'Load this beach';
    load.addEventListener('click', () => { pushUndo(); setBeach(decodeBeach(s.data)); });
    const del = document.createElement('button');
    del.textContent = 'Delete';
    del.addEventListener('click', () => {
      const l = store.get('shorebreak.saved') || [];
      l.splice(k, 1);
      store.set('shorebreak.saved', l);
      renderSaved();
    });
    row.append(load, del);
    el.append(row);
  });
}
$('save').addEventListener('click', () => {
  const list = store.get('shorebreak.saved') || [];
  const name = `Beach ${list.length + 1} · ${new Date().toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}`;
  list.push({ name, data: encodeBeach(beach) });
  if (!store.set('shorebreak.saved', list)) {
    $('save').textContent = 'Browser storage is full or blocked';
    setTimeout(() => ($('save').textContent = 'Save this beach'), 2500);
    return;
  }
  renderSaved();
  $('save').textContent = 'Saved';
  setTimeout(() => ($('save').textContent = 'Save this beach'), 1500);
});

// --- new beach dialog ---------------------------------------------------------------
const dlg = $('dlg');
const DESCR = {
  crescent: 'Crescent-shaped bars with rip channels between peaks',
  longbar: 'One bar running the length of the beach',
  point: 'Waves wrap around a rocky headland and peel down the coast',
  reef: 'A V-shaped reef that makes an A-frame peak',
  cove: 'Sand tucked between rocky headlands',
  blank: 'An even slope to sculpt from scratch',
};
let chosenPreset = 'crescent';
const presetsEl = $('presets');
for (const [key, p] of Object.entries(PRESETS)) {
  const b = document.createElement('button');
  b.type = 'button';
  b.className = 'preset';
  b.dataset.key = key;
  b.innerHTML = `<b>${p.name}</b><span>${DESCR[key]}</span>`;
  b.addEventListener('click', () => fillForm(key));
  presetsEl.append(b);
}
const bFields = ['slope', 'barDist', 'rocks', 'roughness'];
function fillForm(key) {
  chosenPreset = key;
  const p = PRESETS[key];
  $('b-shape').value = p.shape;
  $('b-bar').value = p.bar;
  for (const f of bFields) $(`b-${f}`).value = p[f];
  presetsEl.querySelectorAll('.preset').forEach((b) => b.setAttribute('aria-pressed', b.dataset.key === key));
  readForm();
}
function readForm() {
  const o = {
    shape: $('b-shape').value, bar: $('b-bar').value,
    slope: +$('b-slope').value, barDist: +$('b-barDist').value, rocks: +$('b-rocks').value,
    roughness: +$('b-roughness').value, seed: +$('b-seed').value | 0,
  };
  $('o-b-slope').value = o.slope < 0.3 ? 'gentle' : o.slope < 0.7 ? 'medium' : 'steep';
  $('o-b-barDist').value = `${o.barDist} m`;
  $('o-b-rocks').value = `${Math.round(o.rocks * 100)}%`;
  $('o-b-roughness').value = `${Math.round(o.roughness * 100)}%`;
  $('b-barDist').disabled = o.bar === 'none';
  return o;
}
for (const id of ['b-shape', 'b-bar', ...bFields.map((f) => `b-${f}`), 'b-seed']) $(id).addEventListener('input', readForm);
$('b-dice').addEventListener('click', () => { $('b-seed').value = (Math.random() * 99999) | 0; });
$('newBeach').addEventListener('click', () => {
  $('b-seed').value = (Math.random() * 99999) | 0;
  fillForm(chosenPreset);
  if (dlg.showModal) dlg.showModal(); else dlg.setAttribute('open', '');
});
dlg.addEventListener('close', () => {
  if (dlg.returnValue !== 'build') return;
  pushUndo();
  setBeach({ ...generate(readForm()) }, true);
});

// --- display controls ---------------------------------------------------------------
$('speed').addEventListener('input', () => {
  speed = +$('speed').value;
  $('o-speed').value = `${speed}×`;
});
$('contours').addEventListener('change', () => R.setContours($('contours').checked));
$('xray').addEventListener('change', () => R.setSeeThrough($('xray').checked));
$('pause').addEventListener('click', () => {
  paused = !paused;
  $('pause').setAttribute('aria-pressed', paused);
  $('pause').textContent = paused ? 'Play' : 'Pause';
});
// What the camera presets need to know about the current beach.
function viewInfo() {
  const i = N >> 1;
  let jS = N - 1;
  for (let j = 0; j < N; j++) if (beach.z[j * N + i] > beach.waterLevel) { jS = j; break; }
  return {
    shoreZ: jS * DX - EXT / 2,
    breakDist: last ? last.summary.breakDist : 120,
    breakHeight: last ? 0.82 * last.summary.breakDepth : 3,
    breakDistTypical: last ? last.summary.breakDistTypical || last.summary.breakDist : 100,
    spot: last ? last.summary.spot : null,
    ground: (x, z) => sampleZ(beach.z, x + EXT / 2, z + EXT / 2),
  };
}
document.querySelectorAll('[data-view]').forEach((b) => b.addEventListener('click', () => R.setView(b.dataset.view, viewInfo())));
$('menuBtn').addEventListener('click', () => {
  const open = $('rail').classList.toggle('open');
  $('menuBtn').setAttribute('aria-expanded', open);
});

// --- tools & sculpting -------------------------------------------------------------------
function setTool(t) {
  tool = t;
  document.querySelectorAll('[data-tool]').forEach((b) => b.setAttribute('aria-pressed', b.dataset.tool === t));
  const c = R.controls;
  if (t === 'view') {
    c.mouseButtons = { LEFT: 0, MIDDLE: 1, RIGHT: 2 };
    c.touches = { ONE: 0, TWO: 2 };
  } else {
    c.mouseButtons = { LEFT: null, MIDDLE: 1, RIGHT: 0 };
    c.touches = { ONE: null, TWO: 3 };
  }
  canvas.classList.toggle('sculpt', t !== 'view');
  if (t === 'view') R.setBrush(0, 0, 0, false);
  if (t !== 'view' && !$('contours').checked) { $('contours').checked = true; R.setContours(true); }
}
document.querySelectorAll('[data-tool]').forEach((b) => b.addEventListener('click', () => setTool(b.dataset.tool)));
const radiusIn = $('radius'), strengthIn = $('strength');
const readBrush = () => {
  $('o-radius').value = `${radiusIn.value} m`;
  $('o-strength').value = `${(+strengthIn.value).toFixed(2)}`;
};
radiusIn.addEventListener('input', readBrush);
strengthIn.addEventListener('input', readBrush);

let painting = false, paintPt = null, flatTarget = 0, lastSolveReq = 0, pointer = null;
function hitAt(x, y) {
  const { o, d } = R.gridRay(x, y);
  return raycast(beach.z, o, d);
}
canvas.addEventListener('pointerdown', (e) => {
  if (tool === 'view' || e.button !== 0) return;
  const hit = hitAt(e.clientX, e.clientY);
  if (!hit) return;
  e.preventDefault();
  canvas.setPointerCapture(e.pointerId);
  pushUndo();
  painting = true;
  paintPt = hit;
  flatTarget = sampleZ(beach.z, hit[0], hit[1]);
});
canvas.addEventListener('pointermove', (e) => {
  pointer = { x: e.clientX, y: e.clientY, type: e.pointerType };
  if (painting) {
    const hit = hitAt(e.clientX, e.clientY);
    if (hit) paintPt = hit;
  }
});
const endPaint = () => {
  if (!painting) return;
  painting = false;
  requestSolve();
  autosave();
};
canvas.addEventListener('pointerup', endPaint);
canvas.addEventListener('pointercancel', endPaint);
canvas.addEventListener('pointerleave', () => { pointer = null; $('probe').hidden = true; });

// Hover probe: what the water is doing under the cursor.
const probe = $('probe');
function updateProbe() {
  if (!pointer || pointer.type === 'touch' || painting) { probe.hidden = true; return; }
  const hit = hitAt(pointer.x, pointer.y);
  if (!hit) { probe.hidden = true; return; }
  const [x, y] = hit;
  const z = sampleZ(beach.z, x, y);
  const depth = beach.waterLevel - z;
  if (tool !== 'view') R.setBrush(x, y, +radiusIn.value, true);
  let txt;
  if (depth <= 0.02) txt = `Dry sand · ${z.toFixed(1)} m above sea level`;
  else {
    txt = `Depth ${depth.toFixed(1)} m`;
    if (last) {
      const c = Math.min(N - 1, Math.round(y / DX)) * N + Math.min(N - 1, Math.round(x / DX));
      const hs = last.hs[c];
      const breaking = 1.6 * hs >= last.gamma[c] * depth * 0.95;
      const xi = last.xi[c];
      const kind = xi < 0.4 ? 'spilling' : xi < 2.4 ? 'plunging' : 'surging';
      txt += ` · waves ${hs.toFixed(1)} m` + (breaking ? ` · breaking, ${kind}` : '');
    }
  }
  probe.textContent = txt;
  probe.style.left = `${pointer.x}px`;
  probe.style.top = `${pointer.y}px`;
  probe.hidden = false;
}

window.addEventListener('keydown', (e) => {
  if (e.target.closest && e.target.closest('input, select, dialog')) return;
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); undo(); return; }
  const t = ['view', 'sand', 'dig', 'smooth', 'flatten', 'rock'][+e.key - 1];
  if (t) setTool(t);
  if (e.key === ' ') { e.preventDefault(); $('pause').click(); }
});

// --- main loop ---------------------------------------------------------------------------
let prevT = performance.now(), probeTick = 0;
function loop(t) {
  const dt = Math.min(0.05, (t - prevT) / 1000);
  prevT = t;
  if (painting && paintPt) {
    brush(beach, tool, paintPt[0], paintPt[1], +radiusIn.value, +strengthIn.value, dt, flatTarget);
    R.setHeight(beach);
    R.setBrush(paintPt[0], paintPt[1], +radiusIn.value, true);
    if (t - lastSolveReq > 350) { lastSolveReq = t; requestSolve(); }
  }
  if (++probeTick % 4 === 0) updateProbe();
  R.frame(paused ? 0 : dt * speed);
  requestAnimationFrame(loop);
}

// --- boot ---------------------------------------------------------------------------------
const savedSwell = store.get('shorebreak.swell');
if (savedSwell) {
  $('height').value = savedSwell.height;
  $('period').value = savedSwell.period;
  $('direction').value = savedSwell.direction;
  $('distance').value = savedSwell.distance;
}
readSwell();
readBrush();
$('o-speed').value = '1×';
renderSaved();
let start = null;
try { const cur = store.get('shorebreak.current'); if (cur) start = decodeBeach(cur); } catch { start = null; }
setBeach(start || { ...generate({ ...PRESETS.crescent, seed: 7 }), waterLevel: 0 });
requestAnimationFrame(loop);
window.__app = { beach, swell, R, setTool, requestSolve, viewInfo, preset: (k) => setBeach({ ...generate({ ...PRESETS[k], seed: 7 }), waterLevel: 0 }), get last() { return last; } };
