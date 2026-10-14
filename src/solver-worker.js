// Runs the wave solver off the main thread so sculpting stays smooth.
import { solve } from './waves.js';

self.onmessage = (e) => {
  const { id, N, dx, z, wl, swell } = e.data;
  const res = solve({ N, dx, z, wl, swell });
  const transfer = [...res.S.map((a) => a.buffer), ...res.A.map((a) => a.buffer)];
  for (const k of ['h', 'D', 'gamma', 'plunge', 'surge', 'ux', 'uy', 'speed', 'hs', 'xi', 'tau', 'lambda']) transfer.push(res[k].buffer);
  self.postMessage({ id, res }, transfer);
};
