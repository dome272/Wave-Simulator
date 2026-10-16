// Real beaches: seabed grids cut from surveyed bathymetry by tools/cut_beach.py
// and shipped in beaches/<id>.json (the saved-beach format plus a `real`
// block). Each grid is rotated so the beach faces row 0; `facing` is the
// compass bearing it looks out to sea along, which turns the app's swell
// direction (degrees from straight in) into a compass bearing:
//   bearing the swell comes from = facing - direction.

export const REAL_BEACHES = [
  {
    id: 'davenport-landing',
    name: 'Davenport Landing',
    place: 'Santa Cruz County',
    blurb: 'Sand cove between mudstone reef shelves, with a deeper channel running out the middle',
    note: 'Surf-zone lidar from 2014 covers the reef shelves; the channel middle and the gaps between shelves are blended from older surveys, so treat their exact depths as approximate.',
  },
  {
    id: 'trestles',
    name: 'Lower Trestles',
    place: 'San Clemente',
    blurb: 'Cobble and sand peak off the San Mateo Creek delta',
    note: 'Surf-zone lidar from 2014. The cobble moves little; the sand on top shifts with the seasons.',
  },
];

const cache = new Map();
export async function loadRealBeach(id) {
  if (!cache.has(id)) {
    const p = fetch(new URL(`../beaches/${id}.json`, import.meta.url)).then((r) => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json();
    });
    cache.set(id, p);
    p.catch(() => cache.delete(id));
  }
  return cache.get(id);
}

const POINTS = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'];
export const norm360 = (d) => ((Math.round(d) % 360) + 360) % 360;
export const compass = (deg) => `${norm360(deg)}° ${POINTS[Math.round(norm360(deg) / 22.5) % 16]}`;
