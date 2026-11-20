/* Hazard geography of each struck county, computed in a web browser (used by 17_hazard_geography.py).
 *
 * The FEMA flood map (Esri Living Atlas copy of the National Flood Hazard Layer), NOAA's storm-surge hazard tiles and
 * the terrain tiles are served to browsers but not to the analysis workspace, so this step runs in a browser
 * console (any page on https://tigerweb.geo.census.gov, so the block queries are same-origin):
 *   1. paste this file;  2. await H.runAll();  3. copy(JSON.stringify(H.results)) and save as data/hazard_geography_raw.json
 * For each county it takes every 2010 Census block (TIGERweb, internal point and 2010 housing units) and records
 * whether the block's internal point lies in
 *   - a FEMA Special Flood Hazard Area (SFHA_TF = 'T'; point-in-polygon on the returned polygons),
 *   - NOAA Category 3 storm-surge inundation (coloured pixel of the v3 surge hazard map at zoom 14),
 *   - a NOAA 'levee area' (grey hatched pixel of the same map),
 *   - ground below 1.5, 3 or 5 m (Tilezen terrarium elevation tiles, zoom 14).
 * Shares are housing-weighted. For 2005-2008 events, 2000 housing (H.HU00: Census 2000 block housing allocated to
 * 2010 tracts by land area, computed in 17_hazard_geography.py) is distributed within each tract like 2010 housing. */
window.H = {
  TW: 'https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/tigerWMS_Census2010/MapServer/18/query',
  FL: 'https://services.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services/USA_Flood_Hazard_Reduced_Set_gdb/FeatureServer/0/query',
  SU: 'https://tiles.arcgis.com/tiles/C8EMgrsFcRFL6LrL/arcgis/rest/services/Storm_Surge_HazardMaps_Category3_v3/MapServer/tile/14/',
  EL: 'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/14/', TC: {}, G: {}, results: [], HU00: {},
  // [state, county, coastal storm (surge map applies)]
  queue: [['22','071',1],['22','087',1],['22','075',1],['22','051',1],['28','045',1],['28','047',1],['28','059',1],['22','023',1],
          ['48','007',1],['48','167',1],['22','033',0],['37','155',0],['48','245',1],['12','087',1],['37','133',1],['12','005',1],['22','019',1]],
};
H.getj = async (u, body) => { for (let a = 0; a < 5; a++) { try { const r = body ? await fetch(u, {method: 'POST', headers: {'Content-Type': 'application/x-www-form-urlencoded'}, body}) : await fetch(u);
  if (r.ok) { const j = await r.json(); if (!j.error) return j; } } catch (e) {} await new Promise(r => setTimeout(r, 3000 * (a + 1))); } throw new Error('fail ' + u.slice(0, 80)); };
H.tileOf = (lat, lon) => { const n = 2 ** 14, x = (lon + 180) / 360 * n, r = lat * Math.PI / 180, y = (1 - Math.log(Math.tan(r) + 1 / Math.cos(r)) / Math.PI) / 2 * n;
  return [Math.floor(x), Math.floor(y), Math.min(255, Math.floor((x % 1) * 256)), Math.min(255, Math.floor((y % 1) * 256))]; };
H.tile = async (base, x, y, order) => { const k = base + x + '/' + y; if (k in H.TC) return H.TC[k]; const u = order === 'zyx' ? `${base}${y}/${x}` : `${base}${x}/${y}.png`; let d = null;
  for (let a = 0; a < 3 && !d; a++) { try { const r = await fetch(u); if (r.status === 404) break; if (r.ok) { const b = await createImageBitmap(await r.blob()); const c = new OffscreenCanvas(256, 256).getContext('2d'); c.drawImage(b, 0, 0); d = c.getImageData(0, 0, 256, 256).data; } } catch (e) { await new Promise(r => setTimeout(r, 1500)); } }
  H.TC[k] = d; return d; };
H.pip = (x, y, rings) => { let ins = false; for (const r of rings) for (let i = 0, j = r.length - 1; i < r.length; j = i++) { const [xi, yi] = r[i], [xj, yj] = r[j]; if (((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) ins = !ins; } return ins; };
H.sha = async t => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(t)))).map(b => b.toString(16).padStart(2, '0')).join('').slice(0, 16);
H.run = async (st, co, coastal) => {
  let bl = [], off = 0;
  while (true) { const j = await H.getj(`${H.TW}?where=STATE%3D'${st}'+AND+COUNTY%3D'${co}'&outFields=GEOID,INTPTLAT,INTPTLON,HU100&returnGeometry=false&orderByFields=GEOID&resultOffset=${off}&resultRecordCount=20000&f=json`);
    const f = j.features || []; bl = bl.concat(f.map(x => [x.attributes.GEOID, +x.attributes.INTPTLAT, +x.attributes.INTPTLON, x.attributes.HU100])); if (f.length < 20000) break; off += 20000; }
  bl.sort((a, b) => a[0] < b[0] ? -1 : 1);
  let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9; for (const b of bl) { x0 = Math.min(x0, b[2]); x1 = Math.max(x1, b[2]); y0 = Math.min(y0, b[1]); y1 = Math.max(y1, b[1]); }
  const bb = [x0 - 0.01, y0 - 0.01, x1 + 0.01, y1 + 0.01], polys = []; off = 0;
  while (true) { const body = `where=SFHA_TF%3D'T'&geometry=${bb.join(',')}&geometryType=esriGeometryEnvelope&inSR=4326&spatialRel=esriSpatialRelIntersects&outFields=FLD_ZONE&returnGeometry=true&outSR=4326&maxAllowableOffset=0.00003&geometryPrecision=6&resultOffset=${off}&resultRecordCount=750&f=json`;
    const j = await H.getj(H.FL, body), f = j.features || [];
    for (const p of f) { if (!p.geometry) continue; const R = p.geometry.rings; let a0 = 1e9, b0 = 1e9, a1 = -1e9, b1 = -1e9; for (const r of R) for (const [x, y] of r) { a0 = Math.min(a0, x); a1 = Math.max(a1, x); b0 = Math.min(b0, y); b1 = Math.max(b1, y); } polys.push({R, b: [a0, b0, a1, b1]}); }
    if (f.length === 0 || (!j.exceededTransferLimit && f.length < 750)) break; off += f.length; }
  const CS = 0.01, grid = {}; polys.forEach((p, i) => { for (let gx = Math.floor(p.b[0] / CS); gx <= Math.floor(p.b[2] / CS); gx++) for (let gy = Math.floor(p.b[1] / CS); gy <= Math.floor(p.b[3] / CS); gy++) (grid[gx + ',' + gy] = grid[gx + ',' + gy] || []).push(i); });
  const sfha = bl.map(([g, la, lo]) => (grid[Math.floor(lo / CS) + ',' + Math.floor(la / CS)] || []).some(i => { const b = polys[i].b; return lo >= b[0] && lo <= b[2] && la >= b[1] && la <= b[3] && H.pip(lo, la, polys[i].R); }) ? 1 : 0);
  const need = new Set(bl.map(([g, la, lo]) => { const t = H.tileOf(la, lo); return t[0] + '/' + t[1]; })), jobs = [];
  for (const k of need) { const [x, y] = k.split('/').map(Number); if (coastal) jobs.push(() => H.tile(H.SU, x, y, 'zyx')); jobs.push(() => H.tile(H.EL, x, y, 'xy')); }
  let q = 0; await Promise.all(Array.from({length: 12}, async () => { while (q < jobs.length) await jobs[q++](); }));
  const surge = [], levee = [], elev = [];
  for (const [g, la, lo] of bl) { const [x, y, i, j] = H.tileOf(la, lo), k = (j * 256 + i) * 4;
    if (coastal) { const d = H.TC[H.SU + x + '/' + y]; let s = 0, l = 0; if (d && d[k + 3] > 0) { if (d[k] === d[k + 1] && d[k + 1] === d[k + 2]) l = 1; else s = 1; } surge.push(s); levee.push(l); }
    const e = H.TC[H.EL + x + '/' + y]; elev.push(e ? e[k] * 256 + e[k + 1] + e[k + 2] / 256 - 32768 : NaN); }
  H.G[st + co] = {bl, sfha, surge, levee, elev, coastal, polys: polys.length, hash: await H.sha(bl.map(b => b[0]).join(','))};
  return H.summarise(st + co);
};
H.summarise = fips => {
  const {bl, sfha, surge, levee, elev, coastal, polys, hash} = H.G[fips];
  const prim = bl.map((b, i) => sfha[i] || (coastal && (surge[i] || levee[i])) ? 1 : 0), ab3 = elev.map(v => v < 3 ? 0 : 1);
  const w = m => { let a = 0, t = 0; bl.forEach((b, i) => { t += b[3]; if (m[i]) a += b[3]; }); return a / t; };
  const out = {fips, blocks: bl.length, hash, flood_polygons: polys, coastal: !!coastal, hu10: bl.reduce((a, b) => a + b[3], 0),
    outside_primary10: 1 - w(prim), outside_sfha10: 1 - w(sfha), above3m10: coastal ? w(ab3) : null, surge10: coastal ? w(surge) : null, levee10: coastal ? w(levee) : null};
  if (H.HU00[fips]) { const T = {};
    bl.forEach((b, i) => { const t = b[0].slice(5, 11), x = T[t] = T[t] || {hu: 0, n: 0, p: 0, s: 0, a: 0, pn: 0, sn: 0, an: 0}; x.hu += b[3]; x.n++; x.p += prim[i] * b[3]; x.s += sfha[i] * b[3]; x.a += ab3[i] * b[3]; x.pn += prim[i]; x.sn += sfha[i]; x.an += ab3[i]; });
    let tot = 0, P = 0, S = 0, A = 0, miss = 0;
    for (const [t, h0] of Object.entries(H.HU00[fips])) { const x = T[t]; if (!x) { miss += h0; continue; } tot += h0;
      if (x.hu > 0) { P += h0 * x.p / x.hu; S += h0 * x.s / x.hu; A += h0 * x.a / x.hu; } else { P += h0 * x.pn / x.n; S += h0 * x.sn / x.n; A += h0 * x.an / x.n; } }
    Object.assign(out, {hu00: tot, hu00_unmatched: miss, outside_primary00: 1 - P / tot, outside_sfha00: 1 - S / tot, above3m00: coastal ? A / tot : null}); }
  return out;
};
H.runAll = async () => { for (const [s, c, f] of H.queue) H.results.push(await H.run(s, c, f)); return H.results; };
