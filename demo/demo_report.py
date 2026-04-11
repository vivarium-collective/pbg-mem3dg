"""Demo: Mem3DG multi-configuration membrane report with 3D viewers.

Runs three distinct membrane simulations (osmotic deflation, membrane
patch bulging, tubular constriction), generates interactive 3D mesh
viewers with Three.js, Plotly charts, bigraph-viz diagrams, and
navigatable PBG document trees — all in a single self-contained HTML.
"""

import json
import os
import base64
import tempfile
import numpy as np
from process_bigraph import allocate_core
from pbg_mem3dg.processes import Mem3DGProcess
from pbg_mem3dg.composites import make_membrane_document


# ── Simulation Configs ──────────────────────────────────────────────

CONFIGS = [
    {
        'id': 'deflation',
        'title': 'Osmotic Deflation',
        'subtitle': 'Vesicle shrinkage under osmotic pressure',
        'description': (
            'A spherical vesicle (icosphere) deflates to 50% of its initial '
            'volume under harmonic osmotic pressure. Bending rigidity maintains '
            'the smooth spherical shape as the membrane contracts. This models '
            'osmotic shock in biological vesicles.'
        ),
        'config': {
            'mesh_type': 'icosphere',
            'radius': 1.0,
            'subdivision': 3,
            'Kbc': 5e-4,
            'tension_modulus': 0.01,
            'osmotic_strength': 0.1,
            'preferred_volume_fraction': 0.5,
            'characteristic_timestep': 1.0,
            'tolerance': 1e-12,
        },
        'n_snapshots': 25,
        'total_time': 500.0,
        'camera': [2.2, 1.5, 2.2],
        'color_scheme': 'indigo',
    },
    {
        'id': 'patch',
        'title': 'Membrane Patch Bulging',
        'subtitle': 'Flat hexagonal membrane inflated by pressure',
        'description': (
            'A flat hexagonal membrane patch with fixed boundary edges is '
            'inflated by constant osmotic pressure, forming a dome. This '
            'models local membrane deformation events such as bleb formation '
            'or pressure-driven bulging in confined geometries.'
        ),
        'config': {
            'mesh_type': 'hexagon',
            'radius': 1.0,
            'subdivision': 4,
            'Kbc': 3e-4,
            'tension_modulus': 0.01,
            'preferred_area_scale': 1.2,
            'osmotic_model': 'constant',
            'osmotic_pressure': 0.08,
            'boundary_condition': 'fixed',
            'characteristic_timestep': 0.5,
            'tolerance': 1e-14,
        },
        'n_snapshots': 25,
        'total_time': 1000.0,
        'camera': [0.0, -1.8, 1.5],
        'color_scheme': 'emerald',
    },
    {
        'id': 'tube',
        'title': 'Tubular Constriction',
        'subtitle': 'Cylinder narrowing under volume reduction',
        'description': (
            'A cylindrical tube constricts as osmotic pressure drives volume '
            'reduction to 60% of its initial value. The tube narrows uniformly '
            'while pinned boundary conditions hold the ends. This models '
            'membrane tubule dynamics in endoplasmic reticulum networks.'
        ),
        'config': {
            'mesh_type': 'cylinder',
            'radius': 0.5,
            'subdivision': 20,
            'axial_subdivision': 30,
            'Kbc': 5e-4,
            'tension_modulus': 0.05,
            'preferred_area_scale': 0.8,
            'osmotic_strength': 0.08,
            'preferred_volume_fraction': 0.6,
            'boundary_condition': 'pin',
            'characteristic_timestep': 0.5,
            'tolerance': 1e-14,
        },
        'n_snapshots': 25,
        'total_time': 1500.0,
        'camera': [1.5, 1.0, 5.0],
        'color_scheme': 'rose',
    },
]


def run_simulation(cfg_entry):
    """Run a single simulation, returning faces + snapshot list."""
    core = allocate_core()
    core.register_link('Mem3DGProcess', Mem3DGProcess)

    proc = Mem3DGProcess(config=cfg_entry['config'], core=core)
    state0 = proc.initial_state()
    faces = proc.get_faces()

    interval = cfg_entry['total_time'] / cfg_entry['n_snapshots']
    snapshots = [_snap(0.0, state0)]

    t = 0.0
    for i in range(cfg_entry['n_snapshots']):
        result = proc.update({}, interval=interval)
        t += interval
        if not result:
            break
        snapshots.append(_snap(round(t, 2), result))
        if result.get('converged'):
            break

    return faces, snapshots


def _snap(t, s):
    return {
        'time': t,
        'vertices': s['vertex_positions'],
        'curvatures': s['mean_curvatures'],
        'total_energy': s['total_energy'],
        'bending_energy': s['bending_energy'],
        'surface_energy': s['surface_energy'],
        'pressure_energy': s['pressure_energy'],
        'surface_area': s['surface_area'],
        'volume': s['volume'],
    }


def generate_bigraph_svg(cfg_entry):
    """Generate a colored bigraph-viz SVG for the composite document."""
    from bigraph_viz import plot_bigraph

    doc = make_membrane_document(**{
        k: v for k, v in cfg_entry['config'].items()
        if k in ('mesh_type', 'radius', 'subdivision', 'Kbc',
                 'tension_modulus', 'osmotic_strength',
                 'preferred_volume_fraction')
    }, interval=cfg_entry['total_time'] / cfg_entry['n_snapshots'])

    # Color nodes by role
    node_colors = {
        ('membrane',): '#6366f1',
        ('emitter',): '#8b5cf6',
        ('stores',): '#e0e7ff',
    }
    # Color store children
    if 'stores' in doc:
        for k in doc['stores']:
            node_colors[('stores', k)] = '#f0f0ff'

    outdir = tempfile.mkdtemp()
    plot_bigraph(
        state=doc,
        out_dir=outdir,
        filename='bigraph',
        file_format='svg',
        remove_process_place_edges=True,
        rankdir='TB',
        node_fill_colors=node_colors,
        dpi='150',
        node_label_size='13pt',
        port_labels=True,
        port_label_size='9pt',
    )
    svg_path = os.path.join(outdir, 'bigraph.svg')
    with open(svg_path) as f:
        svg = f.read()
    svg = svg.replace('<?xml version="1.0" encoding="UTF-8" standalone="no"?>', '')
    svg = svg.replace('<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"\n "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">', '')
    return svg.strip()


def build_pbg_document(cfg_entry):
    """Build the PBG composite document dict for display."""
    doc = make_membrane_document(**{
        k: v for k, v in cfg_entry['config'].items()
        if k in ('mesh_type', 'radius', 'subdivision', 'Kbc',
                 'tension_modulus', 'osmotic_strength',
                 'preferred_volume_fraction')
    }, interval=cfg_entry['total_time'] / cfg_entry['n_snapshots'])
    return doc


COLOR_SCHEMES = {
    'indigo': {'primary': '#6366f1', 'light': '#e0e7ff', 'dark': '#4338ca',
               'bg': '#eef2ff', 'accent': '#818cf8', 'text': '#312e81'},
    'emerald': {'primary': '#10b981', 'light': '#d1fae5', 'dark': '#059669',
                'bg': '#ecfdf5', 'accent': '#34d399', 'text': '#064e3b'},
    'rose': {'primary': '#f43f5e', 'light': '#ffe4e6', 'dark': '#e11d48',
             'bg': '#fff1f2', 'accent': '#fb7185', 'text': '#881337'},
}


def generate_html(sim_results, output_path):
    """Generate comprehensive HTML report."""

    sections_html = []
    all_js_data = {}

    for idx, (cfg, (faces, snapshots)) in enumerate(sim_results):
        sid = cfg['id']
        cs = COLOR_SCHEMES[cfg['color_scheme']]
        n_verts = len(snapshots[0]['vertices'])
        n_faces = len(faces)

        # Curvature range
        all_c = []
        for s in snapshots:
            all_c.extend(s['curvatures'])
        c_min = float(np.percentile(all_c, 2))
        c_max = float(np.percentile(all_c, 98))

        # Time series
        times = [s['time'] for s in snapshots]
        total_e = [s['total_energy'] for s in snapshots]
        bend_e = [s['bending_energy'] for s in snapshots]
        surf_e = [s['surface_energy'] for s in snapshots]
        press_e = [s['pressure_energy'] for s in snapshots]
        areas = [s['surface_area'] for s in snapshots]
        volumes = [s['volume'] for s in snapshots]

        # JS mesh data
        all_js_data[sid] = {
            'faces': faces,
            'snapshots': [{'time': s['time'], 'vertices': s['vertices'],
                           'curvatures': s['curvatures']} for s in snapshots],
            'curvature_range': [c_min, c_max],
            'camera': cfg['camera'],
            'charts': {
                'times': times, 'total_energy': total_e,
                'bending_energy': bend_e, 'surface_energy': surf_e,
                'pressure_energy': press_e, 'surface_area': areas,
                'volume': volumes,
            },
        }

        # Bigraph SVG
        print(f'  Generating bigraph diagram for {sid}...')
        bigraph_svg = generate_bigraph_svg(cfg)

        # PBG document JSON
        pbg_doc = build_pbg_document(cfg)

        # Volume/area changes
        sa0, sa1 = areas[0], areas[-1]
        v0, v1 = volumes[0], volumes[-1]
        sa_pct = f'{sa1/sa0*100:.1f}' if sa0 > 0 else 'N/A'
        v_pct = f'{v1/v0*100:.1f}' if v0 > 0 else 'N/A'

        section = f"""
    <div class="sim-section" id="sim-{sid}">
      <div class="sim-header" style="border-left: 4px solid {cs['primary']};">
        <div class="sim-number" style="background:{cs['light']}; color:{cs['dark']};">{idx+1}</div>
        <div>
          <h2 class="sim-title">{cfg['title']}</h2>
          <p class="sim-subtitle">{cfg['subtitle']}</p>
        </div>
      </div>
      <p class="sim-description">{cfg['description']}</p>

      <div class="metrics-row">
        <div class="metric"><span class="metric-label">Vertices</span><span class="metric-value">{n_verts:,}</span></div>
        <div class="metric"><span class="metric-label">Faces</span><span class="metric-value">{n_faces:,}</span></div>
        <div class="metric"><span class="metric-label">Energy</span><span class="metric-value">{total_e[-1]:.2e}</span></div>
        <div class="metric"><span class="metric-label">Area</span><span class="metric-value">{sa_pct}%</span><span class="metric-sub">{sa0:.2f} &rarr; {sa1:.2f}</span></div>
        <div class="metric"><span class="metric-label">Volume</span><span class="metric-value">{v_pct}%</span><span class="metric-sub">{v0:.2f} &rarr; {v1:.2f}</span></div>
        <div class="metric"><span class="metric-label">Snapshots</span><span class="metric-value">{len(snapshots)}</span></div>
      </div>

      <h3 class="subsection-title">3D Membrane Viewer</h3>
      <div class="viewer-wrap">
        <canvas id="canvas-{sid}" class="mesh-canvas"></canvas>
        <div class="viewer-info">
          <strong>{n_verts}</strong> vertices &middot; <strong>{n_faces}</strong> faces<br>
          Drag to rotate &middot; Scroll to zoom
        </div>
        <div class="colorbar-box">
          <div class="cb-title">Mean Curvature</div>
          <div class="cb-val">{c_max:.4f}</div>
          <div class="cb-gradient"></div>
          <div class="cb-val">{c_min:.4f}</div>
        </div>
        <div class="slider-controls">
          <button class="play-btn" style="border-color:{cs['primary']}; color:{cs['primary']};" onclick="togglePlay('{sid}')">Play</button>
          <label>Time</label>
          <input type="range" class="time-slider" id="slider-{sid}" min="0" max="{len(snapshots)-1}" value="0" step="1"
                 style="accent-color:{cs['primary']};">
          <span class="time-val" id="tval-{sid}">t = 0</span>
        </div>
      </div>

      <h3 class="subsection-title">Energy &amp; Geometry</h3>
      <div class="charts-row">
        <div class="chart-box"><div id="chart-energy-{sid}" class="chart"></div></div>
        <div class="chart-box"><div id="chart-components-{sid}" class="chart"></div></div>
        <div class="chart-box"><div id="chart-area-{sid}" class="chart"></div></div>
        <div class="chart-box"><div id="chart-volume-{sid}" class="chart"></div></div>
      </div>

      <div class="pbg-row">
        <div class="pbg-col">
          <h3 class="subsection-title">Bigraph Architecture</h3>
          <div class="bigraph-viewer" id="bgv-{sid}">
            <div class="bgv-inner" id="bgvi-{sid}">{bigraph_svg}</div>
            <div class="bgv-controls">
              <button class="bgv-btn" onclick="bgvZoom('{sid}',1.3)" title="Zoom in">+</button>
              <button class="bgv-btn" onclick="bgvZoom('{sid}',0.77)" title="Zoom out">&minus;</button>
              <button class="bgv-btn" onclick="bgvReset('{sid}')" title="Fit to view">Fit</button>
            </div>
            <div class="bgv-hint">Scroll to zoom &middot; Drag to pan</div>
          </div>
        </div>
        <div class="pbg-col">
          <h3 class="subsection-title">Composite Document</h3>
          <div class="json-tree" id="json-{sid}"></div>
        </div>
      </div>
    </div>
"""
        sections_html.append(section)

    # Navigation
    nav_items = ''.join(
        f'<a href="#sim-{c["id"]}" class="nav-link" '
        f'style="border-color:{COLOR_SCHEMES[c["color_scheme"]]["primary"]};">'
        f'{c["title"]}</a>'
        for c in [r[0] for r in sim_results])

    # PBG docs for JSON viewer
    pbg_docs = {r[0]['id']: build_pbg_document(r[0]) for r in sim_results}

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Mem3DG Membrane Simulation Report</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
       background:#fff; color:#1e293b; line-height:1.6; }}
.page-header {{
  background:linear-gradient(135deg,#f8fafc 0%,#eef2ff 50%,#fdf2f8 100%);
  border-bottom:1px solid #e2e8f0; padding:3rem;
}}
.page-header h1 {{ font-size:2.2rem; font-weight:800; color:#0f172a; margin-bottom:.3rem; }}
.page-header p {{ color:#64748b; font-size:.95rem; max-width:700px; }}
.nav {{ display:flex; gap:.8rem; padding:1rem 3rem; background:#f8fafc;
        border-bottom:1px solid #e2e8f0; position:sticky; top:0; z-index:100; }}
.nav-link {{ padding:.4rem 1rem; border-radius:8px; border:1.5px solid;
             text-decoration:none; font-size:.85rem; font-weight:600;
             transition:all .15s; }}
.nav-link:hover {{ transform:translateY(-1px); box-shadow:0 2px 8px rgba(0,0,0,.08); }}
.sim-section {{ padding:2.5rem 3rem; border-bottom:1px solid #e2e8f0; }}
.sim-header {{ display:flex; align-items:center; gap:1rem; margin-bottom:.8rem;
               padding-left:1rem; }}
.sim-number {{ width:36px; height:36px; border-radius:10px; display:flex;
               align-items:center; justify-content:center; font-weight:800; font-size:1.1rem; }}
.sim-title {{ font-size:1.5rem; font-weight:700; color:#0f172a; }}
.sim-subtitle {{ font-size:.9rem; color:#64748b; }}
.sim-description {{ color:#475569; font-size:.9rem; margin-bottom:1.5rem; max-width:800px; }}
.subsection-title {{ font-size:1.05rem; font-weight:600; color:#334155;
                     margin:1.5rem 0 .8rem; }}
.metrics-row {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
                gap:.8rem; margin-bottom:1.5rem; }}
.metric {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
           padding:.8rem; text-align:center; }}
.metric-label {{ display:block; font-size:.7rem; text-transform:uppercase;
                 letter-spacing:.06em; color:#94a3b8; margin-bottom:.2rem; }}
.metric-value {{ display:block; font-size:1.3rem; font-weight:700; color:#1e293b; }}
.metric-sub {{ display:block; font-size:.7rem; color:#94a3b8; }}
.viewer-wrap {{ position:relative; background:#f1f5f9; border:1px solid #e2e8f0;
                border-radius:14px; overflow:hidden; margin-bottom:1rem; }}
.mesh-canvas {{ width:100%; height:500px; display:block; cursor:grab; }}
.mesh-canvas:active {{ cursor:grabbing; }}
.viewer-info {{ position:absolute; top:.8rem; left:.8rem; background:rgba(255,255,255,.92);
                border:1px solid #e2e8f0; border-radius:8px; padding:.5rem .8rem;
                font-size:.75rem; color:#64748b; backdrop-filter:blur(4px); }}
.viewer-info strong {{ color:#1e293b; }}
.colorbar-box {{ position:absolute; top:.8rem; right:.8rem; background:rgba(255,255,255,.92);
                 border:1px solid #e2e8f0; border-radius:8px; padding:.6rem;
                 display:flex; flex-direction:column; align-items:center; gap:.2rem;
                 backdrop-filter:blur(4px); }}
.cb-title {{ font-size:.65rem; text-transform:uppercase; letter-spacing:.04em; color:#64748b; }}
.cb-gradient {{ width:16px; height:100px; border-radius:3px;
  background:linear-gradient(to bottom, #e61a0d, #e6c01a, #4dd94d, #12b5c9, #3112cc); }}
.cb-val {{ font-size:.65rem; color:#94a3b8; }}
.slider-controls {{ position:absolute; bottom:0; left:0; right:0;
                    background:linear-gradient(transparent,rgba(241,245,249,.97));
                    padding:1.5rem 1.5rem 1rem; display:flex; align-items:center; gap:.8rem; }}
.slider-controls label {{ font-size:.8rem; color:#64748b; }}
.time-slider {{ flex:1; height:5px; }}
.time-val {{ font-size:.95rem; font-weight:600; color:#334155; min-width:100px; text-align:right; }}
.play-btn {{ background:#fff; border:1.5px solid; padding:.3rem .8rem; border-radius:7px;
             cursor:pointer; font-size:.8rem; font-weight:600; transition:all .15s; }}
.play-btn:hover {{ transform:scale(1.05); }}
.charts-row {{ display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-bottom:1rem; }}
.chart-box {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; overflow:hidden; }}
.chart {{ height:280px; }}
.pbg-row {{ display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; margin-top:1rem; }}
.pbg-col {{ min-width:0; }}
.bigraph-viewer {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
                   position:relative; overflow:hidden; height:500px; cursor:grab; }}
.bigraph-viewer:active {{ cursor:grabbing; }}
.bgv-inner {{ position:absolute; transform-origin:0 0; }}
.bgv-inner svg {{ display:block; }}
.bgv-controls {{ position:absolute; top:.6rem; right:.6rem; display:flex; gap:.3rem; z-index:2; }}
.bgv-btn {{ width:32px; height:32px; border-radius:7px; border:1px solid #e2e8f0;
            background:#fff; color:#334155; font-size:1rem; font-weight:600;
            cursor:pointer; display:flex; align-items:center; justify-content:center;
            box-shadow:0 1px 3px rgba(0,0,0,.06); }}
.bgv-btn:hover {{ background:#eef2ff; border-color:#c7d2fe; }}
.bgv-hint {{ position:absolute; bottom:.5rem; left:.8rem; font-size:.7rem; color:#94a3b8; }}
.json-tree {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
              padding:1rem; max-height:500px; overflow-y:auto; font-family:'SF Mono',
              Menlo,Monaco,'Courier New',monospace; font-size:.78rem; line-height:1.5; }}
.jt-key {{ color:#7c3aed; font-weight:600; }}
.jt-str {{ color:#059669; }}
.jt-num {{ color:#2563eb; }}
.jt-bool {{ color:#d97706; }}
.jt-null {{ color:#94a3b8; }}
.jt-toggle {{ cursor:pointer; user-select:none; color:#94a3b8; margin-right:.3rem; }}
.jt-toggle:hover {{ color:#1e293b; }}
.jt-collapsed {{ display:none; }}
.jt-bracket {{ color:#64748b; }}
.footer {{ text-align:center; padding:2rem; color:#94a3b8; font-size:.8rem;
           border-top:1px solid #e2e8f0; }}
@media(max-width:900px) {{
  .charts-row,.pbg-row {{ grid-template-columns:1fr; }}
  .sim-section,.page-header {{ padding:1.5rem; }}
}}
</style>
</head>
<body>

<div class="page-header">
  <h1>Mem3DG Membrane Simulation Report</h1>
  <p>Three membrane mechanics simulations wrapped as <strong>process-bigraph</strong>
  Processes using discrete differential geometry. Each configuration demonstrates
  a distinct biophysical scenario with interactive 3D visualization.</p>
</div>

<div class="nav">{nav_items}</div>

{''.join(sections_html)}

<div class="footer">
  Generated by <strong>pbg-mem3dg</strong> &mdash;
  Mem3DG + process-bigraph &mdash;
  Discrete Differential Geometry on Triangulated Meshes
</div>

<script>
const DATA = {json.dumps(all_js_data)};
const DOCS = {json.dumps(pbg_docs, indent=2)};

// ─── JSON Tree Viewer ───
function renderJson(obj, depth) {{
  if (depth === undefined) depth = 0;
  if (obj === null) return '<span class="jt-null">null</span>';
  if (typeof obj === 'boolean') return '<span class="jt-bool">' + obj + '</span>';
  if (typeof obj === 'number') return '<span class="jt-num">' + obj + '</span>';
  if (typeof obj === 'string') return '<span class="jt-str">"' + obj.replace(/</g,'&lt;') + '"</span>';
  if (Array.isArray(obj)) {{
    if (obj.length === 0) return '<span class="jt-bracket">[]</span>';
    // Collapse arrays of primitives on one line if short
    if (obj.length <= 5 && obj.every(x => typeof x !== 'object' || x === null)) {{
      const items = obj.map(x => renderJson(x, depth+1)).join(', ');
      return '<span class="jt-bracket">[</span>' + items + '<span class="jt-bracket">]</span>';
    }}
    const id = 'jt' + Math.random().toString(36).slice(2,9);
    let html = '<span class="jt-toggle" onclick="toggleJt(\\'' + id + '\\')">&blacktriangledown;</span>';
    html += '<span class="jt-bracket">[</span> <span style="color:#94a3b8;font-size:.7rem;">' + obj.length + ' items</span>';
    html += '<div id="' + id + '" style="margin-left:1.2rem;">';
    obj.forEach((v, i) => {{ html += '<div>' + renderJson(v, depth+1) + (i < obj.length-1 ? ',' : '') + '</div>'; }});
    html += '</div><span class="jt-bracket">]</span>';
    return html;
  }}
  if (typeof obj === 'object') {{
    const keys = Object.keys(obj);
    if (keys.length === 0) return '<span class="jt-bracket">{{}}</span>';
    const id = 'jt' + Math.random().toString(36).slice(2,9);
    const collapsed = depth >= 2;
    let html = '<span class="jt-toggle" onclick="toggleJt(\\'' + id + '\\')">' +
               (collapsed ? '&blacktriangleright;' : '&blacktriangledown;') + '</span>';
    html += '<span class="jt-bracket">{{</span>';
    html += '<div id="' + id + '"' + (collapsed ? ' class="jt-collapsed"' : '') + ' style="margin-left:1.2rem;">';
    keys.forEach((k, i) => {{
      html += '<div><span class="jt-key">' + k + '</span>: ' +
              renderJson(obj[k], depth+1) + (i < keys.length-1 ? ',' : '') + '</div>';
    }});
    html += '</div><span class="jt-bracket">}}</span>';
    return html;
  }}
  return String(obj);
}}
function toggleJt(id) {{
  const el = document.getElementById(id);
  const tog = el.previousElementSibling.previousElementSibling || el.parentElement.querySelector('.jt-toggle');
  if (el.classList.contains('jt-collapsed')) {{
    el.classList.remove('jt-collapsed');
    // find toggle before this div
    const prev = el.previousElementSibling;
    if (prev && prev.previousElementSibling && prev.previousElementSibling.classList.contains('jt-toggle'))
      prev.previousElementSibling.innerHTML = '&blacktriangledown;';
  }} else {{
    el.classList.add('jt-collapsed');
    const prev = el.previousElementSibling;
    if (prev && prev.previousElementSibling && prev.previousElementSibling.classList.contains('jt-toggle'))
      prev.previousElementSibling.innerHTML = '&blacktriangleright;';
  }}
}}
// Render JSON trees
Object.keys(DOCS).forEach(sid => {{
  const el = document.getElementById('json-' + sid);
  if (el) el.innerHTML = renderJson(DOCS[sid], 0);
}});

// ─── Three.js Viewers ───
const viewers = {{}};
const playStates = {{}};

function initViewer(sid) {{
  const d = DATA[sid];
  const canvas = document.getElementById('canvas-' + sid);
  const W = canvas.parentElement.clientWidth;
  const H = 500;
  canvas.width = W * window.devicePixelRatio;
  canvas.height = H * window.devicePixelRatio;
  canvas.style.width = W + 'px';
  canvas.style.height = H + 'px';

  const renderer = new THREE.WebGLRenderer({{canvas, antialias:true}});
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(W, H);
  renderer.setClearColor(0xf1f5f9);

  const scene = new THREE.Scene();
  const cam = new THREE.PerspectiveCamera(45, W/H, 0.01, 100);
  cam.position.set(...d.camera);

  const controls = new THREE.OrbitControls(cam, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.8;

  scene.add(new THREE.AmbientLight(0xffffff, 0.5));
  const dl1 = new THREE.DirectionalLight(0xffffff, 0.7);
  dl1.position.set(3,5,4); scene.add(dl1);
  const dl2 = new THREE.DirectionalLight(0xcbd5e1, 0.4);
  dl2.position.set(-3,-2,-4); scene.add(dl2);

  const snap0 = d.snapshots[0];
  const nv = snap0.vertices.length;
  const positions = new Float32Array(nv * 3);
  const colors = new Float32Array(nv * 3);
  const geometry = new THREE.BufferGeometry();
  const indices = [];
  for (let i = 0; i < d.faces.length; i++)
    indices.push(d.faces[i][0], d.faces[i][1], d.faces[i][2]);
  geometry.setIndex(indices);
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

  const material = new THREE.MeshPhongMaterial({{
    vertexColors:true, side:THREE.DoubleSide, shininess:40, specular:0xdddddd, flatShading:false
  }});
  scene.add(new THREE.Mesh(geometry, material));

  const wireMat = new THREE.MeshBasicMaterial({{color:0x94a3b8, wireframe:true, transparent:true, opacity:0.06}});
  scene.add(new THREE.Mesh(geometry, wireMat));

  function updateMesh(idx) {{
    const snap = d.snapshots[idx];
    const [cmin, cmax] = d.curvature_range;
    for (let i = 0; i < nv; i++) {{
      positions[i*3]   = snap.vertices[i][0];
      positions[i*3+1] = snap.vertices[i][1];
      positions[i*3+2] = snap.vertices[i][2];
      let t = (snap.curvatures[i] - cmin) / (cmax - cmin + 1e-12);
      t = Math.max(0, Math.min(1, t));
      // Turbo-like colormap: blue → cyan → green → yellow → red
      let r, g, b;
      if (t < 0.25) {{
        const s = t / 0.25;
        r = 0.19; g = 0.07 + 0.63*s; b = 0.99 - 0.19*s;  // dark blue → cyan
      }} else if (t < 0.5) {{
        const s = (t - 0.25) / 0.25;
        r = 0.19 + 0.11*s; g = 0.70 + 0.15*s; b = 0.80 - 0.55*s;  // cyan → green
      }} else if (t < 0.75) {{
        const s = (t - 0.5) / 0.25;
        r = 0.30 + 0.60*s; g = 0.85 - 0.10*s; b = 0.25 - 0.15*s;  // green → yellow
      }} else {{
        const s = (t - 0.75) / 0.25;
        r = 0.90 + 0.10*s; g = 0.75 - 0.55*s; b = 0.10 - 0.05*s;  // yellow → red
      }}
      colors[i*3]   = r;
      colors[i*3+1] = g;
      colors[i*3+2] = b;
    }}
    geometry.attributes.position.needsUpdate = true;
    geometry.attributes.color.needsUpdate = true;
    geometry.computeVertexNormals();
  }}

  updateMesh(0);

  const slider = document.getElementById('slider-' + sid);
  const tval = document.getElementById('tval-' + sid);
  slider.addEventListener('input', () => {{
    const idx = parseInt(slider.value);
    updateMesh(idx);
    tval.textContent = 't = ' + d.snapshots[idx].time;
  }});

  viewers[sid] = {{ renderer, scene, cam, controls, updateMesh, slider, tval }};
  playStates[sid] = {{ playing: false, interval: null }};

  function animate() {{
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, cam);
  }}
  animate();
}}

function togglePlay(sid) {{
  const ps = playStates[sid];
  const v = viewers[sid];
  const d = DATA[sid];
  const btn = event.target;
  ps.playing = !ps.playing;
  if (ps.playing) {{
    btn.textContent = 'Pause';
    v.controls.autoRotate = false;
    ps.interval = setInterval(() => {{
      let idx = parseInt(v.slider.value) + 1;
      if (idx >= d.snapshots.length) idx = 0;
      v.slider.value = idx;
      v.updateMesh(idx);
      v.tval.textContent = 't = ' + d.snapshots[idx].time;
    }}, 350);
  }} else {{
    btn.textContent = 'Play';
    v.controls.autoRotate = true;
    clearInterval(ps.interval);
  }}
}}

// Init all viewers
Object.keys(DATA).forEach(sid => initViewer(sid));

// ─── Plotly Charts ───
const pLayout = {{
  paper_bgcolor:'#f8fafc', plot_bgcolor:'#f8fafc',
  font:{{ color:'#64748b', family:'-apple-system,sans-serif', size:11 }},
  margin:{{ l:50, r:15, t:35, b:40 }},
  xaxis:{{ gridcolor:'#e2e8f0', zerolinecolor:'#e2e8f0',
           title:{{ text:'Time', font:{{ size:10 }} }} }},
  yaxis:{{ gridcolor:'#e2e8f0', zerolinecolor:'#e2e8f0' }},
}};
const pCfg = {{ responsive:true, displayModeBar:false }};

Object.keys(DATA).forEach(sid => {{
  const c = DATA[sid].charts;
  Plotly.newPlot('chart-energy-'+sid, [{{
    x:c.times, y:c.total_energy, type:'scatter', mode:'lines+markers',
    line:{{ color:'#6366f1', width:2 }}, marker:{{ size:4 }},
  }}], {{...pLayout, title:{{ text:'Total Energy', font:{{ size:12, color:'#334155' }} }},
    yaxis:{{...pLayout.yaxis, title:{{ text:'Energy', font:{{ size:10 }} }} }}
  }}, pCfg);

  Plotly.newPlot('chart-components-'+sid, [
    {{ x:c.times, y:c.bending_energy, type:'scatter', mode:'lines+markers',
       line:{{ color:'#6366f1', width:1.5 }}, marker:{{ size:3 }}, name:'Bending' }},
    {{ x:c.times, y:c.surface_energy, type:'scatter', mode:'lines+markers',
       line:{{ color:'#10b981', width:1.5 }}, marker:{{ size:3 }}, name:'Surface' }},
    {{ x:c.times, y:c.pressure_energy, type:'scatter', mode:'lines+markers',
       line:{{ color:'#f43f5e', width:1.5 }}, marker:{{ size:3 }}, name:'Osmotic' }},
  ], {{...pLayout, title:{{ text:'Energy Components', font:{{ size:12, color:'#334155' }} }},
    yaxis:{{...pLayout.yaxis, title:{{ text:'Energy', font:{{ size:10 }} }} }},
    legend:{{ font:{{ size:9 }}, bgcolor:'rgba(0,0,0,0)' }}, showlegend:true
  }}, pCfg);

  Plotly.newPlot('chart-area-'+sid, [{{
    x:c.times, y:c.surface_area, type:'scatter', mode:'lines+markers',
    line:{{ color:'#10b981', width:2 }}, marker:{{ size:4 }},
    fill:'tozeroy', fillcolor:'rgba(16,185,129,0.06)',
  }}], {{...pLayout, title:{{ text:'Surface Area', font:{{ size:12, color:'#334155' }} }},
    yaxis:{{...pLayout.yaxis, title:{{ text:'Area', font:{{ size:10 }} }} }}, showlegend:false
  }}, pCfg);

  Plotly.newPlot('chart-volume-'+sid, [{{
    x:c.times, y:c.volume, type:'scatter', mode:'lines+markers',
    line:{{ color:'#f43f5e', width:2 }}, marker:{{ size:4 }},
    fill:'tozeroy', fillcolor:'rgba(244,63,94,0.06)',
  }}], {{...pLayout, title:{{ text:'Enclosed Volume', font:{{ size:12, color:'#334155' }} }},
    yaxis:{{...pLayout.yaxis, title:{{ text:'Volume', font:{{ size:10 }} }} }}, showlegend:false
  }}, pCfg);
}});

// ─── Bigraph Zoom/Pan ───
const bgvState = {{}};

function initBgv(sid) {{
  const ctr = document.getElementById('bgv-' + sid);
  const inner = document.getElementById('bgvi-' + sid);
  if (!ctr || !inner) return;
  const svg = inner.querySelector('svg');
  if (!svg) return;

  // Fit SVG to container initially
  const svgW = svg.getAttribute('width') ? parseFloat(svg.getAttribute('width')) : 800;
  const svgH = svg.getAttribute('height') ? parseFloat(svg.getAttribute('height')) : 400;
  const ctrW = ctr.clientWidth;
  const ctrH = ctr.clientHeight;
  const fitScale = Math.min(ctrW / svgW, ctrH / svgH) * 0.9;

  const st = {{ scale: fitScale, tx: (ctrW - svgW * fitScale) / 2, ty: (ctrH - svgH * fitScale) / 2,
                dragging: false, sx: 0, sy: 0, fitScale }};
  bgvState[sid] = st;

  function apply() {{
    inner.style.transform = `translate(${{st.tx}}px,${{st.ty}}px) scale(${{st.scale}})`;
  }}
  apply();

  ctr.addEventListener('wheel', function(e) {{
    e.preventDefault();
    const f = e.deltaY < 0 ? 1.15 : 0.87;
    const r = ctr.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    st.tx = mx - f * (mx - st.tx);
    st.ty = my - f * (my - st.ty);
    st.scale *= f;
    apply();
  }}, {{ passive: false }});

  ctr.addEventListener('mousedown', function(e) {{
    st.dragging = true;
    st.sx = e.clientX - st.tx;
    st.sy = e.clientY - st.ty;
    e.preventDefault();
  }});
  window.addEventListener('mousemove', function(e) {{
    if (!st.dragging) return;
    st.tx = e.clientX - st.sx;
    st.ty = e.clientY - st.sy;
    apply();
  }});
  window.addEventListener('mouseup', function() {{ st.dragging = false; }});
}}

function bgvZoom(sid, factor) {{
  const st = bgvState[sid];
  const ctr = document.getElementById('bgv-' + sid);
  const inner = document.getElementById('bgvi-' + sid);
  const cx = ctr.clientWidth / 2, cy = ctr.clientHeight / 2;
  st.tx = cx - factor * (cx - st.tx);
  st.ty = cy - factor * (cy - st.ty);
  st.scale *= factor;
  inner.style.transform = `translate(${{st.tx}}px,${{st.ty}}px) scale(${{st.scale}})`;
}}

function bgvReset(sid) {{
  const st = bgvState[sid];
  const ctr = document.getElementById('bgv-' + sid);
  const inner = document.getElementById('bgvi-' + sid);
  const svg = inner.querySelector('svg');
  const svgW = parseFloat(svg.getAttribute('width')) || 800;
  const svgH = parseFloat(svg.getAttribute('height')) || 400;
  st.scale = st.fitScale;
  st.tx = (ctr.clientWidth - svgW * st.scale) / 2;
  st.ty = (ctr.clientHeight - svgH * st.scale) / 2;
  inner.style.transform = `translate(${{st.tx}}px,${{st.ty}}px) scale(${{st.scale}})`;
}}

Object.keys(DATA).forEach(sid => initBgv(sid));
</script>
</body>
</html>"""

    with open(output_path, 'w') as f:
        f.write(html)
    print(f'Report saved to {output_path}')


def run_demo():
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(demo_dir, 'report.html')

    sim_results = []
    for cfg in CONFIGS:
        print(f'Running: {cfg["title"]}...')
        faces, snapshots = run_simulation(cfg)
        sim_results.append((cfg, (faces, snapshots)))
        print(f'  {len(snapshots)} snapshots collected')

    print('Generating HTML report...')
    generate_html(sim_results, output_path)


if __name__ == '__main__':
    run_demo()
