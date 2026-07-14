from __future__ import annotations
import math
import os
import re
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib.patches import FancyArrowPatch
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, minimize
pd.set_option('display.max_columns', 160)
pd.set_option('display.width', 220)
RUN_PROFILE = 'paper'
FLOW_SCALE = 1000.0
THETA = 0.6
ETA_GRID = [0.0, 0.2, 0.4, 0.6, 0.8]
R0 = 1.0
ETA = 0.6
K_PATHS = 5
PATH_RATIO_MAX = 2.2
ALPHA_GRID = [0.8, 0.85, 0.9, 0.96, 0.965, 0.975]
DPI_FIG = 600
SHOW_FIGURES = False
PARALLEL_EDGE_SEPARATION_FRAC = 0.01
PARALLEL_EDGE_LABEL_GAP_FRAC = 0.016
PARALLEL_EDGE_LABEL_TANGENT_FRAC = 0.006
USE_EQUALS_FLOW_LABELS = True
STOP_ON_NONCONVERGENCE = False
PLOT_ONLY_CONVERGED_MODEL_FLOWS = True
MODEL_B_SOLVER = 'ru_trust_constr'
MODEL_B_RU_MAXITER = 800
MODEL_B_RU_GTOL = 1e-07
MODEL_B_RU_XTOL = 1e-09
MODEL_B_RU_BARRIER_TOL = 1e-08
MODEL_B_RU_VERBOSE = 0
USE_CONVEX_RU_MODELB = MODEL_B_SOLVER.startswith('ru')
RU_MODELB_ALPHA_MIN = 0.95
RU_MODELB_MAXITER = MODEL_B_RU_MAXITER
RU_MODELB_FTOL = 1e-08
RU_MODELB_VERBOSE = bool(MODEL_B_RU_VERBOSE)
SOLVE_SCENARIO_TSUE = True
USE_COMMON_TAIL_CERTIFIED_EQUALITY = True
COMMON_TAIL_CERTIFY_ALPHA_VALUES = [0.9]
COMMON_TAIL_CERT_TOL = 1e-08
RANDOM_SEED = 123
np.random.seed(RANDOM_SEED)
TOL_REL_GAP_B = 0.0002
TOL_REL_GAP_A = 0.0005
MAX_ITER_B = 30000
MAX_ITER_A = 2500
STRICT_CONVERGENCE = True
SCRIPT_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd().resolve()
INPUT_DIR = Path(os.environ.get('SIOUX_INPUT_DIR', str(SCRIPT_DIR))).expanduser().resolve()

def _resolve_input_file(env_name: str, default_names: list[str], required: bool=True) -> Path | None:
    explicit = os.environ.get(env_name)
    search_bases = []
    for base in [INPUT_DIR, SCRIPT_DIR, Path.cwd()]:
        base = base.expanduser().resolve()
        if base not in search_bases:
            search_bases.append(base)
    if explicit is not None:
        explicit = explicit.strip()
        if explicit.lower() in {'', 'none', 'null', 'false'}:
            if required:
                raise FileNotFoundError(f'{env_name} is required but was set to {explicit!r}.')
            return None
        p = Path(explicit).expanduser()
        candidates = [p] if p.is_absolute() else [base / p for base in search_bases]
    else:
        candidates = [base / name for base in search_bases for name in default_names]
        for base in search_bases:
            for name in default_names:
                name_path = Path(name)
                candidates.extend(sorted(base.glob(f'{name_path.stem}(*){name_path.suffix}')))
    seen = set()
    tried = []
    for p in candidates:
        p = p.expanduser()
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        tried.append(key)
        if p.is_file():
            return p.resolve()
    if required:
        raise FileNotFoundError(f'Could not find required input file for {env_name}. Tried: {tried}')
    return None
NET_FILE = _resolve_input_file('SIOUX_NET_FILE', ['SiouxFalls_net.txt', 'SiouxFalls_net.tntp'], required=True)
TRIPS_FILE = _resolve_input_file('SIOUX_TRIPS_FILE', ['SiouxFalls_trips.txt', 'SiouxFalls_trips.tntp'], required=True)
NODE_FILE = _resolve_input_file('SIOUX_NODE_FILE', ['SiouxFalls_node.txt', 'SiouxFalls_node.tntp'], required=True)
FLOW_FILE = _resolve_input_file('SIOUX_FLOW_FILE', ['SiouxFalls_flow.txt', 'SiouxFalls_flow.tntp'], required=False)
HAS_FLOW_FILE = FLOW_FILE is not None
OUTPUT_BASE = Path(os.environ.get('SIOUX_OUTPUT_DIR', str(NET_FILE.parent))).expanduser()
OUTPUT_DIR = OUTPUT_BASE / 'sioux_falls_full_od_modelAB_reference_style_outputs'
FIG_DIR = OUTPUT_DIR / 'plots_600dpi'
TABLE_DIR = OUTPUT_DIR / 'tables'
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
print('RUN_PROFILE:', RUN_PROFILE)
print('INPUT_DIR:', INPUT_DIR)
print('NET_FILE:', NET_FILE)
print('TRIPS_FILE:', TRIPS_FILE)
print('NODE_FILE:', NODE_FILE)
print('FLOW_FILE:', FLOW_FILE if FLOW_FILE is not None else 'None / optional input not provided')
print('OUTPUT_DIR:', OUTPUT_DIR)

def parse_net_tntp(path: Path, flow_scale: float=1000.0) -> pd.DataFrame:
    rows = []
    in_data = False
    for line in path.read_text(errors='ignore').splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith('<END OF METADATA>'):
            in_data = True
            continue
        if not in_data or s.startswith('~'):
            continue
        s = s.split(';')[0].strip()
        if not s:
            continue
        parts = re.split('\\s+', s)
        if len(parts) >= 10:
            rows.append(parts[:10])
    cols = ['init', 'term', 'capacity', 'length', 'free_time', 'B', 'power', 'speed', 'toll', 'type']
    df = pd.DataFrame(rows, columns=cols)
    for c in cols:
        df[c] = pd.to_numeric(df[c])
    df['init'] = df['init'].astype(int)
    df['term'] = df['term'].astype(int)
    df['capacity_raw'] = df['capacity']
    df['capacity'] = df['capacity'] / flow_scale
    df['link_id0'] = np.arange(len(df), dtype=int)
    df['link_id'] = np.arange(1, len(df) + 1, dtype=int)
    return df

def parse_trips_tntp(path: Path, flow_scale: float=1000.0) -> pd.DataFrame:
    od = []
    origin = None
    for line in path.read_text(errors='ignore').splitlines():
        s = line.strip()
        if not s or s.startswith('<'):
            continue
        m = re.match('Origin\\s+(\\d+)', s)
        if m:
            origin = int(m.group(1))
            continue
        if origin is not None:
            for dest, val in re.findall('(\\d+)\\s*:\\s*([0-9.]+)', s):
                q_raw = float(val)
                d = int(dest)
                if q_raw > 0 and d != origin:
                    od.append((origin, d, q_raw / flow_scale, q_raw))
    return pd.DataFrame(od, columns=['origin', 'dest', 'demand', 'demand_raw'])

def parse_nodes_tntp(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(errors='ignore').splitlines():
        s = line.strip().replace(';', '')
        if not s or s.lower().startswith('node'):
            continue
        parts = re.split('\\s+', s)
        if len(parts) >= 3:
            rows.append((int(parts[0]), float(parts[1]), float(parts[2])))
    return pd.DataFrame(rows, columns=['node', 'x', 'y'])

def parse_flow_tntp(path: Path | None, flow_scale: float=1000.0) -> pd.DataFrame:
    rows = []
    if path is None or not Path(path).exists():
        return pd.DataFrame(columns=['init', 'term', 'volume', 'cost', 'volume_raw'])
    path = Path(path)
    for line in path.read_text(errors='ignore').splitlines():
        s = line.strip()
        if not s or s.lower().startswith('from'):
            continue
        parts = re.split('\\s+', s)
        if len(parts) >= 4:
            rows.append((int(parts[0]), int(parts[1]), float(parts[2]) / flow_scale, float(parts[3]), float(parts[2])))
    return pd.DataFrame(rows, columns=['init', 'term', 'volume', 'cost', 'volume_raw'])
net = parse_net_tntp(NET_FILE, FLOW_SCALE)
trips = parse_trips_tntp(TRIPS_FILE, FLOW_SCALE)
nodes = parse_nodes_tntp(NODE_FILE)
flow_ref = parse_flow_tntp(FLOW_FILE, FLOW_SCALE)
net = net.merge(flow_ref[['init', 'term', 'volume', 'cost', 'volume_raw']], on=['init', 'term'], how='left')
net['ref_volume'] = net['volume'].fillna(0.0)
net['ref_volume_raw'] = net['volume_raw'].fillna(0.0)
net['ref_tntp_cost'] = net['cost'].fillna(np.nan)
net = net.drop(columns=[c for c in ['volume', 'volume_raw', 'cost'] if c in net.columns])
HAS_REFERENCE_FLOW = len(flow_ref) > 0
node_xy = nodes.set_index('node')[['x', 'y']]
net['x0'] = net['init'].map(node_xy['x'])
net['y0'] = net['init'].map(node_xy['y'])
net['x1'] = net['term'].map(node_xy['x'])
net['y1'] = net['term'].map(node_xy['y'])
net['xm'] = 0.5 * (net['x0'] + net['x1'])
net['ym'] = 0.5 * (net['y0'] + net['y1'])
print(f'Links: {len(net)}')
print(f'Nodes: {len(nodes)}')
print(f'Positive-demand OD pairs retained: {len(trips)}')
print(f'Total demand: {trips.demand_raw.sum():,.1f} veh/h = {trips.demand.sum():.1f} thousand veh/h')
print(f'Reference flow rows parsed: {len(flow_ref)}')
if not HAS_REFERENCE_FLOW:
    print('No optional flow file was parsed. Reference-style input plots will use link id labels and capacity-based widths.')
display(net.head())
display(trips.head())

def _parallel_edge_layout(net_df: pd.DataFrame) -> Dict[int, dict]:
    node_lookup = nodes.set_index('node')[['x', 'y']]
    groups: Dict[Tuple[int, int], List[int]] = {}
    for idx, r in net_df.iterrows():
        key = tuple(sorted((int(r.init), int(r.term))))
        groups.setdefault(key, []).append(int(idx))
    layout: Dict[int, dict] = {}
    debug_rows: List[dict] = []
    for key, idxs in groups.items():
        u_can, v_can = key
        x_u, y_u = (float(node_lookup.loc[u_can, 'x']), float(node_lookup.loc[u_can, 'y']))
        x_v, y_v = (float(node_lookup.loc[v_can, 'x']), float(node_lookup.loc[v_can, 'y']))
        dx_c, dy_c = (x_v - x_u, y_v - y_u)
        len_c = math.hypot(dx_c, dy_c)
        if len_c <= 0:
            ux_c, uy_c, nx_c, ny_c = (1.0, 0.0, 0.0, 1.0)
        else:
            ux_c, uy_c = (dx_c / len_c, dy_c / len_c)
            nx_c, ny_c = (-uy_c, ux_c)
        if len(idxs) == 1:
            i = idxs[0]
            layout[i] = {'bidirectional': False, 'pair_key': key, 'lane_side': 0.0, 'nx': nx_c, 'ny': ny_c, 'ux_can': ux_c, 'uy_can': uy_c}
            debug_rows.append({'link_id': int(net_df.loc[i, 'link_id']), 'init': int(net_df.loc[i, 'init']), 'term': int(net_df.loc[i, 'term']), 'pair_key': f'{u_can}-{v_can}', 'bidirectional': False, 'lane_side': 0.0, 'nx': nx_c, 'ny': ny_c})
            continue
        canonical = [i for i in idxs if int(net_df.loc[i, 'init']) == u_can and int(net_df.loc[i, 'term']) == v_can]
        reverse = [i for i in idxs if int(net_df.loc[i, 'init']) == v_can and int(net_df.loc[i, 'term']) == u_can]
        other = [i for i in idxs if i not in canonical and i not in reverse]
        ordered = sorted(canonical, key=lambda i: int(net_df.loc[i, 'link_id'])) + sorted(reverse, key=lambda i: int(net_df.loc[i, 'link_id'])) + sorted(other, key=lambda i: int(net_df.loc[i, 'link_id']))
        if len(ordered) == 2 and len(canonical) == 1 and (len(reverse) == 1):
            side_map = {canonical[0]: +1.0, reverse[0]: -1.0}
        else:
            sides = np.linspace(+1.0, -1.0, len(ordered))
            side_map = {i: float(s) for i, s in zip(ordered, sides)}
        for i in ordered:
            side = side_map[i]
            layout[i] = {'bidirectional': True, 'pair_key': key, 'lane_side': side, 'nx': nx_c, 'ny': ny_c, 'ux_can': ux_c, 'uy_can': uy_c}
            debug_rows.append({'link_id': int(net_df.loc[i, 'link_id']), 'init': int(net_df.loc[i, 'init']), 'term': int(net_df.loc[i, 'term']), 'pair_key': f'{u_can}-{v_can}', 'bidirectional': True, 'lane_side': side, 'nx': nx_c, 'ny': ny_c})
    try:
        pd.DataFrame(debug_rows).sort_values('link_id').to_csv(TABLE_DIR / 'parallel_edge_plot_layout.csv', index=False)
    except Exception:
        pass
    return layout

def plot_sioux_reference_style(values_for_width: np.ndarray | None=None, values_for_color: np.ndarray | None=None, labels: np.ndarray | None=None, title: str='', out_file: Path | None=None, colorbar_label: str | None=None, cmap: str='viridis', black_edges: bool=False, node_size: float=185, edge_label_fontsize: float=5.0, node_fontsize: float=7.0):
    fig, ax = plt.subplots(figsize=(8.0, 7.5))
    n_links_local = len(net)
    values_for_width = np.ones(n_links_local) if values_for_width is None else np.asarray(values_for_width, dtype=float)
    if values_for_color is None:
        values_for_color = np.zeros(n_links_local, dtype=float)
    else:
        values_for_color = np.asarray(values_for_color, dtype=float)
    max_width_val = max(float(np.nanmax(np.abs(values_for_width))), 1e-12)
    widths = 0.48 + 1.35 * np.sqrt(np.maximum(values_for_width, 0.0) / max_width_val)
    if black_edges:
        colors = ['#222222'] * n_links_local
        norm = None
        cmap_obj = None
    else:
        vmin = float(np.nanmin(values_for_color))
        vmax = float(np.nanmax(values_for_color))
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1.0
        vmax_abs = max(abs(vmin), abs(vmax))
        norm = plt.Normalize(vmin=-vmax_abs, vmax=vmax_abs) if vmin < 0 < vmax else plt.Normalize(vmin=vmin, vmax=vmax)
        cmap_obj = plt.get_cmap(cmap)
        colors = [cmap_obj(norm(v)) for v in values_for_color]
    xrange = float(nodes['x'].max() - nodes['x'].min())
    yrange = float(nodes['y'].max() - nodes['y'].min())
    scale = max(xrange, yrange)
    lane_half_gap = 0.5 * PARALLEL_EDGE_SEPARATION_FRAC * scale
    node_shrink = 0.026 * scale
    label_outward_gap = PARALLEL_EDGE_LABEL_GAP_FRAC * scale
    label_tangent_gap = PARALLEL_EDGE_LABEL_TANGENT_FRAC * scale
    layout = _parallel_edge_layout(net)
    for idx, r in net.iterrows():
        x0, y0, x1, y1 = (float(r.x0), float(r.y0), float(r.x1), float(r.y1))
        dx, dy = (x1 - x0, y1 - y0)
        length = math.hypot(dx, dy)
        if length <= 0:
            continue
        ux, uy = (dx / length, dy / length)
        local_nx, local_ny = (-uy, ux)
        lay = layout[int(idx)]
        is_bidir = bool(lay['bidirectional'])
        side = float(lay['lane_side'])
        nx_c, ny_c = (float(lay['nx']), float(lay['ny']))
        lane_offset = side * lane_half_gap if is_bidir else 0.0
        start = (x0 + nx_c * lane_offset + ux * node_shrink, y0 + ny_c * lane_offset + uy * node_shrink)
        end = (x1 + nx_c * lane_offset - ux * node_shrink, y1 + ny_c * lane_offset - uy * node_shrink)
        patch = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=8.0, linewidth=float(widths[int(idx)]), color=colors[int(idx)], alpha=0.94, zorder=2, shrinkA=0, shrinkB=0)
        ax.add_patch(patch)
        if labels is not None:
            if is_bidir:
                t = 0.5 + 0.045 * side
                xm = start[0] + t * (end[0] - start[0]) + nx_c * side * label_outward_gap + ux * side * label_tangent_gap
                ym = start[1] + t * (end[1] - start[1]) + ny_c * side * label_outward_gap + uy * side * label_tangent_gap
            else:
                xm = 0.5 * (start[0] + end[0]) + local_nx * 0.0025 * scale
                ym = 0.5 * (start[1] + end[1]) + local_ny * 0.0025 * scale
            angle = math.degrees(math.atan2(dy, dx))
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            ax.text(xm, ym, str(labels[int(idx)]), fontsize=edge_label_fontsize, rotation=angle, rotation_mode='anchor', ha='center', va='center', color='black', bbox=dict(facecolor='white', edgecolor='none', alpha=0.92, pad=0.2), zorder=4)
    ax.scatter(nodes['x'], nodes['y'], s=node_size, facecolor='yellow', edgecolor='black', linewidth=0.85, zorder=5)
    for _, r in nodes.iterrows():
        ax.text(float(r.x), float(r.y), str(int(r.node)), fontsize=node_fontsize, ha='center', va='center', zorder=6)
    if not black_edges and colorbar_label is not None:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.015)
        cbar.set_label(colorbar_label, fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=10)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    pad_x = 0.1 * xrange
    pad_y = 0.1 * yrange
    ax.set_xlim(nodes['x'].min() - pad_x, nodes['x'].max() + pad_x)
    ax.set_ylim(nodes['y'].min() - pad_y, nodes['y'].max() + pad_y)
    fig.tight_layout(pad=0.2)
    if out_file is not None:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file, dpi=DPI_FIG, bbox_inches='tight')
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)

def id_flow_labels(flow_thousand: np.ndarray | pd.Series) -> np.ndarray:
    raw = np.asarray(flow_thousand, dtype=float) * FLOW_SCALE
    if USE_EQUALS_FLOW_LABELS:
        return np.array([f'L{int(lid)}={int(round(v))}' for lid, v in zip(net['link_id'], raw)], dtype=object)
    return np.array([f'{int(lid)}/{int(round(v))}' for lid, v in zip(net['link_id'], raw)], dtype=object)

def link_id_labels() -> np.ndarray:
    return np.array([f'L{int(lid)}' if USE_EQUALS_FLOW_LABELS else str(int(lid)) for lid in net['link_id']], dtype=object)

def reference_plot_widths() -> np.ndarray:
    return net['ref_volume'].to_numpy() if HAS_REFERENCE_FLOW else net['capacity'].to_numpy()

def reference_plot_labels() -> np.ndarray:
    return id_flow_labels(net['ref_volume'].to_numpy()) if HAS_REFERENCE_FLOW else link_id_labels()
plot_sioux_reference_style(values_for_width=reference_plot_widths(), values_for_color=reference_plot_widths() * FLOW_SCALE if HAS_REFERENCE_FLOW else reference_plot_widths() * FLOW_SCALE, labels=reference_plot_labels(), title='Sioux Falls reference assignment: link id / reference flow' if HAS_REFERENCE_FLOW else 'Sioux Falls network: link id labels', out_file=FIG_DIR / 'sioux_falls_reference_link_id_flow_600dpi.png', colorbar_label='Reference flow (veh/h)' if HAS_REFERENCE_FLOW else 'Capacity (veh/h)', cmap='Reds', black_edges=False)
print('Saved reference-style baseline plot:', FIG_DIR / 'sioux_falls_reference_link_id_flow_600dpi.png')

def _link_lookup_by_id() -> pd.DataFrame:
    return net[['link_id', 'init', 'term', 'xm', 'ym']].copy().set_index('link_id')

def build_link_region_groups() -> pd.DataFrame:
    link_lookup = _link_lookup_by_id()
    group_specs = [('Upper flooding core', [6, 8, 9, 11, 12, 15], 'Flood-upper links with the largest realized severity in the scenario-rule table.', 'flood_upper'), ('Middle flooding band', [27, 29, 30, 32, 33, 36, 48, 49, 50, 51, 52, 55], 'High-severity middle-band flooding links; this broad band contains both west and east middle links.', 'flood_middle'), ('Middle-east incident core', [29, 30, 48, 49, 51, 52], 'Localized middle-east incident links with the largest realized incident severity.', 'minor_incident'), ('Middle-west core', [27, 32, 33, 36], 'West-side subset of the middle flooding band, separated for visual comparison with the middle-east incident core.', 'flood_middle'), ('Lower flooding core', [42, 46, 59, 61, 70, 72], 'Flood-lower links with the largest realized severity in the scenario-rule table.', 'flood_lower')]
    rows = []
    for group, link_ids, desc, source_scenario in group_specs:
        for lid in link_ids:
            if lid not in link_lookup.index:
                continue
            r = link_lookup.loc[lid]
            rows.append({'region_group': group, 'link_id': int(lid), 'init': int(r['init']), 'term': int(r['term']), 'source_scenario': source_scenario, 'description': desc})
    return pd.DataFrame(rows)

def plot_sioux_region_groups(region_df: pd.DataFrame, out_file: Path):
    fig, ax = plt.subplots(figsize=(10.0, 8.0))
    xrange = float(nodes['x'].max() - nodes['x'].min())
    yrange = float(nodes['y'].max() - nodes['y'].min())
    scale = max(xrange, yrange)
    lane_half_gap = 0.5 * PARALLEL_EDGE_SEPARATION_FRAC * scale
    node_shrink = 0.026 * scale
    layout = _parallel_edge_layout(net)

    def edge_points(idx: int):
        r = net.loc[idx]
        x0, y0, x1, y1 = (float(r.x0), float(r.y0), float(r.x1), float(r.y1))
        dx, dy = (x1 - x0, y1 - y0)
        length = math.hypot(dx, dy)
        if length <= 0:
            return None
        ux, uy = (dx / length, dy / length)
        lay = layout[int(idx)]
        side = float(lay['lane_side'])
        nx_c, ny_c = (float(lay['nx']), float(lay['ny']))
        is_bidir = bool(lay['bidirectional'])
        lane_offset = side * lane_half_gap if is_bidir else 0.0
        start = (x0 + nx_c * lane_offset + ux * node_shrink, y0 + ny_c * lane_offset + uy * node_shrink)
        end = (x1 + nx_c * lane_offset - ux * node_shrink, y1 + ny_c * lane_offset - uy * node_shrink)
        return (start, end, ux, uy, nx_c, ny_c, side, is_bidir)
    group_order = ['Middle flooding band', 'Upper flooding core', 'Lower flooding core', 'Middle-west core', 'Middle-east incident core']
    group_colors = {'Upper flooding core': '#d73027', 'Middle flooding band': '#fdae61', 'Middle-east incident core': '#7b3294', 'Middle-west core': '#1a9850', 'Lower flooding core': '#4575b4'}
    group_widths = {'Middle flooding band': 2.8, 'Upper flooding core': 3.6, 'Lower flooding core': 3.6, 'Middle-west core': 4.4, 'Middle-east incident core': 4.4}
    membership = region_df.groupby('link_id')['region_group'].apply(lambda s: list(dict.fromkeys(s))).to_dict()
    for idx, r in net.iterrows():
        pts = edge_points(int(idx))
        if pts is None:
            continue
        start, end, ux, uy, nx_c, ny_c, side, is_bidir = pts
        patch = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=6.5, linewidth=0.7, color='#d0d0d0', alpha=0.7, zorder=1, shrinkA=0, shrinkB=0)
        ax.add_patch(patch)
        t = 0.5 + (0.04 * side if is_bidir else 0.0)
        sign = side if is_bidir else 1.0
        xm = start[0] + t * (end[0] - start[0]) + nx_c * sign * 0.012 * scale + ux * sign * 0.004 * scale
        ym = start[1] + t * (end[1] - start[1]) + ny_c * sign * 0.012 * scale + uy * sign * 0.004 * scale
        groups = membership.get(int(r.link_id), [])
        bbox_fc = 'white' if groups else '#f8f8f8'
        ax.text(xm, ym, f'L{int(r.link_id)}', fontsize=5.2, ha='center', va='center', color='black', bbox=dict(facecolor=bbox_fc, edgecolor='none', alpha=0.95, pad=0.18), zorder=2)
    for g in group_order:
        sub = region_df[region_df['region_group'].eq(g)]
        for _, row in sub.iterrows():
            idx_arr = net.index[net['link_id'].eq(int(row.link_id))].to_list()
            if not idx_arr:
                continue
            idx = int(idx_arr[0])
            pts = edge_points(idx)
            if pts is None:
                continue
            start, end, *_ = pts
            patch = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=9.0, linewidth=group_widths[g], color=group_colors[g], alpha=0.95, zorder=4 if g != 'Middle flooding band' else 3, shrinkA=0, shrinkB=0)
            ax.add_patch(patch)
    ax.scatter(nodes['x'], nodes['y'], s=185, facecolor='yellow', edgecolor='black', linewidth=0.85, zorder=7)
    for _, r in nodes.iterrows():
        ax.text(float(r.x), float(r.y), str(int(r.node)), fontsize=7.0, ha='center', va='center', zorder=8)
    handles = [plt.Line2D([0], [0], color=group_colors[g], lw=4.0, label=g) for g in group_order]
    handles.append(plt.Line2D([0], [0], color='#d0d0d0', lw=2.0, label='Other links'))
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=7, frameon=True, framealpha=0.92, borderaxespad=0.0)
    ax.set_title('Sioux Falls link groups used in the scenario table', fontsize=10)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    pad_x = 0.06 * xrange
    pad_y = 0.06 * yrange
    ax.set_xlim(nodes['x'].min() - pad_x, nodes['x'].max() + pad_x)
    ax.set_ylim(nodes['y'].min() - pad_y, nodes['y'].max() + pad_y)
    fig.tight_layout(rect=[0.0, 0.0, 0.84, 1.0], pad=0.2)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=DPI_FIG, bbox_inches='tight')
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)
link_region_groups = build_link_region_groups()
link_region_groups.to_csv(TABLE_DIR / 'sioux_falls_link_region_groups_for_scenario_table.csv', index=False)
region_membership_wide = link_region_groups.groupby(['link_id', 'init', 'term'])['region_group'].apply(lambda s: '; '.join(sorted(set(s)))).reset_index(name='region_memberships').sort_values('link_id')
region_membership_wide.to_csv(TABLE_DIR / 'sioux_falls_link_region_membership_wide.csv', index=False)
plot_sioux_region_groups(link_region_groups, FIG_DIR / 'sioux_falls_link_region_groups_600dpi.png')
print('Saved region-group map:', FIG_DIR / 'sioux_falls_link_region_groups_600dpi.png')
print('Saved region-group membership tables under:', TABLE_DIR)

def build_graph(net_df: pd.DataFrame, weight_col: str='free_time') -> nx.DiGraph:
    G = nx.DiGraph()
    for _, r in net_df.iterrows():
        G.add_edge(int(r.init), int(r.term), weight=float(r[weight_col]), link_id0=int(r.link_id0), length=float(r.length))
    return G

def generate_candidate_paths(net_df: pd.DataFrame, trips_df: pd.DataFrame, k_paths: int=5, ratio_max: float=2.2) -> Tuple[pd.DataFrame, List[np.ndarray], List[Tuple[int, int]]]:
    G = build_graph(net_df)
    path_rows: List[dict] = []
    paths_by_od: List[np.ndarray] = []
    skipped: List[Tuple[int, int]] = []
    for od_idx, row in trips_df.iterrows():
        o, d = (int(row.origin), int(row.dest))
        start = len(path_rows)
        try:
            gen = nx.shortest_simple_paths(G, o, d, weight='weight')
            first_cost = None
            seen = set()
            kept = 0
            for node_path in gen:
                edge_ids0 = []
                cost = 0.0
                length = 0.0
                for u, v in zip(node_path[:-1], node_path[1:]):
                    ed = G[u][v]
                    edge_ids0.append(int(ed['link_id0']))
                    cost += float(ed['weight'])
                    length += float(ed['length'])
                if first_cost is None:
                    first_cost = cost
                if cost > ratio_max * first_cost + 1e-12:
                    break
                key = tuple(edge_ids0)
                if key in seen:
                    continue
                seen.add(key)
                path_rows.append({'path_index': len(path_rows), 'od_index': int(od_idx), 'origin': o, 'dest': d, 'demand': float(row.demand), 'demand_raw': float(row.demand_raw), 'nodes': tuple(node_path), 'links0': tuple(edge_ids0), 'links_1based': tuple((int(net_df.loc[a, 'link_id']) for a in edge_ids0)), 'ff_time': cost, 'length': length, 'n_links': len(edge_ids0)})
                kept += 1
                if kept >= k_paths:
                    break
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            skipped.append((o, d))
        end = len(path_rows)
        if end == start:
            skipped.append((o, d))
            paths_by_od.append(np.array([], dtype=int))
        else:
            paths_by_od.append(np.arange(start, end, dtype=int))
    return (pd.DataFrame(path_rows), paths_by_od, skipped)
t_path = time.perf_counter()
paths, paths_by_od, skipped_ods = generate_candidate_paths(net, trips, K_PATHS, PATH_RATIO_MAX)
path_gen_seconds = time.perf_counter() - t_path
n_paths = len(paths)
n_links = len(net)
n_ods = len(trips)
total_demand = float(trips['demand'].sum())
q = trips['demand'].to_numpy(float)
row_idx, col_idx, data = ([], [], [])
for p_idx, link_ids0 in enumerate(paths['links0']):
    for a0 in link_ids0:
        row_idx.append(p_idx)
        col_idx.append(a0)
        data.append(1.0)
incidence = sparse.csr_matrix((data, (row_idx, col_idx)), shape=(n_paths, n_links))
path_counts = paths.groupby('od_index').size()
path_stats = pd.DataFrame({'metric': ['positive OD pairs in TNTP', 'positive OD pairs retained', 'candidate paths', 'average paths per OD', 'minimum paths per OD', 'maximum paths per OD', 'K_PATHS requested', 'PATH_RATIO_MAX', 'path generation CPU seconds'], 'value': [len(trips), len(path_counts), n_paths, float(path_counts.mean()), int(path_counts.min()), int(path_counts.max()), K_PATHS, PATH_RATIO_MAX, path_gen_seconds]})
print(f'Candidate paths: {n_paths:,}')
print(f'Average paths per positive OD: {path_counts.mean():.2f}')
print(f'Skipped OD pairs: {len(skipped_ods)}')
display(path_stats)
path_stats.to_csv(TABLE_DIR / 'path_generation_summary.csv', index=False)
paths.to_csv(TABLE_DIR / 'candidate_paths_full_od.csv', index=False)
xmid = net['xm'].to_numpy(float)
ymid = net['ym'].to_numpy(float)
xn = (xmid - xmid.min()) / (xmid.max() - xmid.min())
yn = (ymid - ymid.min()) / (ymid.max() - ymid.min())
h_incident = np.exp(-((xn - 0.6) ** 2 / (2 * 0.16 ** 2) + (yn - 0.52) ** 2 / (2 * 0.16 ** 2)))
h_incident = h_incident / max(float(h_incident.max()), 1e-12)

def band_field(center: float, sigma: float=0.16, floor: float=0.55) -> np.ndarray:
    h = np.exp(-(yn - center) ** 2 / (2 * sigma ** 2))
    h = h / max(float(h.max()), 1e-12)
    return np.clip(floor + (1.0 - floor) * h, 0.0, 1.0)
h_upper = band_field(0.82)
h_middle = band_field(0.52)
h_lower = band_field(0.18)
base_t0 = net['free_time'].to_numpy(dtype=float)
base_cap = net['capacity'].to_numpy(dtype=float)
base_beta = net['B'].to_numpy(dtype=float)
base_power = net['power'].to_numpy(dtype=float)
scenario_specs = [{'name': 'normal', 'prob': 0.55, 'type': 'baseline', 'severity': np.zeros(n_links)}, {'name': 'light_rain', 'prob': 0.2, 'type': 'rain', 'severity': 0.22 + 0.08 * yn}, {'name': 'minor_incident', 'prob': 0.15, 'type': 'incident', 'severity': h_incident}, {'name': 'flood_upper', 'prob': 0.04, 'type': 'flood', 'severity': h_upper}, {'name': 'flood_middle', 'prob': 0.035, 'type': 'flood', 'severity': h_middle}, {'name': 'flood_lower', 'prob': 0.025, 'type': 'flood', 'severity': h_lower}]
scenarios: List[dict] = []
link_change_rows: List[dict] = []
for spec in scenario_specs:
    name = spec['name']
    typ = spec['type']
    sev = np.asarray(spec['severity'], dtype=float)
    if typ == 'baseline':
        cap_mult = np.ones(n_links)
        t0_mult = np.ones(n_links)
        beta_mult = np.ones(n_links)
        delay = np.zeros(n_links)
    elif typ == 'rain':
        cap_mult = np.clip(1.0 - 0.1 * sev, 0.8, 1.0)
        t0_mult = 1.0 + 0.1 * sev
        beta_mult = 1.0 + 0.15 * sev
        delay = 0.4 + 1.1 * sev
    elif typ == 'incident':
        cap_mult = np.clip(1.0 - 0.28 * sev, 0.65, 1.0)
        t0_mult = 1.0 + 0.12 * sev
        beta_mult = 1.0 + 0.4 * sev
        delay = 0.2 + 4.5 * sev
    elif typ == 'flood':
        cap_mult = np.clip(1.0 - 0.82 * sev, 0.18, 0.58)
        t0_mult = 1.0 + 0.72 * sev
        beta_mult = 1.0 + 1.45 * sev
        delay = 6.0 + 24.0 * sev
    else:
        raise ValueError(typ)
    cap_s = np.maximum(0.05, base_cap * cap_mult)
    t0_s = base_t0 * t0_mult
    beta_s = base_beta * beta_mult
    power_s = base_power.copy()
    scenarios.append({'name': name, 'prob': float(spec['prob']), 'type': typ, 'severity': sev, 'cap': cap_s, 't0': t0_s, 'beta': beta_s, 'power': power_s, 'delay': delay, 'cap_mult': cap_mult, 't0_mult': t0_mult, 'beta_mult': beta_mult})
    for a0 in range(n_links):
        ref_x = float(net.loc[a0, 'ref_volume'])
        base_time_at_ref = base_t0[a0] * (1.0 + base_beta[a0] * (ref_x / base_cap[a0]) ** base_power[a0])
        scen_time_at_ref = t0_s[a0] * (1.0 + beta_s[a0] * (ref_x / cap_s[a0]) ** power_s[a0]) + delay[a0]
        link_change_rows.append({'scenario': name, 'scenario_type': typ, 'probability': float(spec['prob']), 'link_id': int(net.loc[a0, 'link_id']), 'init': int(net.loc[a0, 'init']), 'term': int(net.loc[a0, 'term']), 'length': float(net.loc[a0, 'length']), 'severity': float(sev[a0]), 'base_capacity_veh_h': float(net.loc[a0, 'capacity_raw']), 'scenario_capacity_veh_h': float(cap_s[a0] * FLOW_SCALE), 'capacity_multiplier': float(cap_mult[a0]), 'capacity_change_pct': float(100.0 * (cap_mult[a0] - 1.0)), 'base_free_time': float(base_t0[a0]), 'scenario_free_time': float(t0_s[a0]), 'free_time_multiplier': float(t0_mult[a0]), 'free_time_change_pct': float(100.0 * (t0_mult[a0] - 1.0)), 'base_B': float(base_beta[a0]), 'scenario_B': float(beta_s[a0]), 'B_multiplier': float(beta_mult[a0]), 'B_change_pct': float(100.0 * (beta_mult[a0] - 1.0)), 'power': float(power_s[a0]), 'additive_delay': float(delay[a0]), 'reference_volume_veh_h': float(ref_x * FLOW_SCALE), 'base_time_at_reference_flow': float(base_time_at_ref), 'scenario_time_at_reference_flow': float(scen_time_at_ref), 'time_at_reference_flow_change_pct': float(100.0 * (scen_time_at_ref / base_time_at_ref - 1.0)) if base_time_at_ref > 0 else np.nan})
scenario_probs = np.array([s['prob'] for s in scenarios], dtype=float)
scenario_probs = scenario_probs / scenario_probs.sum()
scenario_names = [s['name'] for s in scenarios]
n_scenarios = len(scenarios)
scenario_df = pd.DataFrame({'scenario': scenario_names, 'probability': scenario_probs, 'type': [s['type'] for s in scenarios]})
link_change_table = pd.DataFrame(link_change_rows)
scenario_df.to_csv(TABLE_DIR / 'scenario_probabilities.csv', index=False)
link_change_table.to_csv(TABLE_DIR / 'edge_scenario_changes_all_links_all_scenarios_long.csv', index=False)
wide_cols = ['capacity_change_pct', 'free_time_change_pct', 'B_change_pct', 'additive_delay', 'time_at_reference_flow_change_pct']
link_change_wide = link_change_table.pivot_table(index=['link_id', 'init', 'term', 'length', 'base_capacity_veh_h', 'base_free_time', 'reference_volume_veh_h'], columns='scenario', values=wide_cols, aggfunc='first')
link_change_wide.columns = [f'{metric}__{scenario}' for metric, scenario in link_change_wide.columns]
link_change_wide = link_change_wide.reset_index()
link_change_wide.to_csv(TABLE_DIR / 'edge_scenario_changes_all_links_all_scenarios_wide.csv', index=False)
print('Scenario probabilities:')
display(scenario_df)
print('Flood probability mass:', scenario_df.loc[scenario_df['type'].eq('flood'), 'probability'].sum())
print('Edge change table preview:')
display(link_change_table.head(10))
for s in scenarios:
    scen_tbl = link_change_table[link_change_table['scenario'].eq(s['name'])].sort_values('link_id')
    plot_sioux_reference_style(values_for_width=reference_plot_widths(), values_for_color=scen_tbl['time_at_reference_flow_change_pct'].to_numpy(float), labels=reference_plot_labels(), title=f"Scenario {s['name']}: " + ('link id / reference flow; color = time-change %' if HAS_REFERENCE_FLOW else 'link id; color = time-change at capacity %'), out_file=FIG_DIR / f"scenario_{s['name']}_reference_style_id_ref_flow_600dpi.png", colorbar_label='time at reference flow change (%)', cmap='Reds', black_edges=False)
print('Saved 600dpi reference-style scenario network plots to:', FIG_DIR)

def compute_link_flows(f: np.ndarray) -> np.ndarray:
    return np.asarray(incidence.T @ f).ravel()

def evaluate_times(f: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = compute_link_flows(f)
    link_times = np.empty((n_scenarios, n_links), dtype=float)
    potentials = np.empty(n_scenarios, dtype=float)
    for s_idx, s in enumerate(scenarios):
        cap = s['cap']
        t0 = s['t0']
        beta = s['beta']
        power = s['power']
        delay = s['delay']
        ratio = x / cap
        link_times[s_idx] = t0 * (1.0 + beta * ratio ** power) + delay
        potentials[s_idx] = np.sum(t0 * (x + beta / (power + 1.0) * x ** (power + 1.0) / cap ** power) + delay * x)
    path_times = np.asarray(link_times @ incidence.T)
    return (x, link_times, path_times, potentials)

def cvar_selector(values: np.ndarray, probs: np.ndarray, alpha: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    probs = np.asarray(probs, dtype=float)
    order = np.argsort(values, kind='mergesort')
    cum = np.cumsum(probs[order])
    m = int(np.searchsorted(cum, alpha, side='left'))
    if m >= len(values):
        m = len(values) - 1
    chi = np.zeros_like(values)
    if m + 1 < len(values):
        chi[order[m + 1:]] = 1.0
    frac = (cum[m] - alpha) / probs[order[m]]
    chi[order[m]] = np.clip(frac, 0.0, 1.0)
    return chi

def cvar_selectors_matrix(path_times_sp: np.ndarray, probs: np.ndarray, alpha: float) -> np.ndarray:
    vals = np.asarray(path_times_sp, dtype=float)
    S, P = vals.shape
    order = np.argsort(vals, axis=0, kind='mergesort')
    sorted_probs = probs[order]
    cum = np.cumsum(sorted_probs, axis=0)
    m = np.argmax(cum >= alpha, axis=0)
    chi_sorted = np.zeros((S, P), dtype=float)
    row_idx = np.arange(S)[:, None]
    chi_sorted[row_idx > m[None, :]] = 1.0
    cols = np.arange(P)
    frac = (cum[m, cols] - alpha) / sorted_probs[m, cols]
    chi_sorted[m, cols] = np.clip(frac, 0.0, 1.0)
    chi = np.zeros((S, P), dtype=float)
    chi[order, cols] = chi_sorted
    return chi.T

def modelB_costs(f: np.ndarray, alpha: float, eta: float) -> Tuple[np.ndarray, dict]:
    x, link_times, path_times, potentials = evaluate_times(f)
    chi_B = cvar_selector(potentials, scenario_probs, alpha)
    ptilde_B = scenario_probs * (1.0 - eta + eta / (1.0 - alpha) * chi_B)
    costs = np.asarray(ptilde_B @ path_times).ravel()
    return (costs, {'x': x, 'link_times': link_times, 'path_times': path_times, 'potentials': potentials, 'chi_B': chi_B, 'pure_tail_B': scenario_probs * chi_B / (1.0 - alpha), 'ptilde_B': ptilde_B})

def modelA_costs(f: np.ndarray, alpha: float, eta: float) -> Tuple[np.ndarray, dict]:
    x, link_times, path_times, potentials = evaluate_times(f)
    chi_A = cvar_selectors_matrix(path_times, scenario_probs, alpha)
    ptilde_A = scenario_probs[None, :] * (1.0 - eta + eta / (1.0 - alpha) * chi_A)
    costs = np.sum(ptilde_A * path_times.T, axis=1)
    return (costs, {'x': x, 'link_times': link_times, 'path_times': path_times, 'potentials': potentials, 'chi_A': chi_A, 'pure_tail_A': chi_A * scenario_probs[None, :] / (1.0 - alpha), 'ptilde_A': ptilde_A})

def uniform_flow() -> np.ndarray:
    f = np.zeros(n_paths, dtype=float)
    for od_idx, idx in enumerate(paths_by_od):
        if len(idx) == 0:
            continue
        f[idx] = q[od_idx] / len(idx)
    return f

def assign_truncated(costs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    f = np.zeros_like(costs, dtype=float)
    pis = np.zeros(n_ods, dtype=float)
    for od_idx, idx in enumerate(paths_by_od):
        if len(idx) == 0:
            continue
        demand = float(q[od_idx])
        c = np.asarray(costs[idx], dtype=float)
        order = np.argsort(c, kind='mergesort')
        cs = c[order]
        cmin = float(cs[0])
        w = np.exp(-THETA * (cs - cmin))
        cw = np.cumsum(w)
        pi = None
        mstar = len(c)
        for m in range(1, len(c) + 1):
            pi_m = cmin + 1.0 / THETA * math.log((demand / R0 + m) / cw[m - 1])
            lower_ok = pi_m > cs[m - 1] - 1e-12
            upper_ok = m == len(c) or pi_m <= cs[m] + 1e-12
            if lower_ok and upper_ok:
                pi = pi_m
                mstar = m
                break
        if pi is None:
            pi = cmin + 1.0 / THETA * math.log((demand / R0 + len(c)) / cw[-1])
            mstar = len(c)
        vals = np.zeros(len(c), dtype=float)
        active = order[:mstar]
        vals[active] = R0 * (np.exp(np.clip(THETA * (pi - c[active]), -60, 60)) - 1.0)
        vals = np.maximum(vals, 0.0)
        if vals.sum() <= 0 or not np.isfinite(vals.sum()):
            vals[:] = demand / len(c)
        else:
            vals *= demand / vals.sum()
        f[idx] = vals
        pis[od_idx] = pi
    return (f, pis)

def vi_gap(f: np.ndarray, costs: np.ndarray) -> Tuple[float, float]:
    F = costs + 1.0 / THETA * np.log1p(f / R0)
    numerator = 0.0
    for od_idx, idx in enumerate(paths_by_od):
        if len(idx) == 0:
            continue
        demand = float(q[od_idx])
        Fo = F[idx]
        fo = f[idx]
        numerator += float(np.dot(Fo, fo) - demand * Fo.min())
    denominator = 1.0 + abs(float(np.dot(F, f)))
    return (max(0.0, numerator), max(0.0, numerator) / denominator)

def solve_tsue(model: str, alpha: float, eta: float, f0: np.ndarray | None=None, max_iter: int=500, tol_rel_gap: float=0.002, verbose_every: int=0) -> dict:
    model = model.upper()
    if model not in {'A', 'B'}:
        raise ValueError("model must be 'A' or 'B'")
    cost_func = modelA_costs if model == 'A' else modelB_costs
    f = uniform_flow() if f0 is None else np.asarray(f0, dtype=float).copy()
    history: List[dict] = []
    start_cpu = time.process_time()
    start_wall = time.perf_counter()
    costs = None
    info = None
    pis = None
    for it in range(1, max_iter + 1):
        costs, info = cost_func(f, alpha, eta)
        target, pis = assign_truncated(costs)
        step = 1.0 if it == 1 and f0 is None else min(0.5, 2.0 / (it + 2.0) ** 0.62)
        f = (1.0 - step) * f + step * target
        for od_idx, idx in enumerate(paths_by_od):
            if len(idx) == 0:
                continue
            sm = float(f[idx].sum())
            if sm > 0:
                f[idx] *= q[od_idx] / sm
        costs, info = cost_func(f, alpha, eta)
        abs_gap, rel_gap = vi_gap(f, costs)
        history.append({'iter': it, 'abs_gap': abs_gap, 'rel_gap': rel_gap, 'step': step})
        if verbose_every and (it == 1 or it % verbose_every == 0):
            print(f'Model {model}, alpha={alpha:.2f}, eta={eta:.2f}, iter={it:4d}, rel_gap={rel_gap:.3e}')
        if rel_gap <= tol_rel_gap:
            break
    cpu_seconds = time.process_time() - start_cpu
    wall_seconds = time.perf_counter() - start_wall
    costs, info = cost_func(f, alpha, eta)
    abs_gap, rel_gap = vi_gap(f, costs)
    target, pis = assign_truncated(costs)
    fixed_point_l1 = float(np.linalg.norm(target - f, 1) / max(total_demand, 1e-12))
    converged = bool(rel_gap <= tol_rel_gap)
    return {'model': model, 'alpha': alpha, 'eta': eta, 'f': f, 'x': compute_link_flows(f), 'costs': costs, 'pis': pis, 'info': info, 'history': pd.DataFrame(history), 'iterations': len(history), 'cpu_seconds': float(cpu_seconds), 'wall_seconds': float(wall_seconds), 'abs_gap': float(abs_gap), 'rel_gap': float(rel_gap), 'fixed_point_l1_per_total_demand': fixed_point_l1, 'converged': converged}

def assert_converged(sol: dict, tol: float):
    if STRICT_CONVERGENCE and (not sol['converged']):
        eta_txt = f", eta={sol.get('eta', float('nan')):.3f}" if 'eta' in sol else ''
        msg = f"Model {sol['model']} alpha={sol.get('alpha', float('nan')):.3f}{eta_txt} did not converge to tol={tol}. Final rel_gap={sol['rel_gap']:.3e}, iterations={sol['iterations']}. Increase MAX_ITER, use a smoother continuation scheme, or do not report this row as solved."
        if STOP_ON_NONCONVERGENCE:
            raise RuntimeError(msg)
        print('WARNING:', msg)

def _build_od_equality_matrix() -> sparse.csr_matrix:
    rows, cols, data = ([], [], [])
    for od_idx, idx in enumerate(paths_by_od):
        for pidx in idx:
            rows.append(od_idx)
            cols.append(int(pidx))
            data.append(1.0)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n_ods, n_paths))
OD_EQ = _build_od_equality_matrix()

def _potentials_and_link_times_from_x(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    link_times = np.empty((n_scenarios, n_links), dtype=float)
    potentials = np.empty(n_scenarios, dtype=float)
    for s_idx, s in enumerate(scenarios):
        cap = s['cap']
        t0 = s['t0']
        beta = s['beta']
        power = s['power']
        delay = s['delay']
        ratio = x / cap
        link_times[s_idx] = t0 * (1.0 + beta * ratio ** power) + delay
        potentials[s_idx] = np.sum(t0 * (x + beta / (power + 1.0) * x ** (power + 1.0) / cap ** power) + delay * x)
    return (potentials, link_times)

def _ru_initial_gamma_u(f0: np.ndarray, alpha: float) -> Tuple[float, np.ndarray]:
    _, _, _, potentials = evaluate_times(f0)
    order = np.argsort(potentials, kind='mergesort')
    cum = np.cumsum(scenario_probs[order])
    m = int(np.searchsorted(cum, alpha, side='left'))
    m = min(m, len(potentials) - 1)
    gamma0 = float(max(0.0, potentials[order[m]]))
    u0 = np.maximum(potentials - gamma0, 0.0) + 1e-06
    return (gamma0, u0)

def _link_time_derivatives_from_x(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    deriv = np.empty((n_scenarios, n_links), dtype=float)
    for s_idx, s in enumerate(scenarios):
        cap = s['cap']
        t0 = s['t0']
        beta = s['beta']
        power = s['power']
        deriv[s_idx] = t0 * beta * power * np.maximum(x, 0.0) ** (power - 1.0) / cap ** power
    return deriv

def _full_sparse_hessian_from_f_block(Hf: sparse.spmatrix) -> sparse.csr_matrix:
    Hf = Hf.tocsr()
    z1 = sparse.csr_matrix((n_paths, 1 + n_scenarios))
    z2 = sparse.csr_matrix((1 + n_scenarios, n_paths))
    z3 = sparse.csr_matrix((1 + n_scenarios, 1 + n_scenarios))
    return sparse.bmat([[Hf, z1], [z2, z3]], format='csr')

def solve_modelB_ru_convex(alpha: float, eta: float, f0: np.ndarray | None=None, maxiter: int=RU_MODELB_MAXITER) -> dict:
    if eta <= 1e-14:
        sol = solve_tsue('B', alpha, eta, f0=f0, max_iter=MAX_ITER_B, tol_rel_gap=TOL_REL_GAP_B, verbose_every=0)
        sol['solver'] = 'fixed_point_MSA_eta0'
        return sol
    start_cpu = time.process_time()
    start_wall = time.perf_counter()
    f_start = uniform_flow() if f0 is None else np.asarray(f0, dtype=float).copy()
    for od_idx, idx in enumerate(paths_by_od):
        if len(idx) == 0:
            continue
        sm = float(f_start[idx].sum())
        if sm > 0:
            f_start[idx] *= q[od_idx] / sm
        else:
            f_start[idx] = q[od_idx] / len(idx)
    gamma0, u0 = _ru_initial_gamma_u(f_start, alpha)
    L_SCALE = max(float(np.max(evaluate_times(f_start)[3])), 1.0)
    z0 = np.concatenate([f_start, np.array([gamma0 / L_SCALE], dtype=float), u0 / L_SCALE])
    nvar = n_paths + 1 + n_scenarios
    lb = np.concatenate([np.zeros(n_paths), np.array([0.0]), np.zeros(n_scenarios)])
    ub = np.full(nvar, np.inf)
    bounds = Bounds(lb, ub)
    Aeq = sparse.hstack([OD_EQ, sparse.csr_matrix((n_ods, 1 + n_scenarios))], format='csr')
    lin_con = LinearConstraint(Aeq, q, q)

    def unpack(z: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        return (z[:n_paths], float(z[n_paths]), z[n_paths + 1:])

    def obj(z: np.ndarray) -> float:
        f, gamma_bar, u_bar = unpack(z)
        x = compute_link_flows(f)
        potentials, _ = _potentials_and_link_times_from_x(x)
        entropy = np.sum((f + R0) * np.log((f + R0) / R0) - f)
        return float((1.0 - eta) * np.dot(scenario_probs, potentials) / L_SCALE + eta * gamma_bar + eta / (1.0 - alpha) * np.dot(scenario_probs, u_bar) + 1.0 / (THETA * L_SCALE) * entropy)

    def grad(z: np.ndarray) -> np.ndarray:
        f, gamma_bar, u_bar = unpack(z)
        x = compute_link_flows(f)
        _, link_times = _potentials_and_link_times_from_x(x)
        weighted_link_times = (1.0 - eta) * (scenario_probs @ link_times)
        grad_f = (np.asarray(incidence @ weighted_link_times).ravel() + 1.0 / THETA * np.log1p(f / R0)) / L_SCALE
        g = np.zeros(nvar, dtype=float)
        g[:n_paths] = grad_f
        g[n_paths] = eta
        g[n_paths + 1:] = eta / (1.0 - alpha) * scenario_probs
        return g

    def hess_obj(z: np.ndarray) -> sparse.csr_matrix:
        f, _, _ = unpack(z)
        x = compute_link_flows(f)
        link_deriv = _link_time_derivatives_from_x(x)
        diag_link = (1.0 - eta) * (scenario_probs @ link_deriv) / L_SCALE
        Hf = incidence @ sparse.diags(diag_link, 0, shape=(n_links, n_links)) @ incidence.T
        Hf = Hf + sparse.diags(1.0 / THETA / (R0 + np.maximum(f, 0.0)) / L_SCALE, 0, shape=(n_paths, n_paths))
        return _full_sparse_hessian_from_f_block(Hf)

    def ru_con_fun(z: np.ndarray) -> np.ndarray:
        f, gamma_bar, u_bar = unpack(z)
        x = compute_link_flows(f)
        potentials, _ = _potentials_and_link_times_from_x(x)
        return potentials / L_SCALE - gamma_bar - u_bar

    def ru_con_jac(z: np.ndarray) -> sparse.csr_matrix:
        f, gamma, u = unpack(z)
        x = compute_link_flows(f)
        _, link_times = _potentials_and_link_times_from_x(x)
        path_times = sparse.csr_matrix(link_times @ incidence.T / L_SCALE)
        Jgamma = -np.ones((n_scenarios, 1))
        Ju = -np.eye(n_scenarios)
        return sparse.hstack([path_times, sparse.csr_matrix(Jgamma), sparse.csr_matrix(Ju)], format='csr')

    def ru_con_hess(z: np.ndarray, v: np.ndarray) -> sparse.csr_matrix:
        f, _, _ = unpack(z)
        x = compute_link_flows(f)
        link_deriv = _link_time_derivatives_from_x(x)
        diag_link = np.asarray(v, dtype=float) @ link_deriv / L_SCALE
        Hf = incidence @ sparse.diags(diag_link, 0, shape=(n_links, n_links)) @ incidence.T
        return _full_sparse_hessian_from_f_block(Hf)
    nlc = NonlinearConstraint(ru_con_fun, -np.inf * np.ones(n_scenarios), np.zeros(n_scenarios), jac=ru_con_jac, hess=ru_con_hess)
    res = minimize(obj, z0, method='trust-constr', jac=grad, hess=hess_obj, bounds=bounds, constraints=[lin_con, nlc], options={'maxiter': int(maxiter), 'gtol': MODEL_B_RU_GTOL, 'xtol': MODEL_B_RU_XTOL, 'barrier_tol': MODEL_B_RU_BARRIER_TOL, 'verbose': int(MODEL_B_RU_VERBOSE), 'sparse_jacobian': True})
    z = np.asarray(res.x, dtype=float)
    f, gamma_bar, u_bar = unpack(z)
    gamma = gamma_bar * L_SCALE
    u = u_bar * L_SCALE
    f = np.maximum(f, 0.0)
    for od_idx, idx in enumerate(paths_by_od):
        if len(idx) == 0:
            continue
        sm = float(f[idx].sum())
        if sm > 0:
            f[idx] *= q[od_idx] / sm
    costs, info = modelB_costs(f, alpha, eta)
    abs_gap, rel_gap = vi_gap(f, costs)
    target, pis = assign_truncated(costs)
    fixed_point_l1 = float(np.linalg.norm(target - f, 1) / max(total_demand, 1e-12))
    cpu_seconds = time.process_time() - start_cpu
    wall_seconds = time.perf_counter() - start_wall
    objective_value = obj(np.concatenate([f, np.array([max(gamma / L_SCALE, 0.0)]), np.maximum(u / L_SCALE, 0.0)]))
    converged = bool(res.success)
    hist = pd.DataFrame([{'iter': int(getattr(res, 'nit', -1)), 'objective': float(objective_value), 'abs_gap': abs_gap, 'rel_gap': rel_gap, 'optimality': float(getattr(res, 'optimality', np.nan)), 'constraint_violation': float(getattr(res, 'constr_violation', np.nan)), 'success': bool(res.success), 'message': str(res.message)}])
    return {'model': 'B', 'solver': 'convex_RU_trust_constr', 'alpha': alpha, 'eta': eta, 'f': f, 'x': compute_link_flows(f), 'costs': costs, 'pis': pis, 'info': info, 'gamma_RU': float(gamma), 'u_RU': np.maximum(u, 0.0), 'objective_RU_scaled': float(objective_value), 'RU_scale': float(L_SCALE), 'optimizer_success': bool(res.success), 'optimizer_status': int(res.status), 'optimizer_message': str(res.message), 'optimizer_optimality': float(getattr(res, 'optimality', np.nan)), 'optimizer_constraint_violation': float(getattr(res, 'constr_violation', np.nan)), 'history': hist, 'iterations': int(getattr(res, 'nit', -1)), 'cpu_seconds': float(cpu_seconds), 'wall_seconds': float(wall_seconds), 'abs_gap': float(abs_gap), 'rel_gap': float(rel_gap), 'fixed_point_l1_per_total_demand': fixed_point_l1, 'converged': converged}

def scenario_costs(f: np.ndarray, scenario_index: int) -> Tuple[np.ndarray, dict]:
    x, link_times, path_times, potentials = evaluate_times(f)
    costs = np.asarray(path_times[scenario_index]).ravel()
    return (costs, {'x': x, 'link_times': link_times, 'path_times': path_times, 'potentials': potentials, 'scenario_index': int(scenario_index), 'scenario_name': scenario_names[scenario_index]})

def solve_scenario_tsue(scenario_index: int, f0: np.ndarray | None=None, max_iter: int=500, tol_rel_gap: float=0.002, verbose_every: int=0) -> dict:
    f = uniform_flow() if f0 is None else np.asarray(f0, dtype=float).copy()
    history: List[dict] = []
    start_cpu = time.process_time()
    start_wall = time.perf_counter()
    for it in range(1, max_iter + 1):
        costs, info = scenario_costs(f, scenario_index)
        target, pis = assign_truncated(costs)
        step = 1.0 if it == 1 and f0 is None else min(0.5, 2.0 / (it + 2.0) ** 0.62)
        f = (1.0 - step) * f + step * target
        for od_idx, idx in enumerate(paths_by_od):
            if len(idx) == 0:
                continue
            sm = float(f[idx].sum())
            if sm > 0:
                f[idx] *= q[od_idx] / sm
        costs, info = scenario_costs(f, scenario_index)
        abs_gap, rel_gap = vi_gap(f, costs)
        history.append({'iter': it, 'abs_gap': abs_gap, 'rel_gap': rel_gap, 'step': step})
        if verbose_every and (it == 1 or it % verbose_every == 0):
            print(f'Scenario TSUE {scenario_names[scenario_index]}, iter={it:4d}, rel_gap={rel_gap:.3e}')
        if rel_gap <= tol_rel_gap:
            break
    cpu_seconds = time.process_time() - start_cpu
    wall_seconds = time.perf_counter() - start_wall
    costs, info = scenario_costs(f, scenario_index)
    abs_gap, rel_gap = vi_gap(f, costs)
    target, pis = assign_truncated(costs)
    fixed_point_l1 = float(np.linalg.norm(target - f, 1) / max(total_demand, 1e-12))
    return {'model': 'scenario_TSUE', 'scenario_index': int(scenario_index), 'scenario_name': scenario_names[scenario_index], 'scenario_probability': float(scenario_probs[scenario_index]), 'f': f, 'x': compute_link_flows(f), 'costs': costs, 'pis': pis, 'info': info, 'history': pd.DataFrame(history), 'iterations': len(history), 'cpu_seconds': float(cpu_seconds), 'wall_seconds': float(wall_seconds), 'abs_gap': float(abs_gap), 'rel_gap': float(rel_gap), 'fixed_point_l1_per_total_demand': fixed_point_l1, 'converged': bool(rel_gap <= tol_rel_gap)}
scenario_tsue_solutions: Dict[int, dict] = {}
if SOLVE_SCENARIO_TSUE:
    scenario_summary_rows: List[dict] = []
    scenario_link_rows: List[dict] = []
    scenario_path_rows: List[dict] = []
    for s_idx, s_name in enumerate(scenario_names):
        print(f'\nSolving scenario-conditioned TSUE for scenario={s_name}')
        solS = solve_scenario_tsue(s_idx, f0=None, max_iter=MAX_ITER_B, tol_rel_gap=TOL_REL_GAP_B, verbose_every=100)
        if STRICT_CONVERGENCE and (not solS['converged']):
            raise RuntimeError(f"Scenario TSUE for {s_name} did not converge to tol={TOL_REL_GAP_B}. Final rel_gap={solS['rel_gap']:.3e}, iterations={solS['iterations']}.")
        scenario_tsue_solutions[s_idx] = solS
        print(f"Scenario TSUE solved: scenario={s_name}, iter={solS['iterations']}, rel_gap={solS['rel_gap']:.3e}, CPU={solS['cpu_seconds']:.2f}s")
        scenario_summary_rows.append({'scenario_index': s_idx, 'scenario': s_name, 'scenario_probability': float(scenario_probs[s_idx]), 'iterations': solS['iterations'], 'cpu_seconds': solS['cpu_seconds'], 'wall_seconds': solS['wall_seconds'], 'rel_gap': solS['rel_gap'], 'fixed_point_l1_per_total_demand': solS['fixed_point_l1_per_total_demand'], 'converged': solS['converged']})
        for a0, r in net.iterrows():
            scenario_link_rows.append({'scenario_index': s_idx, 'scenario': s_name, 'scenario_probability': float(scenario_probs[s_idx]), 'link_id': int(r.link_id), 'init': int(r.init), 'term': int(r.term), 'scenario_TSUE_flow_veh_h': float(solS['x'][a0] * FLOW_SCALE), 'reference_flow_veh_h': float(r.ref_volume_raw)})
        for p0, r in paths.iterrows():
            scenario_path_rows.append({'scenario_index': s_idx, 'scenario': s_name, 'scenario_probability': float(scenario_probs[s_idx]), 'path_index': int(r.path_index), 'od_index': int(r.od_index), 'origin': int(r.origin), 'dest': int(r.dest), 'scenario_TSUE_path_flow_veh_h': float(solS['f'][p0] * FLOW_SCALE), 'scenario_path_cost': float(solS['costs'][p0]), 'links_1based': r.links_1based})
        tag_s = re.sub('[^A-Za-z0-9]+', '_', s_name).strip('_')
        solS['history'].to_csv(TABLE_DIR / f'history_scenario_TSUE_{tag_s}.csv', index=False)
        plot_sioux_reference_style(values_for_width=solS['x'], values_for_color=solS['x'] * FLOW_SCALE, labels=id_flow_labels(solS['x']), title=f'Scenario-conditioned TSUE link flows: {s_name}', out_file=FIG_DIR / f'scenario_TSUE_solved_link_flow_{tag_s}_600dpi.png', colorbar_label='Scenario-conditioned TSUE flow (veh/h)', cmap='Reds', black_edges=False)
    scenario_tsue_summary = pd.DataFrame(scenario_summary_rows)
    scenario_tsue_link_flows = pd.DataFrame(scenario_link_rows)
    scenario_tsue_path_flows = pd.DataFrame(scenario_path_rows)
    scenario_tsue_summary.to_csv(TABLE_DIR / 'scenario_TSUE_solver_summary.csv', index=False)
    scenario_tsue_link_flows.to_csv(TABLE_DIR / 'scenario_TSUE_link_flows_full_network.csv', index=False)
    scenario_tsue_path_flows.to_csv(TABLE_DIR / 'scenario_TSUE_path_flows_full_network.csv', index=False)
    print('\nScenario-conditioned TSUE summary:')
    display(scenario_tsue_summary)
modelB_solutions: Dict[Tuple[float, float], dict] = {}
modelA_solutions: Dict[Tuple[float, float, str], dict] = {}
summary_rows: List[dict] = []
for eta in ETA_GRID:
    for alpha in ALPHA_GRID:
        print()
        print(f'Solving Model B, alpha={alpha:.3f}, eta={eta:.2f}')
        f0_B = None
        previous_alphas = [a0 for a0, e0 in modelB_solutions.keys() if abs(e0 - eta) <= 1e-12 and a0 < alpha]
        if previous_alphas:
            f0_B = modelB_solutions[max(previous_alphas), eta]['f']
        if USE_CONVEX_RU_MODELB and eta > 1e-14 and (alpha >= RU_MODELB_ALPHA_MIN):
            solB = solve_modelB_ru_convex(alpha, eta, f0=f0_B, maxiter=RU_MODELB_MAXITER)
        else:
            solB = solve_tsue('B', alpha, eta, f0=f0_B, max_iter=MAX_ITER_B, tol_rel_gap=TOL_REL_GAP_B, verbose_every=100)
            solB['solver'] = 'fixed_point_MSA'
        assert_converged(solB, TOL_REL_GAP_B)
        modelB_solutions[alpha, eta] = solB
        summary_rows.append({'model': 'B', 'solver': solB.get('solver', ''), 'alpha': alpha, 'eta': eta, 'initialization': 'previous_alpha_warm_or_uniform', 'iterations': solB['iterations'], 'cpu_seconds': solB['cpu_seconds'], 'wall_seconds': solB['wall_seconds'], 'rel_gap': solB['rel_gap'], 'fixed_point_l1_per_total_demand': solB['fixed_point_l1_per_total_demand'], 'optimizer_success': solB.get('optimizer_success', np.nan), 'converged': solB['converged']})
        print(f'Solving Model A, alpha={alpha:.3f}, eta={eta:.2f}, initialization=uniform')
        solA_uniform = solve_tsue('A', alpha, eta, f0=uniform_flow(), max_iter=MAX_ITER_A, tol_rel_gap=TOL_REL_GAP_A, verbose_every=100)
        assert_converged(solA_uniform, TOL_REL_GAP_A)
        modelA_solutions[alpha, eta, 'uniform'] = solA_uniform
        summary_rows.append({'model': 'A', 'alpha': alpha, 'eta': eta, 'initialization': 'uniform', 'iterations': solA_uniform['iterations'], 'cpu_seconds': solA_uniform['cpu_seconds'], 'wall_seconds': solA_uniform['wall_seconds'], 'rel_gap': solA_uniform['rel_gap'], 'fixed_point_l1_per_total_demand': solA_uniform['fixed_point_l1_per_total_demand'], 'converged': solA_uniform['converged']})
        print(f'Solving Model A, alpha={alpha:.3f}, eta={eta:.2f}, initialization=ModelB_warm_start')
        solA_warm = solve_tsue('A', alpha, eta, f0=solB['f'], max_iter=MAX_ITER_A, tol_rel_gap=TOL_REL_GAP_A, verbose_every=100)
        assert_converged(solA_warm, TOL_REL_GAP_A)
        modelA_solutions[alpha, eta, 'ModelB_warm_start'] = solA_warm
        summary_rows.append({'model': 'A', 'alpha': alpha, 'eta': eta, 'initialization': 'ModelB_warm_start', 'iterations': solA_warm['iterations'], 'cpu_seconds': solA_warm['cpu_seconds'], 'wall_seconds': solA_warm['wall_seconds'], 'rel_gap': solA_warm['rel_gap'], 'fixed_point_l1_per_total_demand': solA_warm['fixed_point_l1_per_total_demand'], 'converged': solA_warm['converged']})
solver_summary = pd.DataFrame(summary_rows)
display(solver_summary)
solver_summary.to_csv(TABLE_DIR / 'solver_iterations_cpu_walltime_full_od.csv', index=False)
modelA_comparison_solutions: Dict[Tuple[float, float], dict] = {}
common_tail_certificate_rows: List[dict] = []
for eta in ETA_GRID:
    for alpha in ALPHA_GRID:
        solB = modelB_solutions[alpha, eta]
        cA_at_B, infoA_at_B = modelA_costs(solB['f'], alpha, eta)
        cB_at_B, infoB_at_B = modelB_costs(solB['f'], alpha, eta)
        _, gapA_fB = vi_gap(solB['f'], cA_at_B)
        pureB = infoB_at_B['pure_tail_B']
        ptildeB = infoB_at_B['ptilde_B']
        pureA = infoA_at_B['pure_tail_A']
        ptildeA = infoA_at_B['ptilde_A']
        D_tail_cert = float(np.sum(np.abs(pureA - pureB[None, :]), axis=1).max())
        D_p_cert = float(np.sum(np.abs(ptildeA - ptildeB[None, :]), axis=1).max())
        max_cost_diff_cert = float(np.max(np.abs(cA_at_B - cB_at_B)))
        in_certified_alpha_list = any((abs(alpha - a0) <= 1e-12 for a0 in COMMON_TAIL_CERTIFY_ALPHA_VALUES))
        certified_equal = bool(USE_COMMON_TAIL_CERTIFIED_EQUALITY and in_certified_alpha_list and (D_tail_cert <= COMMON_TAIL_CERT_TOL) and (D_p_cert <= COMMON_TAIL_CERT_TOL) and (max_cost_diff_cert <= max(1e-07, COMMON_TAIL_CERT_TOL)))
        if certified_equal:
            solA_cmp = dict(solB)
            solA_cmp['model'] = 'A_common_tail_certified_equal_to_B'
            solA_cmp['f'] = solB['f'].copy()
            solA_cmp['x'] = solB['x'].copy()
            solA_cmp['costs'] = cA_at_B.copy()
            solA_cmp['info'] = infoA_at_B
            solA_cmp['common_tail_certified_equal_to_ModelB'] = True
        else:
            solA_cmp = modelA_solutions[alpha, eta, 'ModelB_warm_start']
            solA_cmp['common_tail_certified_equal_to_ModelB'] = False
        modelA_comparison_solutions[alpha, eta] = solA_cmp
        common_tail_certificate_rows.append({'alpha': alpha, 'eta': eta, 'certified_equal_for_comparison': certified_equal, 'D_tail_AB_at_fB': D_tail_cert, 'D_p_AB_at_fB': D_p_cert, 'Gap_A_fB': float(gapA_fB), 'max_abs_ModelA_minus_ModelB_cost_at_fB': max_cost_diff_cert, 'comparison_solution_source': 'ModelB_by_common_tail_certificate' if certified_equal else 'independent_ModelA_ModelB_warm_start'})
common_tail_certificates = pd.DataFrame(common_tail_certificate_rows)
common_tail_certificates.to_csv(TABLE_DIR / 'common_tail_equality_certificates.csv', index=False)
display(common_tail_certificates)

def modelAB_diagnostics(alpha: float, eta: float, solB: dict, solA: dict, solA_warm_independent: dict | None=None) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    fB = solB['f']
    cA_at_B, infoA_at_B = modelA_costs(fB, alpha, eta)
    cB_at_B, infoB_at_B = modelB_costs(fB, alpha, eta)
    _, gapA_fB = vi_gap(fB, cA_at_B)
    pureB = infoB_at_B['pure_tail_B']
    ptildeB = infoB_at_B['ptilde_B']
    pureA = infoA_at_B['pure_tail_A']
    ptildeA = infoA_at_B['ptilde_A']
    tail_l1_by_path = np.sum(np.abs(pureA - pureB[None, :]), axis=1)
    tilted_l1_by_path = np.sum(np.abs(ptildeA - ptildeB[None, :]), axis=1)
    D_tail_AB = float(tail_l1_by_path.max())
    D_p_AB = float(tilted_l1_by_path.max())
    D_f_AB = float(np.linalg.norm(solA['f'] - solB['f'], 1) / max(total_demand, 1e-12))
    summary = {'alpha': alpha, 'eta': eta, 'D_tail_AB_at_fB': D_tail_AB, 'D_p_AB_at_fB': D_p_AB, 'D_f_AB_using_comparison_solution': D_f_AB, 'comparison_solution_source': solA.get('model', 'A'), 'Gap_A_fB': float(gapA_fB), 'share_paths_exact_tail_match': float(np.mean(tail_l1_by_path <= 1e-10)), 'ModelB_iterations': solB['iterations'], 'ModelB_cpu_seconds': solB['cpu_seconds'], 'ModelB_rel_gap': solB['rel_gap'], 'ModelA_comparison_iterations': solA['iterations'], 'ModelA_comparison_cpu_seconds': solA['cpu_seconds'], 'ModelA_comparison_rel_gap': solA['rel_gap']}
    if solA_warm_independent is not None:
        summary.update({'D_f_AB_using_independent_ModelA_warm_solution': float(np.linalg.norm(solA_warm_independent['f'] - solB['f'], 1) / max(total_demand, 1e-12)), 'ModelA_warm_iterations': solA_warm_independent['iterations'], 'ModelA_warm_cpu_seconds': solA_warm_independent['cpu_seconds'], 'ModelA_warm_rel_gap': solA_warm_independent['rel_gap']})
    solA_u = modelA_solutions[alpha, eta, 'uniform']
    summary.update({'ModelA_uniform_iterations': solA_u['iterations'], 'ModelA_uniform_cpu_seconds': solA_u['cpu_seconds'], 'ModelA_uniform_rel_gap': solA_u['rel_gap'], 'warm_start_iteration_speedup_uniform_over_Bwarm': solA_u['iterations'] / max((solA_warm_independent or solA)['iterations'], 1), 'warm_start_CPU_speedup_uniform_over_Bwarm': solA_u['cpu_seconds'] / max((solA_warm_independent or solA)['cpu_seconds'], 1e-12)})
    tail_law = pd.DataFrame({'scenario': scenario_names, 'probability': scenario_probs, 'ModelB_chi': infoB_at_B['chi_B'], 'ModelB_pure_tail_law': pureB, 'ModelB_tilted_law': ptildeB, 'potential_value_at_fB': infoB_at_B['potentials']})
    path_diag = paths[['path_index', 'od_index', 'origin', 'dest', 'ff_time', 'length', 'n_links', 'links_1based']].copy()
    path_diag['flow_ModelB'] = fB
    path_diag['flow_ModelA_comparison'] = solA['f']
    if solA_warm_independent is not None:
        path_diag['flow_ModelA_warm_independent'] = solA_warm_independent['f']
    path_diag['tail_l1_vs_ModelB'] = tail_l1_by_path
    path_diag['tilted_l1_vs_ModelB'] = tilted_l1_by_path
    path_diag['top_ModelA_pure_tail_scenario'] = [scenario_names[i] for i in np.argmax(pureA, axis=1)]
    path_diag['cost_A_at_fB'] = cA_at_B
    path_diag['cost_B_at_fB'] = cB_at_B
    for s_idx, name in enumerate(scenario_names):
        path_diag[f'A_pure_tail_{name}'] = pureA[:, s_idx]
        path_diag[f'A_tilted_{name}'] = ptildeA[:, s_idx]
    return (summary, tail_law, path_diag)
relation_rows = []
for eta in ETA_GRID:
    for alpha in ALPHA_GRID:
        solB = modelB_solutions[alpha, eta]
        solA_cmp = modelA_comparison_solutions[alpha, eta]
        solA_w = modelA_solutions[alpha, eta, 'ModelB_warm_start']
        summary, tail_law, path_diag = modelAB_diagnostics(alpha, eta, solB, solA_cmp, solA_warm_independent=solA_w)
        relation_rows.append(summary)
        tag = f'alpha_{alpha:.3f}_eta_{eta:.2f}'.replace('.', 'p')
        tail_law.to_csv(TABLE_DIR / f'modelB_tail_law_{tag}.csv', index=False)
        path_diag.to_csv(TABLE_DIR / f'path_level_modelAB_tail_diagnostics_{tag}.csv', index=False)
modelAB_relation = pd.DataFrame(relation_rows)
display(modelAB_relation)
modelAB_relation.to_csv(TABLE_DIR / 'modelAB_benchmark_relation_Df_Dp_Gap_full_network.csv', index=False)

def save_metric_heatmap(df: pd.DataFrame, value_col: str, title: str, outfile: str):
    pivot = df.pivot(index='eta', columns='alpha', values=value_col).sort_index().sort_index(axis=1)
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    im = ax.imshow(pivot.values, aspect='auto')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f'{a:.3f}' for a in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f'{e:.2f}' for e in pivot.index])
    ax.set_xlabel('alpha')
    ax.set_ylabel('eta')
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=7)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f'{val:.3g}', ha='center', va='center', fontsize=6, bbox=dict(facecolor='white', edgecolor='none', alpha=0.65, pad=0.12))
    fig.tight_layout()
    fig.savefig(FIG_DIR / outfile, dpi=DPI_FIG)
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)
save_metric_heatmap(modelAB_relation, 'D_p_AB_at_fB', 'Model A-B path-tail distance over (alpha, eta)', 'heatmap_modelAB_Dp_600dpi.png')
save_metric_heatmap(modelAB_relation, 'D_f_AB_using_comparison_solution', 'Model A-B flow distance over (alpha, eta)', 'heatmap_modelAB_Df_600dpi.png')
save_metric_heatmap(modelAB_relation, 'Gap_A_fB', 'Model A benchmark gap at the Model B flow over (alpha, eta)', 'heatmap_modelAB_GapA_600dpi.png')
save_metric_heatmap(modelAB_relation, 'warm_start_iteration_speedup_uniform_over_Bwarm', 'Warm-start iteration speedup over (alpha, eta)', 'heatmap_warmstart_speedup_600dpi.png')
for value_col, title, outname in [('iterations', 'Model B iterations over (alpha, eta)', 'heatmap_modelB_iterations_600dpi.png'), ('cpu_seconds', 'Model B CPU seconds over (alpha, eta)', 'heatmap_modelB_cpu_600dpi.png')]:
    tmp = solver_summary[(solver_summary['model'] == 'B') & (solver_summary['initialization'] == 'uniform')][['alpha', 'eta', value_col]].copy()
    save_metric_heatmap(tmp, value_col, title, outname)
for model_name, init_name, outname in [('B', 'uniform', 'heatmap_modelB_converged_600dpi.png'), ('A', 'uniform', 'heatmap_modelA_uniform_converged_600dpi.png'), ('A', 'ModelB_warm_start', 'heatmap_modelA_warm_converged_600dpi.png')]:
    tmp = solver_summary[(solver_summary['model'] == model_name) & (solver_summary['initialization'] == init_name)][['alpha', 'eta', 'converged']].copy()
    tmp['converged'] = tmp['converged'].astype(int)
    save_metric_heatmap(tmp, 'converged', f'{model_name} {init_name} convergence status', outname)
link_flow_rows = []
modelA_rows = []
modelB_rows = []
diff_rows = []
for eta in ETA_GRID:
    for alpha in ALPHA_GRID:
        solB = modelB_solutions[alpha, eta]
        solA = modelA_comparison_solutions[alpha, eta]
        for a0, r in net.iterrows():
            row_common = {'alpha': alpha, 'eta': eta, 'link_id': int(r.link_id), 'init': int(r.init), 'term': int(r.term), 'reference_flow_veh_h': float(r.ref_volume_raw)}
            flowB = float(solB['x'][a0] * FLOW_SCALE)
            flowA = float(solA['x'][a0] * FLOW_SCALE)
            diffAB = flowA - flowB
            link_flow_rows.append({**row_common, 'ModelB_flow_veh_h': flowB, 'ModelA_flow_veh_h': flowA, 'A_minus_B_flow_veh_h': diffAB})
            modelB_rows.append({**row_common, 'model': 'B', 'flow_veh_h': flowB})
            modelA_rows.append({**row_common, 'model': 'A', 'flow_veh_h': flowA})
            diff_rows.append({**row_common, 'A_minus_B_flow_veh_h': diffAB, 'abs_difference_veh_h': abs(diffAB)})
link_flow_table = pd.DataFrame(link_flow_rows)
pd.DataFrame(modelA_rows).to_csv(TABLE_DIR / 'link_flows_ModelA_full_network.csv', index=False)
pd.DataFrame(modelB_rows).to_csv(TABLE_DIR / 'link_flows_ModelB_full_network.csv', index=False)
pd.DataFrame(diff_rows).to_csv(TABLE_DIR / 'link_flow_differences_ModelA_minus_ModelB_full_network.csv', index=False)
link_flow_table.to_csv(TABLE_DIR / 'link_flows_ModelA_ModelB_full_network.csv', index=False)
display(link_flow_table.head())
for eta in ETA_GRID:
    for alpha in ALPHA_GRID:
        solB = modelB_solutions[alpha, eta]
        solA = modelA_comparison_solutions[alpha, eta]
        tag = f'alpha_{alpha:.3f}_eta_{eta:.2f}'.replace('.', 'p')
        if not PLOT_ONLY_CONVERGED_MODEL_FLOWS or solB.get('converged', False):
            plot_sioux_reference_style(values_for_width=solB['x'], values_for_color=solB['x'] * FLOW_SCALE, labels=id_flow_labels(solB['x']), title=f'Model B solved link flows: alpha={alpha:.3f}, eta={eta:.2f}' + ('' if solB.get('converged', False) else ' [NOT CONVERGED]'), out_file=FIG_DIR / f'modelB_solved_link_id_flow_{tag}_600dpi.png', colorbar_label='Model B flow (veh/h)', cmap='Reds', black_edges=False)
        if not PLOT_ONLY_CONVERGED_MODEL_FLOWS or solA.get('converged', False):
            plot_sioux_reference_style(values_for_width=solA['x'], values_for_color=solA['x'] * FLOW_SCALE, labels=id_flow_labels(solA['x']), title=f'Model A solved link flows: alpha={alpha:.3f}, eta={eta:.2f}' + ('' if solA.get('converged', False) else ' [NOT CONVERGED]'), out_file=FIG_DIR / f'modelA_solved_link_id_flow_{tag}_600dpi.png', colorbar_label='Model A flow (veh/h)', cmap='Reds', black_edges=False)
        pd.DataFrame({'path_index': paths['path_index'], 'origin': paths['origin'], 'dest': paths['dest'], 'nodes': paths['nodes'].astype(str), 'links_1based': paths['links_1based'].astype(str), 'flow_ModelB_veh_h': solB['f'] * FLOW_SCALE, 'flow_ModelA_comparison_veh_h': solA['f'] * FLOW_SCALE, 'cost_ModelB': solB['costs'], 'cost_ModelA': solA['costs']}).to_csv(TABLE_DIR / f'path_flows_ModelA_ModelB_{tag}.csv', index=False)
        solB['history'].to_csv(TABLE_DIR / f'history_ModelB_{tag}.csv', index=False)
        for init_name in ['uniform', 'ModelB_warm_start']:
            modelA_solutions[alpha, eta, init_name]['history'].to_csv(TABLE_DIR / f'history_ModelA_{init_name}_{tag}.csv', index=False)
print('Saved all 600dpi plots under:', FIG_DIR)
print('Saved all tables under:', TABLE_DIR)
zip_path = OUTPUT_BASE / 'sioux_falls_full_od_modelAB_reference_style_outputs.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for p in OUTPUT_DIR.rglob('*'):
        if p.is_file():
            zf.write(p, p.relative_to(OUTPUT_DIR.parent))
print('Output zip:', zip_path)
print('\nSanity checks:')
print('Full OD pairs used:', len(trips))
print('Links used:', len(net))
print('Candidate paths used:', len(paths))
print('All Model B converged:', all((sol['converged'] for sol in modelB_solutions.values())))
print('Nonconverged Model B cases:', [(a, e, sol['rel_gap']) for (a, e), sol in modelB_solutions.items() if not sol['converged']])
print('All Model A warm-start runs converged:', all((sol['converged'] for key, sol in modelA_solutions.items() if key[2] == 'ModelB_warm_start')))
print('All Model A uniform runs converged:', all((sol['converged'] for key, sol in modelA_solutions.items() if key[2] == 'uniform')))
display(solver_summary)
display(modelAB_relation)
