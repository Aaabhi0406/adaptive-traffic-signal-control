"""
dashboard_intersection.py — Vision-Based Adaptive Traffic Signal Control
                            (research-prototype dashboard, Streamlit)

PRESENTATION LAYER ONLY.

This file consumes the dict returned by FourWayIntersection.process_tick().
It does NOT perform detection, tracking, demand estimation, RF prediction,
uncertainty calculation, demand fusion, signal control or SUMO simulation.
The backend is the single source of truth for every number shown here.

Architecture this UI is built to communicate
--------------------------------------------
    4 independent camera approaches (N / E / S / W)
        -> YOLOv8 detection
        -> DeepSORT tracking
        -> class-weighted traffic demand
        -> Random-Forest short-horizon prediction
        -> prediction uncertainty (RF tree-spread interval)
        -> ONE intersection-level conflict-safe adaptive signal controller

Backend fields consumed (all read-only)
---------------------------------------
Per approach, tick["approach_results"][a]:
    annotated_frame, vehicle_count, traffic_level,
    pred_density, pred_lower, pred_upper, pred_level, pred_confidence
Intersection level:
    weighted_densities[a]                      current (weighted) demand
    signal_state                               NS_GREEN / NS_YELLOW / ALL_RED /
                                               EW_GREEN / EW_YELLOW
    signal_states[a]                           GREEN / YELLOW / RED per approach
    active_group, green_elapsed_s, green_remaining_s,
    ns_demand, ew_demand, ns_priority, ew_priority,
    decision, decision_reason, is_demo_mode

Display-only conveniences (no backend logic duplicated)
-------------------------------------------------------
* A small in-memory history buffer (st.session_state) feeds the
  "Observed vs Predicted" chart. Predictions are plotted at the frame they
  TARGET (made at tick t -> plotted at t + RF_PREDICT_HORIZON).
* "RF trained" is inferred from the backend's own output: before training the
  predictor returns confidence == 0 and a zero-width interval (EMA fallback).
  Those fallback values are labelled as such and are NOT plotted as RF output.

Run with:
    streamlit run src/traffic_system/dashboard_intersection.py
"""

from __future__ import annotations

import base64
import html
import time
from collections import deque
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from traffic_system import config
from traffic_system.intersection import (
    APPROACHES,
    FourWayIntersection,
    compute_is_demo_mode,
)


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Adaptive Traffic Signal Control",
    layout="wide",
)

APPROACH_LABELS = {"N": "NORTH", "E": "EAST", "S": "SOUTH", "W": "WEST"}

MODE_RESEARCH = "4 Independent Camera Feeds"
MODE_DEMO = "Demo — One Video / Quadrant Split"

HORIZON = config.RF_PREDICT_HORIZON
HISTORY_MAX = 600  # ticks kept in memory for the chart / decision log


# ============================================================================
# THEME
# ============================================================================

TEXT = "#d7dde5"
MUTED = "#8593a3"
FAINT = "#5b6878"
BORDER = "#243040"
PANEL = "#111821"
ACCENT = "#5aa9e6"   # data / pipeline accent (deliberately NOT a signal colour)
WARN = "#e0883a"     # warnings (distinct from the signal-yellow)
OFF = "#2a3340"

SIGNAL_COLORS = {"GREEN": "#2fbf71", "YELLOW": "#f2c230", "RED": "#e5534b"}
LEVEL_COLORS = {"LOW": "#7f8fa3", "MEDIUM": ACCENT, "HIGH": "#b48ead"}

_CSS = """<style>
.stApp{background:#0b0f14;color:#d7dde5}
[data-testid="stHeader"]{background:transparent}
[data-testid="stSidebar"]{background:#0f141b;border-right:1px solid #243040}
[data-testid="stSidebar"] label,[data-testid="stSidebar"] p,[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{color:#c3ccd7}
.block-container{padding-top:1.4rem;padding-bottom:2rem;max-width:1600px}
[data-testid="stVerticalBlock"]{gap:.5rem}
[data-testid="stImage"] img{border:1px solid #243040;border-radius:3px}
.tp-hdr{border-bottom:1px solid #243040;padding:0 0 10px 0;margin-bottom:4px}
.tp-title{font-size:1.6rem;font-weight:650;letter-spacing:.2px;color:#e6ebf1;line-height:1.2}
.tp-sub{color:#8593a3;font-size:.98rem;margin-top:3px}
.tp-tags{margin-top:9px;display:flex;gap:6px;flex-wrap:wrap}
.tp-tag{border:1px solid #2c3745;border-radius:3px;padding:1px 9px;font-size:.72rem;color:#9fb0c3;letter-spacing:.6px;font-family:monospace}
.tp-sec{font-size:.72rem;letter-spacing:1.3px;text-transform:uppercase;color:#8593a3;margin:10px 0 2px 0}
.tp-banner{border:1px solid #243040;border-left-width:3px;border-radius:3px;padding:7px 12px;font-size:.85rem;background:#111821}
.tp-banner b{letter-spacing:.5px}
.tp-pipe{display:flex;flex-wrap:wrap;align-items:stretch;gap:4px}
.tp-step{flex:1 1 120px;background:#111821;border:1px solid #243040;border-radius:3px;padding:6px 9px;min-width:120px}
.tp-step .n{font-size:.62rem;color:#5b6878;letter-spacing:1px}
.tp-step .t{font-size:.78rem;font-weight:600;color:#d7dde5;letter-spacing:.3px;margin-top:1px}
.tp-step .d{font-size:.66rem;color:#8593a3;margin-top:2px;line-height:1.25}
.tp-arrow{align-self:center;color:#5b6878;font-size:.9rem}
.tp-chead{display:flex;justify-content:space-between;align-items:center;background:#111821;border:1px solid #243040;border-radius:3px;padding:5px 10px}
.tp-cname{font-weight:650;letter-spacing:1px;font-size:.85rem}
.tp-ctag{font-size:.64rem;color:#8593a3;border:1px solid #2c3745;border-radius:3px;padding:0 6px;margin-left:8px;letter-spacing:.5px;font-family:monospace}
.tp-pill{font-size:.72rem;border:1px solid;border-radius:10px;padding:0 9px;letter-spacing:.6px;font-family:monospace}
.tp-novid{display:flex;align-items:center;justify-content:center;min-height:190px;background:#0d1218;border:1px dashed #243040;border-radius:3px;color:#5b6878;font-size:.78rem;letter-spacing:1.5px}
.tp-grid{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:#243040;border:1px solid #243040;border-radius:3px;overflow:hidden}
.tp-cell{background:#111821;padding:5px 10px}
.tp-lbl{font-size:.62rem;color:#8593a3;letter-spacing:.9px;text-transform:uppercase}
.tp-val{font-size:1.02rem;font-weight:600;color:#e6ebf1;font-family:monospace}
.tp-sub2{font-size:.62rem;color:#5b6878;margin-top:-1px}
.tp-chip{font-size:.78rem;font-weight:600;letter-spacing:.6px;font-family:monospace}
.tp-foot{font-size:.64rem;color:#5b6878;margin-top:3px}
.tp-conn{text-align:center;color:#5b6878;font-size:.8rem;line-height:1;letter-spacing:2px}
.tp-panel{background:#111821;border:1px solid #243040;border-radius:3px;padding:8px 11px;font-size:.75rem;color:#aeb9c6}
.tp-ptitle{font-size:.64rem;letter-spacing:1.2px;text-transform:uppercase;color:#8593a3;margin-bottom:5px}
.tp-phase{display:flex;align-items:center;gap:10px;border:1px solid;border-radius:3px;padding:7px 12px;background:#0d1218}
.tp-dot{width:11px;height:11px;border-radius:50%;display:inline-block}
.tp-phase-t{font-size:1.15rem;font-weight:700;letter-spacing:1.2px;font-family:monospace}
.tp-dec{margin-left:auto;font-size:.7rem;color:#aeb9c6;letter-spacing:.8px;font-family:monospace}
.tp-bar{display:flex;height:6px;border-radius:3px;overflow:hidden;background:#243040;margin:5px 0 2px 0}
.tp-reason{font-size:.74rem;color:#aeb9c6;border-left:2px solid #2c3745;padding:2px 9px;margin-top:7px;line-height:1.35}
table.tp-tbl{width:100%;border-collapse:collapse;font-size:.8rem}
table.tp-tbl th{text-align:left;font-size:.64rem;font-weight:500;letter-spacing:.9px;text-transform:uppercase;color:#8593a3;border-bottom:1px solid #2c3745;padding:5px 8px}
table.tp-tbl td{padding:6px 8px;border-bottom:1px solid #1b2430;color:#d7dde5;font-family:monospace}
table.tp-tbl td.s{font-family:inherit;color:#8593a3;font-size:.66rem}
.tp-log{font-family:monospace;font-size:.68rem;line-height:1.35;color:#aeb9c6}
.tp-log .t{color:#5b6878}
.tp-sm{display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px}
.tp-smc{border:1px solid #243040;border-radius:3px;padding:3px 6px;text-align:center;font-size:.66rem;font-family:monospace;color:#5b6878}
</style>"""


# ============================================================================
# SMALL HELPERS
# ============================================================================

def _h(s: str) -> str:
    """Strip indentation/newlines. Streamlit's markdown treats lines indented
    by 4+ spaces as code blocks, so HTML is always sent as one flat string."""
    return "".join(line.strip() for line in s.strip().splitlines())


def _md(target, markup: str) -> None:
    """Render HTML into `st` or an st.empty() placeholder."""
    target.markdown(_h(markup), unsafe_allow_html=True)


def _esc(x) -> str:
    return html.escape(str(x), quote=True)


def _fmt(x, nd: int = 1) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def rf_trained(res: dict) -> bool:
    """True once the backend's RF has produced a real (non-fallback) output.

    Untrained predictor => confidence == 0 and lower == upper == point (EMA
    fallback). Inferred purely from the backend's own returned fields."""
    return (
        res.get("pred_confidence", 0.0) > 0
        or res.get("pred_upper", 0.0) > res.get("pred_lower", 0.0)
    )


def phase_label(tick: dict | None) -> str:
    s = (tick or {}).get("signal_state")
    if not s:
        return "STANDBY" if not tick else "UNKNOWN"
    return s.replace("_", " ")


def phase_color(label: str) -> str:
    if "GREEN" in label:
        return SIGNAL_COLORS["GREEN"]
    if "YELLOW" in label:
        return SIGNAL_COLORS["YELLOW"]
    if label == "ALL RED":
        return SIGNAL_COLORS["RED"]
    return FAINT


def _display_name(path: str) -> str:
    return Path(path).name


def duplicate_source_note(sources: dict[str, str]) -> str | None:
    """Presentation-only warning for the case where some (but not all) of the
    four 'independent' cameras point at the same file. The backend treats that
    as research mode; the UI must not let it pass as four independent feeds."""
    by_path: dict[str, list[str]] = {}
    for a in APPROACHES:
        by_path.setdefault(sources[a], []).append(a)
    dupes = [v for v in by_path.values() if len(v) > 1]
    if not dupes or len(by_path) == 1:  # none, or full demo (handled elsewhere)
        return None
    groups = "; ".join(" + ".join(APPROACH_LABELS[a] for a in g) for g in dupes)
    return f"{groups} use the same video file — these are not independent cameras."


# ============================================================================
# VIDEO DISCOVERY
# ============================================================================

def scan_videos() -> list[str]:
    """Return available video files from the project's videos directory."""
    video_dir = Path("videos")
    if not video_dir.exists():
        return []
    return sorted(
        str(f)
        for f in video_dir.iterdir()
        if f.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")
    )


# ============================================================================
# STATIC HEADER
# ============================================================================

st.markdown(_CSS, unsafe_allow_html=True)

_md(st, """
<div class="tp-hdr">
<div class="tp-title">Vision-Based Adaptive Traffic Signal Control</div>
<div class="tp-sub">Uncertainty-Aware Short-Horizon Traffic Demand Prediction</div>
<div class="tp-tags">
<span class="tp-tag">YOLOv8</span><span class="tp-tag">DeepSORT</span>
<span class="tp-tag">Random Forest</span><span class="tp-tag">Adaptive Control</span>
</div>
</div>
""")


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:

    st.header("Configuration")
    st.subheader("Video Sources")

    videos = scan_videos()

    if not videos:
        st.error("No video files found in `videos/`.")
        st.stop()

    mode = st.radio(
        "Mode",
        options=[MODE_RESEARCH, MODE_DEMO],
        index=0,
        help=(
            "Research mode uses one video source per approach. "
            "Demo mode reuses one video and splits it into quadrants. "
            "Demo mode must not be used as the four-camera experimental result."
        ),
    )

    same_source = mode.startswith("Demo")

    if same_source:
        st.warning(
            "DEMO MODE — single video with quadrant split. "
            "Not a four-camera research experiment."
        )
        shared = st.selectbox("Video", videos, index=0)
        sources = {a: shared for a in APPROACHES}
    else:
        sources = {}
        for a in APPROACHES:
            sources[a] = st.selectbox(
                f"Approach {a} ({APPROACH_LABELS[a]})",
                videos,
                index=0,
                key=f"src_{a}",
            )

        if compute_is_demo_mode(sources):
            st.warning(
                "All four selected files are identical, so the backend will "
                "run this as DEMO (quadrant split), not as four independent "
                "cameras. Select four different videos for research mode."
            )
        else:
            note = duplicate_source_note(sources)
            if note:
                st.warning(note)
            else:
                st.success("Research mode — 4 independent camera feeds")

    process_every = st.slider("Process every N frames", 1, 5, 1)

    start_btn = st.button("▶ Start")
    stop_btn = st.button("■ Stop")

    st.subheader("Chart")
    _chart_label = st.selectbox(
        "Approach shown in chart",
        [f"{a} — {APPROACH_LABELS[a]}" for a in APPROACHES],
    )
    chart_approach = _chart_label.split(" ")[0]


# ============================================================================
# SESSION STATE
# ============================================================================

def _new_history() -> dict:
    return {
        "t": 0,
        "obs": {a: deque(maxlen=HISTORY_MAX) for a in APPROACHES},
        "pred": {a: deque(maxlen=HISTORY_MAX) for a in APPROACHES},
        "events": deque(maxlen=6),
    }


if "running_4way" not in st.session_state:
    st.session_state.running_4way = False

if "intersection" not in st.session_state:
    st.session_state.intersection = None

if "history" not in st.session_state:
    st.session_state.history = _new_history()


# ============================================================================
# SVG: THE FOUR-WAY INTERSECTION
# ============================================================================

def _lamps(cx: int, cy: int, vertical: bool, state: str | None) -> str:
    """One signal head (R/Y/G). Lit lamp = the backend's actual state."""
    if vertical:
        hx, hy, hw, hh = cx - 8, cy - 21, 16, 42
        pos = [(cx, cy - 13), (cx, cy), (cx, cy + 13)]
    else:
        hx, hy, hw, hh = cx - 21, cy - 8, 42, 16
        pos = [(cx - 13, cy), (cx, cy), (cx + 13, cy)]

    out = [
        f'<rect x="{hx}" y="{hy}" width="{hw}" height="{hh}" rx="3" '
        f'fill="#090d12" stroke="#2c3745"/>'
    ]
    for (x, y), name in zip(pos, ("RED", "YELLOW", "GREEN")):
        on = state == name
        fill = SIGNAL_COLORS[name] if on else OFF
        if on:
            out.append(
                f'<circle cx="{x}" cy="{y}" r="8" fill="{fill}" opacity="0.22"/>'
            )
        out.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{fill}"/>')
    return "".join(out)


def intersection_svg(tick: dict | None) -> str:
    """Four-way intersection drawn from the controller's signal_states."""
    states = (tick or {}).get("signal_states", {}) if tick else {}
    dens = (tick or {}).get("weighted_densities", {}) if tick else {}

    def bar_col(a: str) -> str:
        return SIGNAL_COLORS.get(states.get(a), "#3a4655")

    label = phase_label(tick)
    pcol = phase_color(label)
    words = label.split(" ", 1)
    line1, line2 = (words[0], words[1]) if len(words) == 2 else (label, "")

    def name_txt(a, x, y, anchor):
        d = f"demand {_fmt(dens[a])}" if a in dens else "demand —"
        return (
            f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="12" '
            f'font-weight="700" letter-spacing="1.5" fill="{TEXT}">'
            f'{APPROACH_LABELS[a]}</text>'
            f'<text x="{x}" y="{y + 15}" text-anchor="{anchor}" font-size="10.5" '
            f'fill="{MUTED}">{d}</text>'
        )

    def arrow(x1, y1, x2, y2, pts):
        return (
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{ACCENT}" stroke-width="2" opacity="0.8"/>'
            f'<polygon points="{pts}" fill="{ACCENT}" opacity="0.9"/>'
        )

    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 460 460" '
        'width="460" height="460" font-family="Helvetica,Arial,sans-serif">',
        '<rect width="460" height="460" rx="4" fill="#0d1218"/>',
        # roads
        '<rect x="190" y="0" width="80" height="460" fill="#19222d"/>',
        '<rect x="0" y="190" width="460" height="80" fill="#19222d"/>',
        # road edges
        '<path d="M190 0V190H0M270 0V190H460M190 460V270H0M270 460V270H460" '
        'fill="none" stroke="#2c3745" stroke-width="1.5"/>',
        # centre box, outlined in the current phase colour
        f'<rect x="190" y="190" width="80" height="80" fill="#0d141b" '
        f'stroke="{pcol}" stroke-width="2"/>',
        f'<text x="230" y="{226 if line2 else 234}" text-anchor="middle" '
        f'font-size="13" font-weight="700" letter-spacing="1" fill="{pcol}">'
        f'{_esc(line1)}</text>',
    ]
    if line2:
        svg.append(
            f'<text x="230" y="244" text-anchor="middle" font-size="12" '
            f'font-weight="700" letter-spacing="1" fill="{pcol}">{_esc(line2)}</text>'
        )

    # stop bars (colour = actual signal state of that approach)
    svg += [
        f'<rect x="190" y="178" width="80" height="5" fill="{bar_col("N")}"/>',
        f'<rect x="190" y="277" width="80" height="5" fill="{bar_col("S")}"/>',
        f'<rect x="178" y="190" width="5" height="80" fill="{bar_col("W")}"/>',
        f'<rect x="277" y="190" width="5" height="80" fill="{bar_col("E")}"/>',
    ]

    # signal heads
    svg += [
        _lamps(176, 156, True, states.get("N")),
        _lamps(284, 304, True, states.get("S")),
        _lamps(304, 176, False, states.get("E")),
        _lamps(156, 284, False, states.get("W")),
    ]

    # inbound arrows: every approach feeds the SAME intersection
    svg += [
        arrow(230, 58, 230, 112, "222,110 238,110 230,124"),
        arrow(230, 402, 230, 348, "222,350 238,350 230,336"),
        arrow(58, 230, 112, 230, "110,222 110,238 124,230"),
        arrow(402, 230, 348, 230, "350,222 350,238 336,230"),
    ]

    # labels
    svg += [
        name_txt("N", 230, 24, "middle"),
        name_txt("S", 230, 432, "middle"),
        name_txt("W", 14, 166, "start"),
        name_txt("E", 446, 296, "end"),
    ]
    svg.append("</svg>")
    return "".join(svg)


def _svg_img(svg: str) -> str:
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return (
        f'<div><img alt="Four-way intersection signal state" '
        f'style="width:100%;display:block;border:1px solid {BORDER};border-radius:3px" '
        f'src="data:image/svg+xml;base64,{b64}"/></div>'
    )


# ============================================================================
# HTML BUILDERS
# ============================================================================

def mode_banner_html(is_demo: bool, sources: dict[str, str]) -> str:
    if is_demo:
        return (
            f'<div class="tp-banner" style="border-left-color:{WARN}">'
            f'<b style="color:{WARN}">DEMO — ONE VIDEO / QUADRANT SPLIT</b>'
            f'<span style="color:{MUTED}"> &nbsp;·&nbsp; This is NOT a four-camera '
            f"research experiment: a single video is split into four quadrant "
            f"ROIs.</span></div>"
        )
    note = duplicate_source_note(sources)
    if note:
        return (
            f'<div class="tp-banner" style="border-left-color:{WARN}">'
            f'<b style="color:{WARN}">SOURCE WARNING</b>'
            f'<span style="color:{MUTED}"> &nbsp;·&nbsp; {_esc(note)}</span></div>'
        )
    return (
        f'<div class="tp-banner" style="border-left-color:{ACCENT}">'
        f'<b style="color:{ACCENT}">4 INDEPENDENT CAMERA FEEDS</b>'
        f'<span style="color:{MUTED}"> &nbsp;·&nbsp; Research mode: one video '
        f"source per approach.</span></div>"
    )


def pipeline_html(is_demo: bool) -> str:
    w = config.VEHICLE_WEIGHTS
    weights = (
        f"car {w['car']:g} · bus/truck {w['bus']:g} · "
        f"motorcycle {w['motorcycle']:g} · bicycle {w['bicycle']:g}"
    )
    if is_demo:
        first = ("1 VIDEO · 4 QUADRANT ROIs", "DEMO ONLY — not four cameras")
    else:
        first = ("4 INDEPENDENT CAMERA APPROACHES", "North · East · South · West")
    steps = [
        ("CAMERA INPUTS", first[0], first[1]),
        ("DETECTION", "YOLOv8", "per-class confidence, ROI filter"),
        ("TRACKING", "DeepSORT", "persistent vehicle IDs"),
        ("DEMAND", "WEIGHTED DEMAND", weights),
        ("PREDICTION", "RF PREDICTION",
         f"{HORIZON}-frame horizon · {config.RF_WINDOW_SIZE}-frame window"),
        ("UNCERTAINTY", "PREDICTION UNCERTAINTY",
         f"RF tree spread, z = {config.RF_UNCERTAINTY_Z:g}"),
        ("CONTROL", "ADAPTIVE SIGNAL CONTROLLER",
         "one intersection · NS / EW conflict-safe"),
    ]
    parts = []
    for i, (n, t, d) in enumerate(steps):
        if i:
            parts.append('<div class="tp-arrow">→</div>')
        parts.append(
            f'<div class="tp-step"><div class="n">{i + 1} · {n}</div>'
            f'<div class="t">{_esc(t)}</div><div class="d">{_esc(d)}</div></div>'
        )
    return f'<div class="tp-pipe">{"".join(parts)}</div>'


def camera_head_html(a: str, tick: dict | None, is_demo: bool) -> str:
    state = (tick or {}).get("signal_states", {}).get(a) if tick else None
    col = SIGNAL_COLORS.get(state, FAINT)
    tag = f"QUADRANT {a} · DEMO" if is_demo else f"CAM {a}"
    pill = _esc(state or "—")
    return (
        f'<div class="tp-chead"><div><span class="tp-cname">{APPROACH_LABELS[a]}</span>'
        f'<span class="tp-ctag">{tag}</span></div>'
        f'<span class="tp-pill" style="color:{col};border-color:{col}">● {pill}</span></div>'
    )


def _cell(label: str, value: str, sub: str = "", color: str | None = None) -> str:
    style = f' style="color:{color}"' if color else ""
    subh = f'<div class="tp-sub2">{sub}</div>' if sub else ""
    return (
        f'<div class="tp-cell"><div class="tp-lbl">{label}</div>'
        f'<div class="tp-val"{style}>{value}</div>{subh}</div>'
    )


def _level_chip(level: str) -> str:
    return (
        f'<span class="tp-chip" style="color:{LEVEL_COLORS.get(level, MUTED)}">'
        f"{_esc(level)}</span>"
    )


def camera_metrics_html(a: str, tick: dict | None) -> str:
    if not tick:
        cells = [
            _cell("Vehicles", "—"), _cell("Current demand", "—"),
            _cell("Predicted demand", "—"), _cell("Uncertainty", "—"),
            _cell("Traffic level", "—"), _cell("Predicted level", "—"),
        ]
        return f'<div class="tp-grid">{"".join(cells)}</div>'

    res = tick["approach_results"][a]
    trained = rf_trained(res)

    if trained:
        pred_sub = f"RF · +{HORIZON} frames"
        unc_val = f"{_fmt(res['pred_lower'])} – {_fmt(res['pred_upper'])}"
        unc_sub = f"width {_fmt(res['pred_upper'] - res['pred_lower'])}"
        foot = f"RF fit R² {res['pred_confidence']:.2f} (in-sample)"
    else:
        pred_sub = "EMA fallback · RF untrained"
        unc_val = "N/A"
        unc_sub = "RF warming up"
        foot = "RF warming up — no uncertainty interval yet"

    cells = [
        _cell("Vehicles", _esc(res["vehicle_count"])),
        _cell("Current demand", _fmt(tick["weighted_densities"][a])),
        _cell("Predicted demand", _fmt(res["pred_density"]), pred_sub),
        _cell("Uncertainty", unc_val, unc_sub),
        _cell("Traffic level", _level_chip(res["traffic_level"])),
        _cell("Predicted level", _level_chip(res["pred_level"])),
    ]
    return (
        f'<div class="tp-grid">{"".join(cells)}</div>'
        f'<div class="tp-foot">{foot}</div>'
    )


def controller_html(tick: dict | None) -> str:
    if not tick:
        return (
            f'<div class="tp-panel"><div class="tp-ptitle">Intersection controller</div>'
            f'<span style="color:{FAINT}">Standby — press Start in the sidebar.</span></div>'
        )

    label = phase_label(tick)
    col = phase_color(label)
    sstate = tick.get("signal_state", "")
    is_green = sstate.endswith("GREEN")

    group = tick.get("active_group")
    gname = {"NS": "NORTH + SOUTH", "EW": "EAST + WEST"}.get(group, str(group))
    if sstate.endswith("GREEN"):
        movement = gname
    elif sstate.endswith("YELLOW"):
        movement = f"{gname} (clearing)"
    elif sstate == "ALL_RED":
        movement = f"none — next: {gname}"
    else:
        movement = gname

    elapsed = f"{_fmt(tick.get('green_elapsed_s'))} s" if is_green else "—"
    remaining = f"{_fmt(tick.get('green_remaining_s'))} s" if is_green else "—"

    ns = float(tick.get("ns_demand", 0.0))
    ew = float(tick.get("ew_demand", 0.0))
    tot = ns + ew
    ns_pct = 100.0 * ns / tot if tot > 0 else 50.0

    cells = [
        _cell("Active movement", _esc(movement)),
        _cell("Controller decision", _esc(tick.get("decision", "—"))),
        _cell("Green elapsed", elapsed),
        _cell("Remaining to max green", remaining),
        _cell("NS demand", _fmt(ns, 2)),
        _cell("EW demand", _fmt(ew, 2)),
    ]
    return (
        f'<div class="tp-panel"><div class="tp-ptitle">Intersection controller</div>'
        f'<div class="tp-phase" style="border-color:{col}">'
        f'<span class="tp-dot" style="background:{col}"></span>'
        f'<span class="tp-phase-t" style="color:{col}">{_esc(label)}</span>'
        f'<span class="tp-dec">DECISION · {_esc(tick.get("decision", "—"))}</span></div>'
        f'<div class="tp-grid" style="margin-top:7px">{"".join(cells)}</div>'
        f'<div class="tp-bar"><div style="width:{ns_pct:.1f}%;background:{ACCENT}"></div>'
        f'<div style="width:{100 - ns_pct:.1f}%;background:#3a4a5e"></div></div>'
        f'<div class="tp-foot">NS vs EW share of smoothed fused demand · '
        f'priority NS {_fmt(tick.get("ns_priority"), 2)} / EW {_fmt(tick.get("ew_priority"), 2)} '
        f'(wait-weighted)</div>'
        f'<div class="tp-reason">{_esc(tick.get("decision_reason", "No decision reason available."))}</div>'
        f"</div>"
    )


def sources_html(sources: dict[str, str], is_demo: bool) -> str:
    rows = "".join(
        f'<div><span style="color:{FAINT}">{a}</span> &nbsp;{_esc(_display_name(sources[a]))}</div>'
        for a in APPROACHES
    )
    note = (
        f'<div style="color:{WARN};margin-top:4px">single video · quadrant ROIs</div>'
        if is_demo
        else f'<div style="color:{MUTED};margin-top:4px">one file per approach</div>'
    )
    return (
        f'<div class="tp-panel"><div class="tp-ptitle">Camera sources</div>'
        f'<div class="tp-log">{rows}</div>{note}</div>'
    )


def state_machine_html(tick: dict | None) -> str:
    cur = (tick or {}).get("signal_state")
    nxt = (tick or {}).get("active_group")  # during ALL_RED: group about to go green

    def chip(txt: str, active: bool, col: str) -> str:
        style = (
            f"border-color:{col};color:{col}" if active else ""
        )
        return f'<div class="tp-smc" style="{style}">{txt}</div>'

    g, y, r = SIGNAL_COLORS["GREEN"], SIGNAL_COLORS["YELLOW"], SIGNAL_COLORS["RED"]
    chips = [
        chip("NS GREEN", cur == "NS_GREEN", g),
        chip("NS YELLOW", cur == "NS_YELLOW", y),
        chip("ALL RED → EW", cur == "ALL_RED" and nxt == "EW", r),
        chip("EW GREEN", cur == "EW_GREEN", g),
        chip("EW YELLOW", cur == "EW_YELLOW", y),
        chip("ALL RED → NS", cur == "ALL_RED" and nxt == "NS", r),
    ]
    return (
        f'<div class="tp-panel"><div class="tp-ptitle">Controller state machine</div>'
        f'<div class="tp-sm">{"".join(chips)}</div></div>'
    )


def decision_log_html() -> str:
    events = list(st.session_state.history["events"])
    if not events:
        body = f'<span style="color:{FAINT}">No switch events yet.</span>'
    else:
        body = "".join(
            f'<div><span class="t">f{t}</span> {_esc(reason)}</div>'
            for t, _state, reason in reversed(events)
        )
    return (
        f'<div class="tp-panel"><div class="tp-ptitle">Recent switch decisions</div>'
        f'<div class="tp-log">{body}</div></div>'
    )


def legend_html() -> str:
    g, y, r = SIGNAL_COLORS["GREEN"], SIGNAL_COLORS["YELLOW"], SIGNAL_COLORS["RED"]
    return (
        f'<div class="tp-panel"><div class="tp-ptitle">Key</div>'
        f'<div><span style="color:{g}">●</span> green phase &nbsp; '
        f'<span style="color:{y}">●</span> yellow &nbsp; '
        f'<span style="color:{r}">●</span> red / all-red</div>'
        f'<div style="margin-top:4px;color:{MUTED}">Signal heads and stop bars show the '
        f"controller's actual state. Demand = class-weighted vehicle count.</div></div>"
    )


def results_table_html(tick: dict | None) -> str:
    head = (
        "<tr><th>Approach</th><th>Current demand</th>"
        f"<th>Predicted demand (+{HORIZON} fr)</th>"
        "<th>Uncertainty [lower – upper]</th><th>Traffic level</th></tr>"
    )
    rows = []
    for a in APPROACHES:
        if not tick:
            rows.append(
                f"<tr><td>{APPROACH_LABELS[a]}</td><td>—</td><td>—</td><td>N/A</td><td>—</td></tr>"
            )
            continue
        res = tick["approach_results"][a]
        trained = rf_trained(res)
        scol = SIGNAL_COLORS.get(tick.get("signal_states", {}).get(a), FAINT)
        pred = (
            f"{_fmt(res['pred_density'])} {_level_chip(res['pred_level'])}"
            if trained
            else f"{_fmt(res['pred_density'])} <span class='s'>EMA fallback</span>"
        )
        unc = (
            f"{_fmt(res['pred_lower'])} – {_fmt(res['pred_upper'])}"
            if trained
            else f"<span style='color:{FAINT}'>N/A</span> <span class='s'>RF untrained</span>"
        )
        rows.append(
            f'<tr><td><span style="color:{scol}">●</span> {APPROACH_LABELS[a]}</td>'
            f"<td>{_fmt(tick['weighted_densities'][a])}</td><td>{pred}</td>"
            f"<td>{unc}</td><td>{_level_chip(res['traffic_level'])}</td></tr>"
        )
    foot = (
        f'<div class="tp-foot">Traffic level = vehicle-count band (LOW ≤ {config.LOW_MAX}, '
        f"MEDIUM ≤ {config.MEDIUM_MAX}, HIGH above). Predicted level = weighted-density band "
        f"(LOW &lt; {config.DENSITY_LOW:g}, MEDIUM &lt; {config.DENSITY_MEDIUM:g}). "
        f"Uncertainty = RF point estimate ± {config.RF_UNCERTAINTY_Z:g} × tree-vote std "
        f"(lower clipped at 0); N/A until the RF has trained.</div>"
    )
    return f'<table class="tp-tbl">{head}{"".join(rows)}</table>{foot}'


# ============================================================================
# LAYOUT (built once; placeholders are filled per tick)
# ============================================================================

_md(st, '<div class="tp-sec">System architecture</div>')
pipeline_ph = st.empty()
mode_ph = st.empty()

_md(st, '<div class="tp-sec">Four approaches → one intersection controller</div>')

cam: dict[str, dict] = {}


def _camera_slot(a: str) -> None:
    cam[a] = {
        "head": st.empty(),
        "video": st.empty(),
        "metrics": st.empty(),
    }


_COLS = [1, 1.15, 1]
row1 = st.columns(_COLS)
row2 = st.columns(_COLS)
row3 = st.columns(_COLS)

with row1[0]:
    sources_ph = st.empty()
with row1[1]:
    _camera_slot("N")
    _md(st, '<div class="tp-conn">▼ &nbsp;N → intersection &nbsp;▼</div>')
with row1[2]:
    statemachine_ph = st.empty()

with row2[0]:
    _camera_slot("W")
    _md(st, '<div class="tp-conn">W → intersection &nbsp;▶</div>')
with row2[1]:
    svg_ph = st.empty()
    controller_ph = st.empty()
with row2[2]:
    _camera_slot("E")
    _md(st, '<div class="tp-conn">◀ &nbsp;E → intersection</div>')

with row3[0]:
    log_ph = st.empty()
with row3[1]:
    _md(st, '<div class="tp-conn">▲ &nbsp;S → intersection &nbsp;▲</div>')
    _camera_slot("S")
with row3[2]:
    _md(st, legend_html())

_md(st, '<div class="tp-sec">Demand, prediction and uncertainty</div>')
left, right = st.columns([1.1, 1])
with left:
    table_ph = st.empty()
with right:
    chart_ph = st.empty()
    chart_note_ph = st.empty()

_md(st, """
<div class="tp-foot" style="margin-top:14px">Research prototype. Signal timings are
illustrative simulation values, not engineering-validated; the controller models two
compatible movement groups (NS / EW) only — no turning movements, pedestrian phases or
preemption. No accuracy or delay-reduction figure is shown here: only live backend
outputs are displayed.</div>
""")


# ============================================================================
# OBSERVED vs PREDICTED CHART
# ============================================================================

def render_chart(a: str) -> None:
    h = st.session_state.history
    obs = list(h["obs"][a])
    pred = list(h["pred"][a])

    if not obs:
        _md(chart_ph, """
<div class="tp-novid" style="min-height:250px">OBSERVED VS PREDICTED DEMAND — AWAITING DATA</div>
""")
        _md(chart_note_ph, "")
        return

    try:
        import altair as alt
    except ImportError:  # Altair ships with Streamlit; defensive only.
        chart_ph.line_chart(
            pd.DataFrame({"observed": [v for _, v in obs]})
        )
        return

    s_obs, s_pred = "Observed demand", f"RF prediction (+{HORIZON} frames)"
    rows = [{"frame": t, "demand": v, "series": s_obs} for t, v in obs]
    rows += [{"frame": t, "demand": p, "series": s_pred} for t, p, _lo, _hi in pred]
    lines_df = pd.DataFrame(rows)

    x = alt.X("frame:Q", title="Frame (prediction plotted at its target frame)",
              scale=alt.Scale(zero=False))
    lines = alt.Chart(lines_df).mark_line(strokeWidth=1.8).encode(
        x=x,
        y=alt.Y("demand:Q", title="Weighted demand"),
        color=alt.Color(
            "series:N",
            scale=alt.Scale(domain=[s_obs, s_pred], range=["#b9c4d0", ACCENT]),
            legend=alt.Legend(title=None, orient="bottom"),
        ),
    )
    layers = [lines]
    if pred:
        band_df = pd.DataFrame(
            [{"frame": t, "lower": lo, "upper": hi} for t, _p, lo, hi in pred]
        )
        band = alt.Chart(band_df).mark_area(opacity=0.18, color=ACCENT).encode(
            x=x, y="lower:Q", y2="upper:Q",
        )
        layers.insert(0, band)

    chart = (
        alt.layer(*layers)
        .properties(height=260, title=f"{APPROACH_LABELS[a]} approach")
        .configure(background="transparent")
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor=MUTED, titleColor=MUTED, gridColor="#1b2430",
            domainColor="#2c3745", tickColor="#2c3745",
        )
        .configure_legend(labelColor=MUTED)
        .configure_title(color=TEXT, fontSize=12, anchor="start")
    )

    for kwargs in (
        {"width": "stretch", "theme": None},
        {"use_container_width": True, "theme": None},
        {"use_container_width": True},
        {},
    ):
        try:
            chart_ph.altair_chart(chart, **kwargs)
            break
        except TypeError:
            continue

    if pred:
        note = (
            "Shaded band = RF interval [lower, upper] (tree-vote spread). A prediction made at "
            f"frame t is drawn at frame t + {HORIZON}, the frame it forecasts."
        )
    else:
        note = (
            "RF not trained yet — only observed demand is shown. EMA-fallback values are "
            "not plotted as RF predictions."
        )
    _md(chart_note_ph, f'<div class="tp-foot">{_esc(note)}</div>')


# ============================================================================
# RENDERING
# ============================================================================

def render_static(is_demo: bool, src: dict[str, str]) -> None:
    _md(pipeline_ph, pipeline_html(is_demo))
    _md(mode_ph, mode_banner_html(is_demo, src))
    _md(sources_ph, sources_html(src, is_demo))


def render_idle(src: dict[str, str]) -> None:
    is_demo = compute_is_demo_mode(src)
    render_static(is_demo, src)
    for a in APPROACHES:
        _md(cam[a]["head"], camera_head_html(a, None, is_demo))
        _md(cam[a]["video"], '<div class="tp-novid">NO SIGNAL — PRESS START</div>')
        _md(cam[a]["metrics"], camera_metrics_html(a, None))
    _md(svg_ph, _svg_img(intersection_svg(None)))
    _md(controller_ph, controller_html(None))
    _md(statemachine_ph, state_machine_html(None))
    _md(log_ph, decision_log_html())
    _md(table_ph, results_table_html(None))
    render_chart(chart_approach)


def record_history(tick: dict) -> None:
    """Append this tick's backend outputs to the in-memory visualisation buffer."""
    h = st.session_state.history
    h["t"] += 1
    t = h["t"]

    for a in APPROACHES:
        res = tick["approach_results"][a]
        h["obs"][a].append((t, float(tick["weighted_densities"][a])))
        if rf_trained(res):
            # Prediction made now forecasts demand HORIZON frames ahead.
            h["pred"][a].append(
                (
                    t + HORIZON,
                    float(res["pred_density"]),
                    float(res["pred_lower"]),
                    float(res["pred_upper"]),
                )
            )

    if tick.get("decision") == "SWITCH":
        reason = tick.get("decision_reason", "")
        if not h["events"] or h["events"][-1][2] != reason:
            h["events"].append((t, tick.get("signal_state"), reason))


def render_tick(tick: dict, src: dict[str, str]) -> None:
    is_demo = bool(tick.get("is_demo_mode"))
    render_static(is_demo, src)

    for a in APPROACHES:
        res = tick["approach_results"][a]
        ph = cam[a]

        _md(ph["head"], camera_head_html(a, tick, is_demo))
        ph["video"].image(cv2.cvtColor(res["annotated_frame"], cv2.COLOR_BGR2RGB))
        _md(ph["metrics"], camera_metrics_html(a, tick))

    _md(svg_ph, _svg_img(intersection_svg(tick)))
    _md(controller_ph, controller_html(tick))
    _md(statemachine_ph, state_machine_html(tick))
    _md(log_ph, decision_log_html())
    _md(table_ph, results_table_html(tick))
    render_chart(chart_approach)


# ============================================================================
# START / STOP
# ============================================================================

if start_btn:

    try:
        fw_new = FourWayIntersection(sources=sources)
        fw_new.open()
    except RuntimeError as exc:
        st.session_state.running_4way = False
        st.error(str(exc))
    else:
        st.session_state.intersection = fw_new
        st.session_state.history = _new_history()
        st.session_state.running_4way = True

if stop_btn:

    st.session_state.running_4way = False

    if st.session_state.intersection:
        st.session_state.intersection.close()


# ============================================================================
# MAIN PROCESSING LOOP
# ============================================================================

render_idle(sources)

if (
    st.session_state.running_4way
    and st.session_state.intersection
):

    fw: FourWayIntersection = st.session_state.intersection

    frame_idx = 0

    try:

        while st.session_state.running_4way:

            frame_idx += 1

            if frame_idx % process_every != 0:
                continue

            tick = fw.process_tick()

            record_history(tick)
            render_tick(tick, fw.sources)

            time.sleep(0.01)

    finally:

        pass
