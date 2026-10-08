"""Visual layer of the Streamlit app: palette, page styles, the hero trace and the Plotly theme.

Palette (validated for colour-blind separation with the dataviz six-checks script):
    signals   log(EU/AU) coral · EU blue · AU teal · TU gold · 1-max p violet
    neutrals  ink for text and error-based detectors, lavender page, night-indigo sidebar / hero
Coral is reserved for the proposed signal and the primary action, so "coral" always means "our method".
"""
from __future__ import annotations

import base64
import html
import json
import os

import numpy as np
import pandas as pd

INK = "#1E2340"
INK_SOFT = "#5A5F7A"
PAGE = "#F6F4F9"
SURFACE = "#FFFFFF"
LINE = "#DCD7E8"
GRID = "#ECE9F2"
NIGHT = "#26254A"
NIGHT_TEXT = "#EEEBF7"
NIGHT_SOFT = "#B9B4D6"
CORAL = "#E0475F"
DROP = "#B7791F"          # the visible accuracy drop: amber, kept clearly apart from the coral signal
RAMP = "#EEE8F7"          # drift ramp band

SIG = {"lr": CORAL, "eu": "#2B67B5", "au": "#13A08E", "tu": "#C98500", "conf": "#9B59C9"}
OTHER = {"err": "#3B3F58", "ks": "#6C5B7B", "em": "#C06C84"}
SIG_NAME = {"lr": "log(EU/AU)", "eu": "Epistemic EU", "au": "Aleatoric AU", "tu": "Total entropy",
            "conf": "1 − max p"}

FONT_BODY = "Atkinson Hyperlegible, sans-serif"
FONT_HEAD = "Bricolage Grotesque, sans-serif"


def detector_color(name: str) -> str:
    if name.endswith("[err]"):
        return OTHER["err"]
    if name.startswith("KS"):
        return OTHER["ks"]
    if name.startswith("EM"):
        return OTHER["em"]
    return SIG.get(name[name.find("[") + 1:-1], INK_SOFT)


CSS = f"""
<style>
.block-container {{ padding-top: 4.2rem; padding-bottom: 4rem; max-width: 1240px; }}
h1, h2, h3, h4 {{ letter-spacing: -0.012em; color: {INK}; }}

/* hero: the instrument screen */
.bdx-hero {{ background: {NIGHT}; color: {NIGHT_TEXT}; border-radius: 22px; padding: 30px 34px 18px 34px;
            margin-bottom: 1.4rem; }}
.bdx-hero h1 {{ font-family: {FONT_HEAD}; font-weight: 700; font-size: 2.55rem; line-height: 1.08; margin: 0 0 .55rem 0;
               color: {NIGHT_TEXT}; max-width: 22ch; padding: 0; }}
.bdx-hero p {{ font-size: 1.12rem; line-height: 1.5; color: {NIGHT_SOFT}; max-width: 62ch; margin: 0 0 .6rem 0; }}
.bdx-hero img.trace {{ width: 100%; height: auto; display: block; margin-top: .5rem; }}
.bdx-hero .key {{ display: flex; flex-wrap: wrap; gap: .5rem 1.6rem; font-size: .92rem; color: {NIGHT_SOFT}; margin-top: .2rem; }}
.bdx-hero .key i {{ display: inline-block; width: 18px; height: 3px; border-radius: 2px; vertical-align: middle;
                    margin-right: .45rem; }}

/* tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: .3rem; border-bottom: 1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ font-family: {FONT_HEAD}; font-weight: 600; font-size: 1.04rem; padding: .55rem .95rem;
                                color: {INK_SOFT}; }}
.stTabs [data-baseweb="tab"][aria-selected="true"] {{ color: {INK}; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: {CORAL}; height: 3px; border-radius: 3px; }}

/* readout strip */
.bdx-readout {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); background: {SURFACE};
               border: 1px solid {LINE}; border-radius: 16px; margin: .4rem 0 1.1rem 0; overflow: hidden; }}
.bdx-readout > div {{ padding: 16px 20px; border-left: 1px solid {GRID}; }}
.bdx-readout > div:first-child {{ border-left: none; }}
.bdx-readout .lab {{ font-size: .93rem; color: {INK_SOFT}; }}
.bdx-readout .val {{ font-family: {FONT_HEAD}; font-weight: 700; font-size: 1.95rem; color: {INK}; line-height: 1.2; }}
.bdx-readout .note {{ font-size: .9rem; color: {INK_SOFT}; }}
.bdx-readout .val.coral {{ color: {CORAL}; }}
.bdx-readout .val.drop {{ color: {DROP}; }}
.bdx-readout .val.ink {{ color: {INK}; }}
@media (max-width: 760px) {{ .bdx-readout {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}

/* hypotheses with verdicts */
.bdx-hyp {{ border-top: 1px solid {LINE}; }}
.bdx-hyp > div {{ display: grid; grid-template-columns: 3.2rem 1fr auto; gap: .9rem; align-items: start;
                 padding: .85rem 0; border-bottom: 1px solid {LINE}; }}
.bdx-hyp .n {{ font-family: {FONT_HEAD}; font-weight: 700; font-size: 1.15rem; color: {INK}; }}
.bdx-hyp .t b {{ color: {INK}; }}
.bdx-hyp .t span {{ display: block; color: {INK_SOFT}; font-size: .95rem; margin-top: .15rem; }}
.chip {{ display: inline-block; white-space: nowrap; font-size: .86rem; font-weight: 700; padding: .22rem .65rem;
        border-radius: 999px; border: 1.5px solid; }}
.chip.yes {{ color: #0B7A6C; border-color: #13A08E; background: #E3F5F2; }}
.chip.no {{ color: #B2283F; border-color: {CORAL}; background: #FCE8EB; }}
.chip.part {{ color: #8A5D00; border-color: #C98500; background: #FBF1DC; }}
.chip.info {{ color: #6E3D99; border-color: #9B59C9; background: #F3EAFA; }}

.bdx-prose {{ max-width: 68ch; font-size: 1.05rem; line-height: 1.6; }}
.bdx-note {{ color: {INK_SOFT}; font-size: .95rem; }}

/* sidebar */
section[data-testid="stSidebar"] h2 {{ font-family: {FONT_HEAD}; font-size: 1.45rem; margin-bottom: .2rem; }}
section[data-testid="stSidebar"] .bdx-side-note {{ color: {NIGHT_SOFT}; font-size: .93rem; line-height: 1.45; }}
[data-testid="stImageCaption"], [data-testid="caption"] {{ font-size: .93rem; }}
</style>
"""


def hero_html(res_dir: str) -> str:
    """The hero panel with the real log(EU/AU) trace of the example experiment (credit, covariate, MC-Dropout)."""
    head = ("<h1>A model that notices it has never seen this before</h1>"
            "<p>Split a model's uncertainty into noise in the data (aleatoric) and the model's own ignorance "
            "(epistemic). Their ratio, log(EU/AU), is watched as an alarm for a changing world, "
            "without waiting for the correct answers.</p>")
    p = os.path.join(res_dir, "example_trace.npz")
    if not os.path.exists(p):
        return f'<div class="bdx-hero">{head}</div>'
    z = np.load(p, allow_pickle=True)
    lr = pd.Series(z["sig_lr"]).rolling(250, min_periods=60).mean().to_numpy()
    L = len(lr)
    t0, t_acc = int(z["t0"]), int(z["t_acc"])
    pr = z["p"]
    ramp_end = t0 + int(np.argmax(pr[t0:] >= 1)) if (pr >= 1).any() else L
    alarms = json.loads(str(z["alarms"]))
    al = [a for a in alarms.get("PH[lr]", []) if a >= t0]
    alarm = al[0] if al else None
    err_al = [a for a in alarms.get("PH[err]", []) if a >= t0]
    err_alarm = err_al[0] if err_al else None

    W, H, x0, x1, y0, y1 = 1000, 230, 10, 990, 44, 196
    idx = np.arange(60, L, 12)
    v = lr[idx]
    lo, hi = np.nanmin(v), np.nanmax(v)
    X = lambda t: x0 + (x1 - x0) * t / (L - 1)
    Y = lambda val: y1 - (y1 - y0) * (val - lo) / (hi - lo + 1e-12)
    pts = " ".join(f"{X(t):.1f},{Y(val):.1f}" for t, val in zip(idx, v))
    area = f"{X(idx[0]):.1f},{y1} " + pts + f" {X(idx[-1]):.1f},{y1}"

    def vline(t, color, dash, *_):
        x = X(t)
        da = 'stroke-dasharray="5 5"' if dash else ""
        return f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y0 - 14}" y2="{y1}" stroke="{color}" stroke-width="2" {da}/>'

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
             f'<style>@font-face{{font-family:"Atkinson Hyperlegible";src:url(data:font/woff2;base64,{_font_b64()}) '
             'format("woff2");}</style>',
             '<defs><linearGradient id="bdxfill" x1="0" x2="0" y1="0" y2="1">'
             f'<stop offset="0%" stop-color="{CORAL}" stop-opacity="0.28"/>'
             f'<stop offset="100%" stop-color="{CORAL}" stop-opacity="0"/></linearGradient></defs>',
             f'<rect x="{X(t0):.1f}" y="{y0 - 14}" width="{X(ramp_end) - X(t0):.1f}" height="{y1 - y0 + 14}" '
             f'fill="#FFFFFF" fill-opacity="0.06" rx="6"/>',
             f'<line x1="{x0}" x2="{x1}" y1="{y1}" y2="{y1}" stroke="{NIGHT_SOFT}" stroke-opacity="0.35"/>',
             f'<polygon points="{area}" fill="url(#bdxfill)"/>',
             f'<polyline points="{pts}" fill="none" stroke="{CORAL}" stroke-width="3" stroke-linejoin="round" '
             'stroke-linecap="round"/>',
             vline(t0, NIGHT_SOFT, True, "drift begins", "end", y0 - 2, "h-t0")]
    if alarm is not None:
        parts.append(f'<circle cx="{X(alarm):.1f}" cy="{Y(lr[alarm]):.1f}" r="6" fill="{CORAL}" stroke="{NIGHT}" '
                     'stroke-width="2.5"/>')
    if t_acc > 0:
        parts.append(vline(t_acc, "#F4C86A", False, f"accuracy drop visible, t = {t_acc}", "start", y0 - 2, "h-drop"))
    if err_alarm is not None:
        parts.append(vline(err_alarm, NIGHT_SOFT, False, f"error-based alarm, t = {err_alarm}", "start", y0 + 22,
                           "h-err"))
    parts.append(f'<text x="{x0}" y="{H - 8}" font-size="13" fill="{NIGHT_SOFT}" '
                 'font-family="Atkinson Hyperlegible, sans-serif">record 0</text>')
    parts.append(f'<text x="{x1}" y="{H - 8}" text-anchor="end" font-size="13" fill="{NIGHT_SOFT}" '
                 f'font-family="Atkinson Hyperlegible, sans-serif">record {L:,}</text>')
    parts.append("</svg>")
    svg = base64.b64encode("".join(parts).encode()).decode()
    img = (f'<img class="trace" src="data:image/svg+xml;base64,{svg}" alt="log(EU/AU) rises once the drift begins; '
           f'our label-free alarm fires before the error-based detectors">')
    def k(color, text, dash=False):
        bar = (f'border-top:2px dashed {color};height:0' if dash else f'background:{color}')
        return f'<span><i style="{bar}"></i>{html.escape(text)}</span>'
    items = [k(CORAL, "log(EU/AU) on credit data, covariate drift (moving average)"),
             k(NIGHT_SOFT, f"drift begins, record {t0:,}", dash=True)]
    if alarm is not None:
        items.append(f'<span><b style="display:inline-block;width:11px;height:11px;border-radius:50%;'
                     f'background:{CORAL};margin-right:.45rem;vertical-align:middle"></b>our alarm, record {alarm:,}'
                     '</span>')
    if t_acc > 0:
        items.append(k("#F4C86A", f"accuracy drop visible, record {t_acc:,}"))
    if err_alarm is not None:
        items.append(k(NIGHT_SOFT, f"first error-based alarm, record {err_alarm:,}"))
    key = f'<div class="key">{"".join(items)}</div>'
    return f'<div class="bdx-hero">{head}{img}{key}</div>'


def _font_b64() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "static", "fonts", "atkinson-hyperlegible-latin-400-normal.woff2"), "rb") as f:
        return base64.b64encode(f.read()).decode()


def readout_html(cells) -> str:
    """cells: list of (label, value, note, css-class)."""
    out = []
    for lab, val, note, cls in cells:
        out.append(f'<div><div class="lab">{html.escape(lab)}</div><div class="val {cls}">{html.escape(val)}</div>'
                   f'<div class="note">{html.escape(note or "")}&nbsp;</div></div>')
    return f'<div class="bdx-readout">{"".join(out)}</div>'


HYPOTHESES = [
    ("H1", "Early warning", "The label-free alarm comes before the accuracy drop becomes visible.",
     "Earlier than every error-based detector, but before the visible drop in only 51 % of runs.",
     "no", "Not confirmed"),
    ("H2", "The split matters", "log(EU/AU) separates drifted data better than EU, AU, total entropy or 1 − max p.",
     "Separation 0.80 vs 0.67 for EU; 8 of 8 tests significant.", "yes", "Confirmed"),
    ("H3", "The boundary", "When only P(y|x) changes, no label-free signal can see the drift.",
     "Every signal sits at 0.50, proven equivalent to chance.", "yes", "Confirmed"),
    ("H4", "Explanations validate the alarm", "SHAP and LIME on EU point at the features that truly drifted.",
     "SHAP does (rank 0.78–0.92); LIME stays at chance.", "part", "SHAP yes, LIME no"),
    ("H5", "Bayesian vs frequentist", "Posterior uncertainty and bootstrap uncertainty behave differently.",
     "The ratio wins for neural networks and fails for the SVM and the random forest.", "info", "Model-dependent"),
]


def hypotheses_html() -> str:
    rows = []
    for n, name, claim, result, cls, verdict in HYPOTHESES:
        rows.append(f'<div><div class="n">{n}</div><div class="t"><b>{html.escape(name)}.</b> {html.escape(claim)}'
                    f'<span>{html.escape(result)}</span></div><div><span class="chip {cls}">{verdict}</span></div></div>')
    return f'<div class="bdx-hyp">{"".join(rows)}</div>'


PIPELINE_DOT = f"""
digraph G {{ rankdir=LR; bgcolor="transparent"; nodesep=0.28; ranksep=0.42;
node [shape=box, style="rounded,filled", fillcolor="{SURFACE}", color="#CFC8E0", penwidth=1.3,
      fontname="Atkinson Hyperlegible", fontsize=15, fontcolor="{INK}", margin="0.2,0.12"];
edge [color="#9C97B5", arrowsize=0.7, penwidth=1.2];
D [label="Real data\nwith checksum"];
I [label="Stream with\na known drift"];
M [label="Probabilistic\nmodel"];
U [label="Uncertainty split\nlog(EU/AU)", fillcolor="#FCE8EB", color="{CORAL}", penwidth=1.8];
A [label="Label-free\ndetectors"];
S [label="Error-based detectors\n(need labels)"];
X [label="Explain\nthe alarm"];
R [label="Statistics\nover all runs"];
D -> I -> M -> U -> A -> X -> R; M -> S -> R; A -> R; }}
"""


def style_fig(fig, height=None):
    """Apply the app's chart theme to a Plotly figure."""
    fig.update_layout(
        font=dict(family=FONT_BODY, size=14, color=INK),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        margin=dict(l=16, r=16, t=56, b=16),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=LINE, font=dict(family=FONT_BODY, size=13, color=INK)),
        legend=dict(orientation="h", y=1.04, x=0, yanchor="bottom", font=dict(size=13)),
    )
    if height:
        fig.update_layout(height=height)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=LINE, tickfont=dict(color=INK_SOFT))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=LINE, tickfont=dict(color=INK_SOFT))
    fig.update_annotations(font=dict(family=FONT_HEAD, size=15, color=INK))
    return fig
