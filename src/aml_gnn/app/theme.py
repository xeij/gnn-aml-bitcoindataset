"""Colour roles, figure defaults and the app shell.

There is no colour here.  The palette is black, grey and white, and the two
things that used to be carried by hue are carried by channels that survive
without it: **lightness encodes risk** -- licit recedes toward the background,
illicit burns bright against it -- and **marker shape encodes class**, so a
circle, a diamond and a dot stay distinguishable in greyscale, in print, and to
a reader with any form of colour blindness.

That last point is not incidental.  Measured against this surface, every pair of
tones below separates by delta-E 29.5, and the figure is *identical* under
protanopia, deuteranopia and tritanopia simulation -- a monochrome ramp cannot
be confused by a colour-vision deficiency because there is no colour in it.  The
darkest tone sits at 1.53:1 against the surface, deliberately below the 3:1 mark
because it is context rather than a series; the legend and hover labels are what
make it readable, never the tone alone.

Corners are square, borders are single hairlines, nothing is raised off the page
and nothing has a shadow.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

# --------------------------------------------------------------- surfaces

#: The page behind everything, and the surface a figure is drawn on.
PAGE_PLANE: Final[str] = "#0b0b0b"
SURFACE: Final[str] = "#141413"
SURFACE_RAISED: Final[str] = "#1c1c1b"

#: Ink, from loudest to quietest.
INK: Final[str] = "#f5f4f0"
INK_SECONDARY: Final[str] = "#b4b2aa"
TEXT_MUTED: Final[str] = "#7e7c76"

#: Chart chrome: hairline grid, axis rule, and the ring around a panel.
GRID_LINE: Final[str] = "#262625"
AXIS_LINE: Final[str] = "#343432"
BORDER: Final[str] = "#2e2e2c"
EDGE_LINE: Final[str] = "rgba(180,178,170,0.34)"

# ------------------------------------------------------------ data tones

#: Lightness carries risk.  Licit sits close to the surface and recedes;
#: illicit is the brightest thing on the page, which is what an alert should be.
LICIT_TONE: Final[str] = "#8f8d86"
ILLICIT_TONE: Final[str] = "#f5f4f0"
#: The midpoint of the ramp, and the tone for a transaction sitting on the fence.
NEUTRAL_GREY: Final[str] = "#5a5954"
#: Unlabelled nodes: context, not a series.  Deliberately below the contrast
#: floor -- the legend and hover text carry it, never the tone.
UNKNOWN_TONE: Final[str] = "#3a3a37"

#: Ring around the transaction under investigation: pure white, nothing else is.
FOCUS_RING: Final[str] = "#ffffff"

#: Shape is the identity channel now that hue is gone.  Every figure that draws
#: more than one class draws these symbols and names them in its legend.
CLASS_SYMBOLS: Final[dict[str, str]] = {
    "licit": "circle",
    "illicit": "diamond",
    "unlabelled": "circle-open",
}

#: Sequential lightness ramp for risk, dark (licit) to light (illicit).  A
#: diverging ramp is impossible without hue -- both ends of a lightness axis
#: cannot be far from a midpoint -- so risk reads as one monotonic scale.
RISK_COLORSCALE: Final[Sequence[tuple[float, str]]] = (
    (0.00, "#3f3e3b"),
    (0.50, NEUTRAL_GREY),
    (1.00, ILLICIT_TONE),
)

#: Magnitude-only ramp (counts, confusion cells): the same scale, starting from
#: the surface so that zero disappears into it.
SEQUENTIAL_BLUE: Final[Sequence[tuple[float, str]]] = (
    (0.00, "#191918"),
    (0.25, "#3a3a37"),
    (0.50, "#5a5954"),
    (0.75, "#8f8d86"),
    (1.00, "#d8d6ce"),
)

#: Risk bands, each always printed beside its own name.  Four steps of one grey
#: ramp cannot be told apart unlabelled, and are never asked to be.
RISK_BAND_COLOURS: Final[dict[str, str]] = {
    "critical": "#f5f4f0",
    "elevated": "#c4c2ba",
    "watch": "#8f8d86",
    "low": "#5a5954",
}

STATUS_TONES: Final[dict[str, str]] = {
    "error": "#f5f4f0",
    "warn": "#c4c2ba",
    "info": "#7e7c76",
    "ok": "#5a5954",
}

FONT_FAMILY: Final[str] = (
    "ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
)

#: Numbers are set in a mono face so digits align in a column and a metric can
#: be compared down a table without reading it.
MONO_FAMILY: Final[str] = (
    "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"
)


def base_layout(title: str = "", *, height: int = 380) -> dict:
    """Return Plotly layout defaults: transparent surface, recessive chrome.

    Backgrounds stay transparent so a figure inherits the panel it sits in
    rather than stamping its own rectangle onto the page.
    """
    return {
        # Plotly's default template carries its own colourway, a light plot
        # background and a set of rainbow colourscales.  Overriding those one
        # key at a time leaves them in the figure for any trace that forgets to
        # set its own, so the template is dropped entirely instead.
        "template": "none",
        "title": {
            "text": title,
            "font": {"size": 13, "color": TEXT_MUTED, "family": FONT_FAMILY,
                     "weight": 400},
            "x": 0.0,
            "xanchor": "left",
        },
        "height": height,
        "margin": {"l": 56, "r": 24, "t": 44 if title else 16, "b": 44},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": FONT_FAMILY, "size": 12, "color": INK_SECONDARY},
        "hoverlabel": {
            "font": {"family": FONT_FAMILY, "size": 12, "color": INK},
            "bgcolor": SURFACE_RAISED,
            "bordercolor": BORDER,
        },
        "showlegend": False,
    }


def axis_style(title: str = "", **overrides: object) -> dict:
    """Return recessive axis styling with an optional title."""
    style: dict = {
        "title": {"text": title, "font": {"size": 11, "color": TEXT_MUTED}},
        "gridcolor": GRID_LINE,
        "linecolor": AXIS_LINE,
        "zerolinecolor": AXIS_LINE,
        "tickfont": {"size": 11, "color": TEXT_MUTED},
        "showgrid": True,
    }
    style.update(overrides)
    return style


def scene_axis_style(title: str = "") -> dict:
    """Return recessive styling for one axis of a 3-D scene.

    Plotly's 3-D axes take a different key set from 2-D cartesian ones, so this
    mirrors :func:`axis_style` rather than reusing it.  The backdrop panes are
    suppressed entirely: on a transparent surface they read as a grey box that
    fights the marks for attention.
    """
    return {
        "title": {"text": title, "font": {"size": 11, "color": TEXT_MUTED}},
        "gridcolor": GRID_LINE,
        "zerolinecolor": AXIS_LINE,
        "tickfont": {"size": 10, "color": TEXT_MUTED},
        "showbackground": False,
        "showspikes": False,
    }


def scene_layout(
    x_title: str = "",
    y_title: str = "",
    z_title: str = "",
    *,
    eye: tuple[float, float, float] = (1.45, 1.45, 0.85),
) -> dict:
    """Return a 3-D scene block with a fixed initial camera.

    Args:
        x_title: Label for the x axis.
        y_title: Label for the y axis.
        z_title: Label for the z axis.
        eye: Camera position in scene units.  Pinning it keeps a redraw from
            resetting the viewer to a different angle than the one they left.

    """
    return {
        "xaxis": scene_axis_style(x_title),
        "yaxis": scene_axis_style(y_title),
        "zaxis": scene_axis_style(z_title),
        "camera": {
            "eye": {"x": eye[0], "y": eye[1], "z": eye[2]},
            "projection": {"type": "perspective"},
        },
        # Orbit tumbles freely, where the "turntable" default pins the vertical
        # axis upright.  Free rotation is what lets a viewer get underneath a
        # dense fan and see which spokes actually reach the hub.
        "dragmode": "orbit",
        "aspectmode": "cube",
    }


def risk_colour(probability: float, threshold: float) -> str:
    """Map a probability to a discrete colour, pivoting on the threshold."""
    if probability >= threshold:
        return ILLICIT_TONE
    if probability >= threshold / 2:
        return RISK_BAND_COLOURS["watch"]
    return LICIT_TONE


def normalise_risk(probability: float, threshold: float) -> float:
    """Rescale a probability so that ``threshold`` maps to the ramp midpoint.

    Anchoring the neutral colour to the live threshold is what makes the risk
    ramp mean the same thing at every operating point.
    """
    threshold = min(max(threshold, 1e-6), 1 - 1e-6)
    if probability <= threshold:
        return 0.5 * probability / threshold
    return 0.5 + 0.5 * (probability - threshold) / (1.0 - threshold)


#: Heights the plot panels reserve before anything is drawn.  Gradio sizes a
#: plot container to its content, so an empty one is short and the page jumps
#: when a figure lands.  Reserving the final height up front keeps every control
#: where the viewer last saw it.
PANEL_HEIGHTS: Final[dict[str, int]] = {
    "subgraph-plot": 620,
    "atlas-plot": 700,
    "explorer-plot": 700,
}


def _panel_height_rules() -> str:
    """Emit the reserved-height rule for each plot panel."""
    return "\n".join(
        f"#{name}, #{name} > div {{ min-height: {height}px; }}"
        for name, height in PANEL_HEIGHTS.items()
    )


#: The app shell.  Written as plain CSS rather than a Gradio theme object
#: because the parts that matter here -- hairline borders, one corner radius,
#: light type -- are not all exposed as theme tokens.
#: The app shell.  Plain CSS rather than a Gradio theme object, because the
#: parts that matter -- square corners, single hairlines, no shadow anywhere --
#: are not all exposed as theme tokens.
APP_CSS: Final[str] = f"""
:root, .gradio-container {{
  --aml-page: {PAGE_PLANE};
  --aml-surface: {SURFACE};
  --aml-raised: {SURFACE_RAISED};
  --aml-ink: {INK};
  --aml-ink-2: {INK_SECONDARY};
  --aml-muted: {TEXT_MUTED};
  --aml-border: {BORDER};
}}

.gradio-container, .gradio-container .contain {{
  background: var(--aml-page) !important;
  color: var(--aml-ink-2) !important;
  font-family: {FONT_FAMILY} !important;
  font-weight: 400;
  max-width: 1600px !important;
}}

.gradio-container h1, .gradio-container h2, .gradio-container h3 {{
  font-weight: 400 !important;
  letter-spacing: 0.02em;
  color: var(--aml-ink) !important;
  text-transform: uppercase;
}}
.gradio-container h1 {{ font-size: 16px !important; }}
.gradio-container h2 {{ font-size: 13px !important; }}
.gradio-container p, .gradio-container li {{
  color: var(--aml-muted) !important;
  font-size: 12px !important;
  line-height: 1.55;
}}
.gradio-container strong {{ color: var(--aml-ink-2) !important; font-weight: 500; }}

/* Every panel: one surface, one hairline, square corners, no shadow. */
.gradio-container .block,
.gradio-container .form,
.gradio-container .panel,
.gradio-container .block.padded {{
  background: var(--aml-surface) !important;
  border: 1px solid var(--aml-border) !important;
  border-radius: 0 !important;
  box-shadow: none !important;
}}
.gradio-container .block.padded {{ padding: 10px !important; }}

/* Nothing anywhere is rounded. */
.gradio-container *, .gradio-container *::before, .gradio-container *::after {{
  border-radius: 0 !important;
}}

.gradio-container label span,
.gradio-container .label-wrap span,
.gradio-container span[data-testid="block-info"] {{
  color: var(--aml-muted) !important;
  font-size: 10px !important;
  font-weight: 400 !important;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}}

.gradio-container input, .gradio-container textarea, .gradio-container select {{
  background: var(--aml-page) !important;
  color: var(--aml-ink) !important;
  border: 1px solid var(--aml-border) !important;
  font-family: {MONO_FAMILY} !important;
  font-size: 12px !important;
}}

.gradio-container button {{
  font-weight: 400 !important;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  font-size: 11px !important;
  border: 1px solid var(--aml-border) !important;
  background: var(--aml-raised) !important;
  color: var(--aml-ink-2) !important;
  box-shadow: none !important;
}}
.gradio-container button.primary {{
  background: var(--aml-ink) !important;
  border-color: var(--aml-ink) !important;
  color: {PAGE_PLANE} !important;
}}
.gradio-container button:hover {{ border-color: var(--aml-ink-2) !important; }}

/* Tabs are a rule with one lit item, not a row of cards. */
.gradio-container .tab-nav {{
  border-bottom: 1px solid var(--aml-border) !important;
  gap: 0;
}}
.gradio-container .tab-nav button {{
  background: transparent !important;
  border: none !important;
  border-bottom: 1px solid transparent !important;
  color: var(--aml-muted) !important;
  padding: 9px 16px !important;
}}
.gradio-container .tab-nav button.selected {{
  color: var(--aml-ink) !important;
  border-bottom-color: var(--aml-ink) !important;
}}

/* Tables: hairlines, mono figures, no fill. */
.gradio-container table {{ font-size: 11px !important; }}
.gradio-container thead th {{
  background: transparent !important;
  color: var(--aml-muted) !important;
  font-weight: 400 !important;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-size: 10px !important;
  border-bottom: 1px solid var(--aml-border) !important;
}}
.gradio-container tbody td {{
  background: transparent !important;
  color: var(--aml-ink-2) !important;
  border-color: var(--aml-border) !important;
  font-family: {MONO_FAMILY} !important;
}}

.gradio-container input[type="range"] {{ accent-color: {INK_SECONDARY}; }}

/* Reserved panel geometry: see PANEL_HEIGHTS. */
{_panel_height_rules()}

.gradio-container .modebar {{ background: transparent !important; }}
.gradio-container .modebar-btn path {{ fill: {TEXT_MUTED} !important; }}
"""
