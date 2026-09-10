"""Colour roles and figure defaults shared by every chart in the app.

One encoding runs through the whole interface: **blue means licit, red means
illicit, grey means unlabelled**, and continuous risk is a diverging blue -> grey
-> red ramp whose neutral point sits exactly on the operating threshold.  Because
the midpoint tracks the threshold, moving the slider visibly repaints the graph
around the new decision boundary instead of leaving the colours to be reinterpreted.

The two poles are the documented diverging pair and pass every colour-vision
separation check against both the light and the dark chart surface, so the
figures stay readable whichever Gradio theme the viewer is using.
"""

from __future__ import annotations

from typing import Final, Sequence

#: Diverging poles: licit (cool) and illicit (warm).
LICIT_BLUE: Final[str] = "#2a78d6"
ILLICIT_RED: Final[str] = "#e34948"
#: Neutral midpoint and the colour for nodes with no ground-truth label.
NEUTRAL_GREY: Final[str] = "#b9b8b2"
UNKNOWN_GREY: Final[str] = "#9a998f"

#: Text and chrome tones chosen to clear 3:1 against both chart surfaces, so a
#: single figure works in light and dark mode without being regenerated.
TEXT_MUTED: Final[str] = "#7d7c78"
GRID_LINE: Final[str] = "rgba(125,124,120,0.22)"
AXIS_LINE: Final[str] = "rgba(125,124,120,0.45)"
EDGE_LINE: Final[str] = "rgba(125,124,120,0.55)"

#: Accent used to ring the transaction under investigation.
FOCUS_RING: Final[str] = "#eda100"

#: Diverging ramp for risk. Plotly consumes it as (position, colour) stops.
RISK_COLORSCALE: Final[Sequence[tuple[float, str]]] = (
    (0.00, LICIT_BLUE),
    (0.50, NEUTRAL_GREY),
    (1.00, ILLICIT_RED),
)

#: Single-hue sequential ramp for magnitude-only charts (counts, confusion cells).
SEQUENTIAL_BLUE: Final[Sequence[tuple[float, str]]] = (
    (0.00, "#cde2fb"),
    (0.25, "#86b6ef"),
    (0.50, "#3987e5"),
    (0.75, "#256abf"),
    (1.00, "#0d366b"),
)

#: Risk bands, ordered from most to least severe. Each ships with a label, so
#: severity is never communicated by colour alone.
RISK_BAND_COLOURS: Final[dict[str, str]] = {
    "critical": "#e34948",
    "elevated": "#eb6834",
    "watch": "#eda100",
    "low": LICIT_BLUE,
}

FONT_FAMILY: Final[str] = (
    "ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
)


def base_layout(title: str = "", *, height: int = 380) -> dict:
    """Return Plotly layout defaults: transparent surface, recessive chrome.

    Backgrounds are transparent and every text tone is theme-neutral so one
    generated figure is legible on Gradio's light and dark surfaces alike.
    """
    return {
        "title": {
            "text": title,
            "font": {"size": 15, "color": TEXT_MUTED, "family": FONT_FAMILY},
            "x": 0.0,
            "xanchor": "left",
        },
        "height": height,
        "margin": {"l": 56, "r": 24, "t": 44 if title else 16, "b": 44},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": FONT_FAMILY, "size": 12, "color": TEXT_MUTED},
        "hoverlabel": {"font": {"family": FONT_FAMILY, "size": 12}},
        "showlegend": False,
    }


def axis_style(title: str = "", **overrides: object) -> dict:
    """Return recessive axis styling with an optional title."""
    style: dict = {
        "title": {"text": title, "font": {"size": 12, "color": TEXT_MUTED}},
        "gridcolor": GRID_LINE,
        "linecolor": AXIS_LINE,
        "zerolinecolor": GRID_LINE,
        "tickfont": {"size": 11, "color": TEXT_MUTED},
        "showgrid": True,
    }
    style.update(overrides)
    return style


def risk_colour(probability: float, threshold: float) -> str:
    """Map a probability to a discrete colour, pivoting on the threshold."""
    if probability >= threshold:
        return ILLICIT_RED
    if probability >= threshold / 2:
        return RISK_BAND_COLOURS["watch"]
    return LICIT_BLUE


def normalise_risk(probability: float, threshold: float) -> float:
    """Rescale a probability so that ``threshold`` maps to the ramp midpoint.

    Anchoring the neutral colour to the live threshold is what makes the risk
    ramp mean the same thing at every operating point.
    """
    threshold = min(max(threshold, 1e-6), 1 - 1e-6)
    if probability <= threshold:
        return 0.5 * probability / threshold
    return 0.5 + 0.5 * (probability - threshold) / (1.0 - threshold)
