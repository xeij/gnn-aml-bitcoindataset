"""Small HTML fragments rendered inside the Gradio layout.

These are written against neutral, translucent surfaces rather than fixed
background colours so the same markup is legible under Gradio's light and dark
themes without being regenerated.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

from ..engine.inference import NodeAssessment, SubgraphExplanation
from .theme import RISK_BAND_COLOURS, TEXT_MUTED

_CARD_STYLE = (
    "border:1px solid rgba(125,124,120,0.32);border-radius:10px;"
    "padding:14px 16px;background:rgba(125,124,120,0.07);"
)
_LABEL_STYLE = f"font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:{TEXT_MUTED};"


def _chip(text: str, colour: str) -> str:
    """Return a coloured pill carrying a text label."""
    return (
        f"<span style='display:inline-block;padding:3px 10px;border-radius:999px;"
        f"background:{colour};color:#ffffff;font-size:12px;font-weight:600;"
        f"letter-spacing:.02em;'>{text}</span>"
    )


def _field(label: str, value: str) -> str:
    """Return one label/value pair for the verdict card grid."""
    return (
        f"<div><div style='{_LABEL_STYLE}'>{label}</div>"
        f"<div style='font-size:14px;font-weight:600;margin-top:2px;'>{value}</div></div>"
    )


def verdict_card(
    assessment: NodeAssessment, explanation: Optional[SubgraphExplanation] = None
) -> str:
    """Render the model's verdict on one transaction as an HTML card."""
    band = assessment.risk_band
    colour = RISK_BAND_COLOURS.get(band, TEXT_MUTED)
    probability = assessment.illicit_probability

    correctness = ""
    if assessment.is_correct is None:
        correctness = (
            "<span style='font-size:12px;color:" + TEXT_MUTED + ";'>"
            "no ground-truth label — this is a genuine prediction</span>"
        )
    else:
        verdict = "agrees with" if assessment.is_correct else "contradicts"
        correctness = (
            f"<span style='font-size:12px;color:{TEXT_MUTED};'>prediction {verdict} "
            f"the ground-truth label</span>"
        )

    neighbourhood = ""
    if explanation is not None:
        neighbourhood = _field(
            "neighbourhood",
            f"{explanation.num_flagged} of {explanation.num_nodes} above threshold",
        )

    return f"""
<div style="{_CARD_STYLE}">
  <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
    {_chip(band.upper(), colour)}
    <span style="font-size:20px;font-weight:700;">P(illicit) = {probability:.4f}</span>
    <span style="font-size:12px;color:{TEXT_MUTED};">
      threshold {assessment.threshold:.3f}</span>
  </div>
  <div style="margin-top:6px;">{correctness}</div>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));
              gap:12px;margin-top:14px;">
    {_field("transaction id", str(assessment.tx_id))}
    {_field("node index", str(assessment.node_index))}
    {_field("time step", str(assessment.time_step))}
    {_field("degree", str(assessment.degree))}
    {_field("predicted", assessment.predicted_name)}
    {_field("ground truth", assessment.true_name)}
    {neighbourhood}
  </div>
</div>
""".strip()


def status_banner(message: str, *, tone: str = "info") -> str:
    """Render a short status message in a translucent banner."""
    colours = {"info": "#2a78d6", "warn": "#eda100", "error": "#e34948", "ok": "#1baf7a"}
    accent = colours.get(tone, colours["info"])
    return (
        f"<div style='border-left:3px solid {accent};padding:8px 12px;"
        f"background:rgba(125,124,120,0.09);border-radius:0 8px 8px 0;font-size:13px;'>"
        f"{message}</div>"
    )


def metric_headline(pairs: Sequence[tuple[str, str]]) -> str:
    """Render a compact KPI row of label/value pairs."""
    cells: Iterable[str] = (
        f"<div style='{_CARD_STYLE}min-width:130px;flex:1;'>"
        f"<div style='{_LABEL_STYLE}'>{label}</div>"
        f"<div style='font-size:22px;font-weight:700;margin-top:4px;'>{value}</div></div>"
        for label, value in pairs
    )
    return (
        "<div style='display:flex;gap:10px;flex-wrap:wrap;'>" + "".join(cells) + "</div>"
    )
