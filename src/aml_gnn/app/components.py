"""Small HTML fragments rendered inside the Gradio layout.

Written against the dark palette in :mod:`.theme`: one surface, one hairline
border, one corner radius, and type that stays light until a number needs to be
read at a glance.  Colour appears only where it carries meaning -- a risk band,
a status tone, a metric that has crossed a line worth noticing.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from ..engine.inference import NodeAssessment, SubgraphExplanation
from .theme import (
    BORDER,
    INK,
    INK_SECONDARY,
    MONO_FAMILY,
    RISK_BAND_COLOURS,
    STATUS_TONES,
    SURFACE_RAISED,
    TEXT_MUTED,
)

_CARD_STYLE = (
    f"border:1px solid {BORDER};"
    f"padding:14px 16px;background:{SURFACE_RAISED};"
)
_LABEL_STYLE = (
    f"font-size:10px;letter-spacing:.07em;text-transform:uppercase;"
    f"color:{TEXT_MUTED};font-weight:400;"
)
_VALUE_STYLE = f"font-size:14px;font-weight:500;margin-top:3px;color:{INK};"


def _chip(text: str, colour: str) -> str:
    """Return a pill carrying a text label, so the colour never stands alone."""
    return (
        f"<span style='display:inline-block;padding:3px 10px;"
        f"background:{colour};color:#0d0d0d;font-size:11px;font-weight:600;"
        f"letter-spacing:.06em;'>{text}</span>"
    )


def _field(label: str, value: str) -> str:
    """Return one label/value pair for the verdict card grid."""
    return (
        f"<div><div style='{_LABEL_STYLE}'>{label}</div>"
        f"<div style='{_VALUE_STYLE}'>{value}</div></div>"
    )


def verdict_card(
    assessment: NodeAssessment, explanation: SubgraphExplanation | None = None
) -> str:
    """Render the model's verdict on one transaction as an HTML card."""
    band = assessment.risk_band
    colour = RISK_BAND_COLOURS.get(band, TEXT_MUTED)
    probability = assessment.illicit_probability

    if assessment.is_correct is None:
        correctness = "no ground-truth label — this is a genuine prediction"
    else:
        verdict = "agrees with" if assessment.is_correct else "contradicts"
        correctness = f"prediction {verdict} the ground-truth label"

    neighbourhood = ""
    if explanation is not None:
        neighbourhood = _field(
            "neighbourhood",
            f"{explanation.num_flagged} of {explanation.num_nodes} above threshold",
        )

    return f"""
<div style="{_CARD_STYLE}">
  <div style="display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;">
    {_chip(band.upper(), colour)}
    <span style="font-size:28px;font-weight:400;color:{INK};font-family:{MONO_FAMILY};">
      {probability:.4f}</span>
    <span style="font-size:11px;color:{TEXT_MUTED};letter-spacing:.06em;
                 text-transform:uppercase;">P(illicit) · threshold
      {assessment.threshold:.3f}</span>
  </div>
  <div style="margin-top:8px;font-size:12px;color:{TEXT_MUTED};">{correctness}</div>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));
              gap:14px;margin-top:16px;">
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
    """Render a short status message behind a coloured rule."""
    accent = STATUS_TONES.get(tone, STATUS_TONES["info"])
    return (
        f"<div style='border-left:2px solid {accent};padding:8px 12px;"
        f"background:{SURFACE_RAISED};font-size:12px;"
        f"font-weight:300;color:{INK_SECONDARY};line-height:1.55;'>{message}</div>"
    )


def metric_headline(pairs: Sequence[tuple[str, str]]) -> str:
    """Render a compact KPI row of label/value pairs."""
    cells: Iterable[str] = (
        f"<div style='{_CARD_STYLE}min-width:120px;flex:1;'>"
        f"<div style='{_LABEL_STYLE}'>{label}</div>"
        f"<div style='font-size:22px;font-weight:400;margin-top:6px;color:{INK};"
        f"font-family:{MONO_FAMILY};'>{value}</div></div>"
        for label, value in pairs
    )
    return (
        "<div style='display:flex;gap:8px;flex-wrap:wrap;'>" + "".join(cells) + "</div>"
    )


def metric_tiles(tiles: Sequence[tuple[str, str, str]]) -> str:
    """Render KPI tiles that each carry a line of interpretation.

    A bare number invites the reader to supply their own baseline, which is how
    a 6.5% base rate gets mistaken for a good precision score.  Each tile states
    what its number should be compared against.

    Args:
        tiles: ``(label, value, note)`` triples; the note sits under the value.

    """
    cells: Iterable[str] = (
        f"<div style='{_CARD_STYLE}min-width:150px;flex:1;'>"
        f"<div style='{_LABEL_STYLE}'>{label}</div>"
        f"<div style='font-size:24px;font-weight:400;margin-top:6px;color:{INK};"
        f"font-family:{MONO_FAMILY};line-height:1.1;'>{value}</div>"
        f"<div style='font-size:11px;color:{TEXT_MUTED};margin-top:6px;"
        f"line-height:1.45;font-weight:300;'>{note}</div></div>"
        for label, value, note in tiles
    )
    return (
        "<div style='display:flex;gap:8px;flex-wrap:wrap;'>" + "".join(cells) + "</div>"
    )


def section_title(text: str, subtitle: str = "") -> str:
    """Render a quiet section heading above a group of panels."""
    tail = (
        f"<div style='font-size:12px;color:{TEXT_MUTED};margin-top:3px;"
        f"font-weight:300;'>{subtitle}</div>"
        if subtitle
        else ""
    )
    return (
        f"<div style='margin:4px 0 2px;'>"
        f"<div style='font-size:11px;letter-spacing:.09em;text-transform:uppercase;"
        f"color:{INK_SECONDARY};font-weight:400;'>{text}</div>{tail}</div>"
    )
