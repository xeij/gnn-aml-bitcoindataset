"""The Gradio investigation interface.

The app is organised the way an analyst actually works: a ranked **queue** of
alerts, a **case view** that explains one transaction in the context of its
neighbourhood, and the **evidence** needed to trust the queue at all -- held-out
metrics, temporal drift, and a look at the learned representation.

All model access goes through :class:`~aml_gnn.app.state.AppState`; these
callbacks only marshal values between that object and the widgets.  The app is
single-analyst local tooling, so one shared state object is intentional -- a
multi-user deployment would move it into ``gr.State``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr

from ..data.splits import ILLICIT_LABEL, UNKNOWN_LABEL
from ..engine.calibration import expected_calibration_error
from ..engine.metrics import (
    METRIC_TABLE_HEADERS,
    format_metric_table,
    precision_recall_points,
)
from ..logging_utils import get_logger
from . import atlas, explorer
from . import visualisations as viz
from .components import (
    metric_headline,
    metric_tiles,
    section_title,
    status_banner,
    verdict_card,
)
from .state import DEFAULT_CHECKPOINT_DIR, AppState
from .theme import APP_CSS

logger = get_logger(__name__)

APP_TITLE = "Bitcoin AML — GNN triage"

#: Gradio 6 takes the theme at ``launch()`` rather than on ``Blocks``.
APP_THEME = gr.themes.Soft()

_ALERT_HEADERS = [
    "tx id", "node", "time step", "P(illicit)", "band", "predicted", "ground truth", "degree",
]

_INTRO = """
# Bitcoin AML — graph neural network triage

Illicit-transaction detection on the **Elliptic** dataset: 203,769 Bitcoin
transactions, 234,355 payment flows, 49 time steps. The model is trained
transductively on the earliest time steps and evaluated strictly on later,
unseen ones — no random split, because transactions within a window are densely
interlinked and a random split would leak neighbours across the boundary.
"""

_NO_MODEL = """
### No trained model found

Train one first:

```bash
python scripts/train.py --config configs/gat.yaml
```

Then restart this app, or press **Rescan checkpoints**.
"""


#: Gradio's ``gr.Plot`` exposes only a ``change`` event -- there is no server-side
#: click on a Plotly figure -- so selection is bridged in the browser.  This runs
#: once on page load (Gradio 6 moved ``head`` off the Blocks constructor, and a
#: ``<script>`` inside ``gr.HTML`` is never executed because it arrives through
#: ``innerHTML``).  It watches for Plotly divs inside the plots we tag, binds
#: ``plotly_click`` to each, and writes the clicked node index into a hidden
#: textbox whose ``change`` event Python does listen to.  Rebinding is
#: observer-driven because every redraw replaces the div.
_SELECT_BRIDGE_JS: str = """
() => {
  // Each watched plot writes to its own hidden relay, because a click means
  // different things in different views: open a case, or explore onward.
  const TARGETS = {
    "subgraph-plot": "node-pick",
    "atlas-plot": "node-pick",
    "explorer-plot": "explorer-pick",
  };

  const relay = (target, nodeIndex) => {
    const host = document.getElementById(target);
    if (!host) return;
    const field = host.querySelector("textarea, input");
    if (!field) return;
    // Svelte binds on the native input event, so set through the prototype
    // setter and dispatch: assigning .value alone does not notify it.
    const setter = Object.getOwnPropertyDescriptor(
      field.constructor.prototype, "value"
    ).set;
    // A repeat click on the same node must still register, so the relayed
    // value is salted: Gradio only fires change when the text differs.
    setter.call(field, String(nodeIndex) + ":" + Date.now());
    field.dispatchEvent(new Event("input", { bubbles: true }));
  };

  const bind = (div, target) => {
    if (!div || div.dataset.amlBound === "1" || typeof div.on !== "function") return;
    div.dataset.amlBound = "1";
    div.on("plotly_click", (event) => {
      const point = event && event.points && event.points[0];
      if (!point) return;
      const payload = point.customdata;
      if (payload === undefined || payload === null) return;
      relay(target, Array.isArray(payload) ? payload[0] : payload);
    });
  };

  const scan = () => {
    for (const [id, target] of Object.entries(TARGETS)) {
      const host = document.getElementById(id);
      if (host) {
        host.querySelectorAll(".js-plotly-plot").forEach((div) => bind(div, target));
      }
    }
  };

  new MutationObserver(scan).observe(document.documentElement, {
    childList: true, subtree: true,
  });
  scan();
}
"""



def build_interface(
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
    *,
    default_checkpoint: str | None = None,
) -> gr.Blocks:
    """Construct the Gradio application.

    Args:
        checkpoint_dir: Directory scanned for trained checkpoints.
        default_checkpoint: Checkpoint to activate on startup.

    Returns:
        A :class:`gr.Blocks` app, ready to ``launch()``.

    """
    state = AppState(checkpoint_dir, default_checkpoint=default_checkpoint)

    # ------------------------------------------------------------ callbacks

    def model_summary() -> str:
        """Return the KPI row describing the active model."""
        if not state.is_ready:
            return status_banner("No model loaded.", tone="warn")
        return metric_headline(list(state.service.describe()))

    def on_model_change(name: str) -> tuple[Any, ...]:
        """Activate a checkpoint and refresh the header widgets."""
        if not name:
            return model_summary(), gr.update(), status_banner("Select a checkpoint.", tone="warn")
        try:
            service = state.activate(name)
        except Exception as error:  # Surface load failures in the UI, not just logs.
            logger.exception("Failed to activate %r", name)
            return model_summary(), gr.update(), status_banner(str(error), tone="error")
        return (
            model_summary(),
            gr.update(value=round(service.threshold, 3)),
            status_banner(
                f"Loaded <b>{name}</b>. Operating threshold restored to "
                f"{service.threshold:.3f}, the value tuned on the validation window.",
                tone="ok",
            ),
        )

    def on_refresh_checkpoints() -> tuple[Any, Any]:
        """Rescan the checkpoint directory."""
        names = state.refresh()
        if not names:
            return gr.update(choices=[], value=None), status_banner(
                f"No checkpoints in {checkpoint_dir}.", tone="warn"
            )
        return (
            gr.update(choices=names, value=state.active_name or names[0]),
            status_banner(f"Found {len(names)} checkpoint(s).", tone="ok"),
        )

    def on_threshold_change(threshold: float) -> str:
        """Apply a new operating threshold to the active model."""
        if not state.is_ready:
            return status_banner("No model loaded.", tone="warn")
        state.service.set_threshold(float(threshold))
        metrics = state.service.evaluate_split("test")
        return status_banner(
            f"At threshold {threshold:.3f} the model raises "
            f"<b>{metrics.alerts_raised:,}</b> alerts on the test window, of which "
            f"<b>{metrics.true_positives:,}</b> are truly illicit "
            f"(precision {metrics.illicit_precision:.3f}, "
            f"recall {metrics.illicit_recall:.3f}).",
            tone="info",
        )

    # ------------------------------------------------------- investigation

    def _resolve_node(query: str, mode: str) -> int:
        """Translate a UI query into a node index."""
        text = (query or "").strip()
        if not text:
            raise ValueError("Enter a transaction id or node index.")
        try:
            value = int(text)
        except ValueError as error:
            raise ValueError(f"'{text}' is not an integer.") from error
        if mode.startswith("Node"):
            return value
        index = state.service.data_module.index_for_tx_id(value)
        if index is None:
            raise ValueError(f"Transaction {value} is not in the Elliptic dataset.")
        return index

    def on_investigate(query: str, mode: str, hops: int, dimensions: str) -> tuple[Any, ...]:
        """Score one transaction and render its neighbourhood in 2-D or 3-D."""
        if not state.is_ready:
            return (
                status_banner("No model loaded.", tone="warn"),
                viz.empty_figure("Load a model to begin."),
                viz.empty_figure("Load a model to begin.", height=320),
                [],
            )
        try:
            node_index = _resolve_node(query, mode)
            service = state.service
            explanation = service.explain(node_index, num_hops=int(hops))
        except (ValueError, IndexError, KeyError) as error:
            return (
                status_banner(str(error), tone="error"),
                viz.empty_figure("Nothing to display."),
                viz.empty_figure("Nothing to display.", height=320),
                [],
            )

        render = (
            viz.subgraph_figure_3d if dimensions.startswith("3") else viz.subgraph_figure
        )
        figure = render(explanation, threshold=service.threshold)
        attribution = viz.feature_attribution_figure(explanation.feature_attribution)
        return (
            verdict_card(explanation.center, explanation),
            figure,
            attribution,
            _neighbour_rows(explanation),
        )

    def _neighbour_rows(explanation: Any, limit: int = 20) -> list[list[str]]:
        """Return the riskiest neighbours as table rows."""
        probabilities = explanation.illicit_probability
        order = probabilities.argsort(descending=True)[:limit]
        rows: list[list[str]] = []
        for position in order.tolist():
            node = int(explanation.node_indices[position])
            if node == explanation.center.node_index:
                continue
            label = int(explanation.labels[position])
            rows.append([
                str(int(explanation.tx_ids[position])),
                str(node),
                str(int(explanation.time_steps[position])),
                f"{float(probabilities[position]):.4f}",
                "unknown" if label == UNKNOWN_LABEL else ("illicit" if label else "licit"),
            ])
        return rows

    def on_node_picked(picked: str) -> tuple[str, str]:
        """Translate a browser-side node click into a populated query box.

        The bridge salts each relayed value with a timestamp so that clicking the
        same node twice still registers as a change, so only the leading index is
        meaningful here.
        """
        text = (picked or "").split(":")[0].strip()
        if not text:
            return "", "Node index"
        logger.info("Node %s selected from a figure", text)
        return text, "Node index"

    def on_random_case(band: str) -> str:
        """Populate the query box with a node matching a requested profile."""
        if not state.is_ready:
            return ""
        service = state.service
        if band == "Highest risk in test window":
            alerts = service.top_alerts(split="test", limit=1)
            return str(alerts[0].node_index) if alerts else ""
        if band == "Known illicit (test)":
            indices = service.data_module.sample_node_indices(
                label=ILLICIT_LABEL, split="test", limit=1
            )
            return str(indices[0]) if indices else ""
        indices = service.data_module.sample_node_indices(split="test", limit=1)
        return str(indices[0]) if indices else ""

    # ------------------------------------------------------------- queue

    def on_load_alerts(split: str, limit: int, include_unlabelled: bool) -> list[list[str]]:
        """Return the ranked alert queue as table rows."""
        if not state.is_ready:
            return []
        alerts = state.service.top_alerts(
            split=split, limit=int(limit), include_unlabelled=include_unlabelled
        )
        return [
            [
                str(alert.tx_id), str(alert.node_index), str(alert.time_step),
                f"{alert.illicit_probability:.4f}", alert.risk_band,
                alert.predicted_name, alert.true_name, str(alert.degree),
            ]
            for alert in alerts
        ]

    def on_alert_selected(event: gr.SelectData, table: Any) -> tuple[str, str]:
        """Send a clicked queue row to the case view."""
        try:
            row = event.index[0]
            values = table.values.tolist() if hasattr(table, "values") else table
            return str(values[row][1]), "Node index"
        except Exception:  # pragma: no cover - defensive against table shape drift
            logger.exception("Could not resolve the selected alert row")
            return "", "Node index"

    # ------------------------------------------------------- performance

    def _triage_tiles(test: Any, prevalence: float, service: Any) -> str:
        """Build the tiles that say what the model is worth to an analyst."""
        reviewed = test.alerts_raised / max(test.num_nodes, 1)
        lift = test.alert_precision / prevalence if prevalence else 0.0
        caught = test.illicit_recall
        return metric_tiles([
            (
                "queue size",
                f"{test.alerts_raised:,}",
                f"of {test.num_nodes:,} transactions — an analyst opens "
                f"{reviewed:.1%} of the window",
            ),
            (
                "hit rate",
                f"{test.alert_precision:.1%}",
                f"of opened alerts are real, against a {prevalence:.1%} base rate "
                f"— {lift:.1f}× lift",
            ),
            (
                "caught",
                f"{caught:.1%}",
                f"of all illicit transactions in the window "
                f"({test.true_positives:,} of "
                f"{test.true_positives + test.false_negatives:,})",
            ),
            (
                "missed",
                f"{test.false_negatives:,}",
                f"illicit transactions no alert was raised for, plus "
                f"{test.false_positives:,} false alarms",
            ),
        ])

    def _calibration_tiles(service: Any, test: Any) -> str:
        """Build the tiles describing how much a reported probability is worth."""
        probabilities = service.probabilities
        labels = service.data.y.cpu()
        mask = service.data.test_mask.cpu()
        error = expected_calibration_error(probabilities[mask], labels[mask])
        confident = probabilities >= 0.99
        confident_labelled = confident & mask & (labels != UNKNOWN_LABEL)
        if confident_labelled.any():
            truly = float((labels[confident_labelled] == ILLICIT_LABEL).float().mean())
            claim = f"{truly:.0%} of them really are illicit"
        else:
            claim = "none in the labelled test window"
        saturated = int((probabilities >= 0.9999).sum())
        return metric_tiles([
            (
                "temperature",
                f"{service.temperature:.2f}",
                "fitted on validation; 1.00 would mean the raw logits were "
                "already calibrated",
            ),
            (
                "calibration error",
                f"{error:.4f}",
                "mean gap between claimed probability and observed rate, on test",
            ),
            (
                "says ≥ 0.99",
                f"{int((confident & mask).sum()):,}",
                f"transactions in the test window — {claim}",
            ),
            (
                "pinned at 1.0",
                f"{saturated:,}",
                "of 203,769 nodes still saturate the float32 softmax",
            ),
        ])

    def _drift_tiles(test: Any, shutdown: int = 43) -> str:
        """Build the tiles that split the test window at the marketplace shutdown."""
        per_step = test.per_time_step
        before = [f for step, f in per_step.items() if step < shutdown]
        after = [f for step, f in per_step.items() if step >= shutdown]
        mean_before = sum(before) / len(before) if before else 0.0
        mean_after = sum(after) / len(after) if after else 0.0
        return metric_tiles([
            (
                f"F1 before step {shutdown}",
                f"{mean_before:.3f}",
                f"mean per-step F1 across the {len(before)} steps preceding the "
                "marketplace seizure",
            ),
            (
                f"F1 from step {shutdown}",
                f"{mean_after:.3f}",
                f"across the {len(after)} steps after it — the headline number "
                "averages these two regimes together",
            ),
            (
                "collapse",
                f"−{(mean_before - mean_after):.3f}",
                "the drop a single averaged test metric would hide entirely",
            ),
        ])

    def on_refresh_performance() -> tuple[Any, ...]:
        """Recompute every held-out evaluation view."""
        if not state.is_ready:
            placeholder = viz.empty_figure("Load a model to begin.")
            return ([], placeholder, placeholder, placeholder, placeholder,
                    "", "", "", "")
        service = state.service
        metrics = [service.evaluate_split(split) for split in ("train", "val", "test")]
        test = metrics[-1]

        probabilities = service.probabilities
        mask = service.data.test_mask.cpu()
        recall, precision = precision_recall_points(
            service.data.y.cpu()[mask], probabilities[mask]
        )
        prevalence = float((service.data.y.cpu()[mask] == ILLICIT_LABEL).float().mean())

        steps, alerts, actual = service.temporal_alert_volume()
        headline = metric_headline([
            ("Illicit F1", f"{test.illicit_f1:.4f}"),
            ("Precision", f"{test.illicit_precision:.4f}"),
            ("Recall", f"{test.illicit_recall:.4f}"),
            ("Average precision", f"{test.average_precision:.4f}"),
            ("ROC-AUC", f"{test.roc_auc:.4f}"),
            ("Balanced accuracy", f"{test.balanced_accuracy:.4f}"),
        ])
        return (
            format_metric_table(metrics),
            viz.precision_recall_figure(
                recall, precision, baseline=prevalence, label=state.active_name or "model"
            ),
            viz.confusion_figure(test),
            viz.temporal_f1_figure(test.per_time_step),
            viz.alert_volume_figure(steps, alerts, actual),
            headline,
            _triage_tiles(test, prevalence, service),
            _calibration_tiles(service, test),
            _drift_tiles(test),
        )

    def on_render_atlas(view: str, budget: int, colour_by: str) -> Any:
        """Render one dataset-scale 3-D view at the requested point budget."""
        if not state.is_ready:
            return viz.empty_figure("Load a model to begin.", height=560)
        service = state.service
        data = service.data
        labels = data.y.cpu()
        tx_ids = data.tx_id.cpu()
        steps = data.time_step.cpu()
        degrees = service.degrees
        probabilities = service.probabilities

        if view.startswith("Layered"):
            return atlas.layered_structure_figure(
                data.edge_index.cpu(), steps, labels, probabilities, tx_ids, degrees
            )
        projection = service.embedding_pca(3)
        if view.startswith("Temporal"):
            return atlas.temporal_drift_figure(
                projection, labels, probabilities, tx_ids, steps, degrees,
                budget=int(budget),
            )
        return atlas.embedding_atlas_figure(
            projection, labels, probabilities, tx_ids, steps, degrees,
            budget=int(budget), colour_by=colour_by, threshold=service.threshold,
        )

    # ------------------------------------------------------- flow explorer

    def _explore(seed: int, hops: int, budget: int, keep: list[int]) -> tuple[Any, Any]:
        """Walk outward from ``seed`` and draw the result, keeping ``keep`` on screen."""
        service = state.service
        nodes, depths = service.expand(
            [seed], hops=int(hops), max_nodes=int(budget), keep=keep
        )
        edges = service.induced_edges(nodes)
        figure = explorer.flow_figure(
            nodes, depths, edges,
            service.probabilities[nodes],
            service.data.y.cpu()[nodes],
            service.data.tx_id.cpu()[nodes],
            service.degrees[nodes],
            threshold=service.threshold,
            seed_index=seed,
        )
        return figure, nodes.tolist()

    def on_explore(query: str, mode: str, hops: int, budget: int) -> tuple[Any, ...]:
        """Start a fresh walk from whatever the query box names."""
        if not state.is_ready:
            return viz.empty_figure("Load a model to begin.", height=560), [], ""
        try:
            seed = _resolve_node(query, mode)
        except (ValueError, IndexError, KeyError) as error:
            return viz.empty_figure(str(error), height=560), [], ""
        figure, nodes = _explore(seed, hops, budget, [])
        return figure, nodes, _explorer_status(len(nodes), int(hops), seed)

    def on_propagate(picked: str, hops: int, budget: int, visited: list[int]) -> tuple[Any, ...]:
        """Expand from a clicked node, adding to what is already on screen."""
        text = (picked or "").split(":")[0].strip()
        if not state.is_ready or not text:
            return viz.empty_figure("Nothing explored yet.", height=560), visited, ""
        seed = int(text)
        before = len(visited)
        # One ring per click keeps the walk legible; the slider governs how far
        # a fresh start reaches, not how far a single step does.
        figure, nodes = _explore(seed, 1, budget, visited)
        added = len(nodes) - before
        return figure, nodes, _explorer_status(len(nodes), int(hops), seed, added=added)

    def _explorer_status(total: int, hops: int, seed: int, *, added: int | None = None) -> str:
        """Describe the current walk in one line."""
        tx = int(state.service.data.tx_id.cpu()[seed])
        if added is None:
            return status_banner(
                f"Walked {hops} hop{'' if hops == 1 else 's'} from tx {tx} — "
                f"{total:,} transactions on screen. Click any node to expand from it.",
                tone="info",
            )
        if added == 0:
            return status_banner(
                f"tx {tx} has no counterparties that are not already on screen. "
                f"Still {total:,} transactions.",
                tone="warn",
            )
        return status_banner(
            f"Expanded tx {tx}: +{added:,} transactions, {total:,} on screen.",
            tone="info",
        )

    def on_refresh_dataset() -> tuple[Any, ...]:
        """Recompute the dataset overview views."""
        if not state.is_ready:
            placeholder = viz.empty_figure("Load a model to begin.")
            return ([], placeholder, placeholder, placeholder)
        service = state.service
        metadata = service.data_module.metadata
        edges, counts = service.risk_distribution()
        return (
            metadata.as_rows(),
            viz.class_balance_figure(metadata.splits),
            viz.risk_histogram_figure(edges, counts, threshold=service.threshold),
            viz.embedding_figure(
                service.embeddings, service.data.y.cpu(), service.probabilities
            ),
        )

    # ---------------------------------------------------------------- layout

    # Gradio's own dark theme still paints raised cards and heavy radii, so the
    # neutral base is taken and the surface is defined in one CSS block instead.
    shell = gr.themes.Base(
        font=["ui-sans-serif", "-apple-system", "Segoe UI", "sans-serif"],
    ).set(body_background_fill="#0d0d0d", body_text_color="#c3c2b7")

    with gr.Blocks(title=APP_TITLE, theme=shell, css=APP_CSS) as demo:
        gr.Markdown(_INTRO)

        if not state.available:
            gr.Markdown(_NO_MODEL)

        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=state.available, value=state.active_name,
                label="Trained checkpoint", scale=3,
            )
            threshold_slider = gr.Slider(
                minimum=0.01, maximum=0.99,
                value=round(state.service.threshold, 3) if state.is_ready else 0.5,
                step=0.01, label="Operating threshold", scale=4,
            )
            refresh_button = gr.Button("Rescan checkpoints", scale=1)

        model_info = gr.HTML(model_summary())
        status = gr.HTML(
            status_banner(
                "Threshold defaults to the value tuned on the validation window. "
                "Raising it trades recall for precision.",
                tone="info",
            )
        )

        # Written to by the browser-side click bridge, never by the user.
        node_pick = gr.Textbox(elem_id="node-pick", visible=False, label="picked node")
        explorer_pick = gr.Textbox(
            elem_id="explorer-pick", visible=False, label="expanded node"
        )
        # Everything currently drawn in the explorer, so a click can add to it.
        explorer_visited = gr.State([])

        with gr.Tabs():
            with gr.Tab("Case view"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=260):
                        query_box = gr.Textbox(
                            label="Transaction id or node index",
                            placeholder="e.g. 230425980",
                        )
                        query_mode = gr.Radio(
                            choices=["Transaction id", "Node index"],
                            value="Node index", label="Interpret input as",
                        )
                        hops_slider = gr.Slider(
                            minimum=1, maximum=3, value=2, step=1,
                            label="Neighbourhood hops",
                            info="Match the model's layer count to see its full receptive field.",
                        )
                        dimension_choice = gr.Radio(
                            choices=["3-D (rotatable)", "2-D"],
                            value="3-D (rotatable)", label="Layout",
                            info="Drag to orbit, scroll to zoom, right-drag to pan, "
                                 "click a node to investigate it.",
                        )
                        investigate_button = gr.Button("Investigate", variant="primary")
                        sample_choice = gr.Radio(
                            choices=[
                                "Highest risk in test window",
                                "Known illicit (test)",
                                "Random test transaction",
                            ],
                            value="Highest risk in test window",
                            label="Or pick an example",
                        )
                        sample_button = gr.Button("Load example")
                    # The controls are a fixed-width rail; everything else grows
                    # into the space the neighbourhood plot needs to be legible.
                    with gr.Column(scale=3, min_width=520):
                        verdict = gr.HTML(
                            status_banner("Enter a transaction to begin.", tone="info")
                        )
                        subgraph_plot = gr.Plot(
                            label="Transaction neighbourhood",
                            elem_id="subgraph-plot",
                            show_label=False,
                        )
                with gr.Row(equal_height=True):
                    attribution_plot = gr.Plot(label="Feature attribution", scale=2)
                    neighbour_table = gr.Dataframe(
                        headers=["tx id", "node", "time step", "P(illicit)", "ground truth"],
                        label="Riskiest neighbours", interactive=False, wrap=True,
                        scale=3, max_height=360,
                    )

            with gr.Tab("Alert queue"):
                gr.Markdown(
                    "Transactions ranked by illicit probability — the queue an "
                    "analyst would actually work. Select a row to open it in the case view."
                )
                with gr.Row():
                    split_choice = gr.Dropdown(
                        choices=["test", "val", "train"], value="test", label="Time window"
                    )
                    limit_slider = gr.Slider(
                        minimum=10, maximum=200, value=50, step=10, label="Queue length"
                    )
                    unlabelled_toggle = gr.Checkbox(
                        value=True, label="Include unlabelled transactions",
                        info="77% of the graph has no ground-truth label.",
                    )
                    load_alerts_button = gr.Button("Build queue", variant="primary")
                alert_table = gr.Dataframe(
                    headers=_ALERT_HEADERS, label="Alert queue", interactive=False, wrap=True
                )

            with gr.Tab("Model performance"):
                gr.Markdown(
                    "Every number here is computed on time steps the model never "
                    "trained on, at the threshold tuned on the validation window."
                )
                performance_button = gr.Button("Evaluate held-out splits", variant="primary")
                performance_headline = gr.HTML()

                gr.HTML(section_title(
                    "Triage economics",
                    "What the queue is worth to the person working it.",
                ))
                triage_panel = gr.HTML()

                gr.HTML(section_title(
                    "Calibration",
                    "Whether a reported probability can be taken at face value.",
                ))
                calibration_panel = gr.HTML()

                gr.HTML(section_title(
                    "Distribution shift",
                    "The test window holds two regimes; the headline averages them.",
                ))
                drift_panel = gr.HTML()

                gr.HTML(section_title("Per-split detail"))
                metrics_table = gr.Dataframe(
                    headers=list(METRIC_TABLE_HEADERS), label="Per-split metrics",
                    interactive=False,
                )
                with gr.Row(equal_height=True):
                    pr_plot = gr.Plot(label="Precision / recall")
                    confusion_plot = gr.Plot(label="Confusion matrix")
                temporal_plot = gr.Plot(label="Detection quality over time")
                volume_plot = gr.Plot(label="Alert volume")

            with gr.Tab("Flow explorer"):
                gr.Markdown(
                    "Follow the money outward. The seed transaction sits at the "
                    "centre and each shell is one hop further out, so distance from "
                    "the middle **is** hop distance. **Click any node to expand from "
                    "it** — the graph grows rather than resetting, so a walk "
                    "accumulates. Elliptic's components run 1,089 to 7,880 "
                    "transactions, so a deep enough walk reaches a whole day's graph."
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=260):
                        explorer_query = gr.Textbox(
                            label="Seed transaction id or node index",
                            placeholder="e.g. 198408",
                        )
                        explorer_mode = gr.Radio(
                            choices=["Node index", "Transaction id"],
                            value="Node index", label="Interpret input as",
                        )
                        explorer_hops = gr.Slider(
                            minimum=1, maximum=14, value=4, step=1,
                            label="Hops from the seed",
                            info="How far the initial walk reaches. Clicking a node "
                                 "afterwards always adds exactly one more ring.",
                        )
                        explorer_budget = gr.Slider(
                            minimum=200, maximum=8_000, value=2_500, step=100,
                            label="Node budget",
                            info="The walk stops here. 8,000 covers the largest "
                                 "component in the dataset.",
                        )
                        explore_button = gr.Button("Explore", variant="primary")
                        explorer_reset = gr.Button("Reset walk")
                        explorer_status = gr.HTML(
                            status_banner(
                                "Enter a seed transaction, or open one from the "
                                "case view.",
                                tone="info",
                            )
                        )
                    with gr.Column(scale=4, min_width=560):
                        explorer_plot = gr.Plot(
                            label="Transaction flow",
                            elem_id="explorer-plot",
                            show_label=False,
                        )

            with gr.Tab("Graph atlas"):
                gr.Markdown(
                    "The whole dataset at once. **Drag to orbit freely, scroll to "
                    "zoom, right-drag to pan, and click any point to open that "
                    "transaction in the case view.** The camera stays where you leave "
                    "it when the figure redraws. Rendering 203,769 nodes and 468,710 "
                    "edges in a browser is not possible, so each view states the "
                    "budget it drew within."
                )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=240):
                        atlas_view = gr.Radio(
                            choices=[
                                "Embedding atlas",
                                "Temporal drift",
                                "Layered structure",
                            ],
                            value="Embedding atlas", label="View",
                        )
                        atlas_colour = gr.Radio(
                            choices=["ground truth", "risk"],
                            value="ground truth", label="Colour by",
                            info="Ground truth checks the encoder; risk shows where "
                                 "the classifier fires, including on unlabelled nodes.",
                        )
                        atlas_budget = gr.Slider(
                            minimum=5_000, maximum=203_769,
                            value=atlas.DEFAULT_POINT_BUDGET,
                            step=5_000, label="Point budget",
                            info="Uniform sample, so the class balance shown is the "
                                 "real one. Above ~80,000 the browser feels it.",
                        )
                        atlas_button = gr.Button("Render view", variant="primary")
                    with gr.Column(scale=4, min_width=560):
                        atlas_plot = gr.Plot(
                            label="Dataset-scale view",
                            elem_id="atlas-plot",
                            show_label=False,
                        )

            with gr.Tab("Dataset"):
                dataset_button = gr.Button("Load dataset overview", variant="primary")
                dataset_table = gr.Dataframe(
                    headers=["Property", "Value"], label="Graph summary", interactive=False
                )
                with gr.Row():
                    balance_plot = gr.Plot(label="Class balance")
                    histogram_plot = gr.Plot(label="Risk distribution")
                embedding_plot = gr.Plot(label="Learned embedding space")

        # ------------------------------------------------------------ wiring

        model_dropdown.change(
            on_model_change, inputs=model_dropdown,
            outputs=[model_info, threshold_slider, status],
        )
        refresh_button.click(on_refresh_checkpoints, outputs=[model_dropdown, status])
        threshold_slider.release(on_threshold_change, inputs=threshold_slider, outputs=status)

        investigate_inputs = [query_box, query_mode, hops_slider, dimension_choice]
        investigate_outputs = [verdict, subgraph_plot, attribution_plot, neighbour_table]
        investigate_button.click(
            on_investigate, inputs=investigate_inputs, outputs=investigate_outputs
        )
        query_box.submit(
            on_investigate, inputs=investigate_inputs, outputs=investigate_outputs
        )
        dimension_choice.change(
            on_investigate, inputs=investigate_inputs, outputs=investigate_outputs
        )

        # A click anywhere in a 3-D scene arrives here as a node index. Clicking
        # a point in the atlas therefore opens that transaction as a case, which
        # is the only way to get from a dataset-scale view back to one record.
        node_pick.change(
            on_node_picked, inputs=node_pick, outputs=[query_box, query_mode]
        ).then(on_investigate, inputs=investigate_inputs, outputs=investigate_outputs)

        neighbour_table.select(
            on_alert_selected, inputs=neighbour_table, outputs=[query_box, query_mode]
        ).then(on_investigate, inputs=investigate_inputs, outputs=investigate_outputs)
        sample_button.click(on_random_case, inputs=sample_choice, outputs=query_box).then(
            lambda: "Node index", outputs=query_mode
        ).then(on_investigate, inputs=investigate_inputs, outputs=investigate_outputs)

        load_alerts_button.click(
            on_load_alerts,
            inputs=[split_choice, limit_slider, unlabelled_toggle],
            outputs=alert_table,
        )
        alert_table.select(
            on_alert_selected, inputs=alert_table, outputs=[query_box, query_mode]
        ).then(on_investigate, inputs=investigate_inputs, outputs=investigate_outputs)

        performance_button.click(
            on_refresh_performance,
            outputs=[
                metrics_table, pr_plot, confusion_plot, temporal_plot,
                volume_plot, performance_headline,
                triage_panel, calibration_panel, drift_panel,
            ],
        )
        # Install the click bridge once the page exists.
        demo.load(None, None, None, js=_SELECT_BRIDGE_JS)

        explorer_inputs = [explorer_query, explorer_mode, explorer_hops, explorer_budget]
        explorer_outputs = [explorer_plot, explorer_visited, explorer_status]
        explore_button.click(on_explore, inputs=explorer_inputs, outputs=explorer_outputs)
        explorer_query.submit(
            on_explore, inputs=explorer_inputs, outputs=explorer_outputs
        )
        explorer_reset.click(on_explore, inputs=explorer_inputs, outputs=explorer_outputs)
        explorer_pick.change(
            on_propagate,
            inputs=[explorer_pick, explorer_hops, explorer_budget, explorer_visited],
            outputs=explorer_outputs,
        )

        atlas_inputs = [atlas_view, atlas_budget, atlas_colour]
        atlas_button.click(on_render_atlas, inputs=atlas_inputs, outputs=atlas_plot)
        atlas_view.change(on_render_atlas, inputs=atlas_inputs, outputs=atlas_plot)
        atlas_colour.change(on_render_atlas, inputs=atlas_inputs, outputs=atlas_plot)

        dataset_button.click(
            on_refresh_dataset,
            outputs=[dataset_table, balance_plot, histogram_plot, embedding_plot],
        )

    blocks: gr.Blocks = demo
    return blocks
