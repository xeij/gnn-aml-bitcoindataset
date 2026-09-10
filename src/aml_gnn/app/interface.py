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
from typing import Any, Optional

import gradio as gr

from ..data.splits import ILLICIT_LABEL, UNKNOWN_LABEL
from ..engine.metrics import (
    METRIC_TABLE_HEADERS,
    format_metric_table,
    precision_recall_points,
)
from ..logging_utils import get_logger
from . import visualisations as viz
from .components import metric_headline, status_banner, verdict_card
from .state import DEFAULT_CHECKPOINT_DIR, AppState

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


def build_interface(
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
    *,
    default_checkpoint: Optional[str] = None,
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

    def on_investigate(query: str, mode: str, hops: int) -> tuple[Any, ...]:
        """Score one transaction and render its neighbourhood."""
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

        figure = viz.subgraph_figure(explanation, threshold=service.threshold)
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

    def on_refresh_performance() -> tuple[Any, ...]:
        """Recompute every held-out evaluation view."""
        if not state.is_ready:
            placeholder = viz.empty_figure("Load a model to begin.")
            return ([], placeholder, placeholder, placeholder, placeholder, "")
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
            ("Test illicit F1", f"{test.illicit_f1:.4f}"),
            ("Precision", f"{test.illicit_precision:.4f}"),
            ("Recall", f"{test.illicit_recall:.4f}"),
            ("Average precision", f"{test.average_precision:.4f}"),
            ("Alerts raised", f"{test.alerts_raised:,}"),
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

    with gr.Blocks(title=APP_TITLE) as demo:
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

        with gr.Tabs():
            with gr.Tab("Case view"):
                with gr.Row():
                    with gr.Column(scale=1):
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
                    with gr.Column(scale=2):
                        verdict = gr.HTML(
                            status_banner("Enter a transaction to begin.", tone="info")
                        )
                subgraph_plot = gr.Plot(label="Transaction neighbourhood")
                with gr.Row():
                    attribution_plot = gr.Plot(label="Feature attribution")
                    neighbour_table = gr.Dataframe(
                        headers=["tx id", "node", "time step", "P(illicit)", "ground truth"],
                        label="Riskiest neighbours", interactive=False, wrap=True,
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
                metrics_table = gr.Dataframe(
                    headers=list(METRIC_TABLE_HEADERS), label="Per-split metrics",
                    interactive=False,
                )
                with gr.Row():
                    pr_plot = gr.Plot(label="Precision / recall")
                    confusion_plot = gr.Plot(label="Confusion matrix")
                temporal_plot = gr.Plot(label="Detection quality over time")
                volume_plot = gr.Plot(label="Alert volume")

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

        investigate_inputs = [query_box, query_mode, hops_slider]
        investigate_outputs = [verdict, subgraph_plot, attribution_plot, neighbour_table]
        investigate_button.click(
            on_investigate, inputs=investigate_inputs, outputs=investigate_outputs
        )
        query_box.submit(
            on_investigate, inputs=investigate_inputs, outputs=investigate_outputs
        )
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
            ],
        )
        dataset_button.click(
            on_refresh_dataset,
            outputs=[dataset_table, balance_plot, histogram_plot, embedding_plot],
        )

    return demo
