# Bitcoin AML with graph neural networks

This project trains graph neural networks to flag illicit Bitcoin transactions in the [Elliptic dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) and serves the results through an investigation UI. The idea is simple: a transaction that looks ordinary on its own can still be suspicious because of who it is connected to, so the model looks at the neighbourhood, not just the row.

Stack: PyTorch, PyTorch Geometric, scikit-learn, Gradio, Plotly. Python 3.10 or newer.

## Quickstart

Needs about 800 MB of disk for the dataset. No API keys, no environment variables. The data is downloaded on the first training run.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .

python scripts/train.py --config configs/gat.yaml   # trains the default model
python scripts/launch_ui.py                          # opens http://127.0.0.1:7860
```

To reproduce the full comparison below:

```bash
scripts/run_benchmark.sh          # trains mlp, gcn, graphsage and gat in turn
python scripts/summarise_runs.py  # prints the results table
pytest                            # 95 tests, runs in a couple of seconds
```

## The dataset

Elliptic is a graph of 203,769 Bitcoin transactions joined by 234,355 payment edges, observed over 49 time steps of roughly two weeks each. Every transaction carries 165 anonymised features. Only 23% of transactions are labelled, and of those about one in ten is illicit. The rest are unknown and only contribute structure.

One property of the data drives most of the design: every edge connects two transactions in the same time step. The graph is really 49 separate islands. That makes a random train/test split a trap, because near-identical neighbours end up on both sides and the score looks far better than it is. This project splits by time instead.

| Split | Time steps | Labelled nodes | Illicit | Illicit share |
|---|---|---|---|---|
| Train | 1 to 30 | 26,905 | 2,954 | 11.0% |
| Validation | 31 to 34 | 2,989 | 508 | 17.0% |
| Test | 35 to 49 | 16,670 | 1,083 | 6.5% |

Nodes outside the training window are masked, not deleted, so message passing still sees the whole graph while the loss only touches training nodes.

## How it works

The pipeline is four stages, and each one lives in its own module under `src/aml_gnn`.

**Data** (`data/`). The PyTorch Geometric loader throws away the transaction ids and time steps after building its own masks. This project reads them back from the raw CSV, caches them, checks that the feature file and the class file are row-aligned, and rebuilds the three temporal splits. Features are standardised using statistics from the training window only. The edge list is made undirected.

**Models** (`models/`). Four architectures share one template: a stack of `conv, norm, ReLU, dropout` blocks with residual connections, jumping knowledge (the outputs of every layer are concatenated before classification), and a small MLP head.

- GAT: three layers of GATv2 attention with four heads. The default. Attention decides how much each neighbour matters, and the learned coefficients double as per-edge explanations in the UI.
- GraphSAGE: mean aggregation with a separate weight for the node's own features.
- GCN: degree-normalised averaging. Cheapest, and the weakest of the three graph models here.
- MLP: the same depth and width but ignores the edges completely. This is the control.

The MLP control matters more than it sounds. Elliptic's 165 features already include 72 hand-built neighbourhood aggregates, so a plain MLP is a real contender. Any claim that "the graph helped" has to be measured against it.

**Training** (`engine/trainer.py`). The whole graph fits in memory (about 135 MB in float32), so training is full-batch: every node sees its exact 3-hop neighbourhood and there is no sampling noise. Loss is class-weighted cross-entropy. Early stopping watches illicit F1 on the validation window, learning rate drops on plateau, gradients are clipped. The decision threshold is not 0.5. It is tuned on validation to maximise illicit F1, because in AML the operating point is a business choice about how many false alarms a team can absorb.

**Calibration** (`engine/calibration.py`). Cross-entropy keeps pushing logits apart long after the classes are separated. On this data the gap reached the thousands, so float32 softmax reported exactly 1.0 on transactions the model got wrong about a quarter of the time. A single temperature fitted on validation now divides the logits before the softmax. This cannot change the ranking, so F1, AUC and average precision are untouched, but the probabilities shown to an analyst now mean what they say. Expected calibration error on the test window dropped to 0.035 for the GAT.

**Serving** (`engine/inference.py` and `app/`). A single `InferenceService` loads a checkpoint, scores the graph once, and answers questions the UI asks: rank the riskiest transactions in a window, extract and explain a k-hop neighbourhood, walk the payment graph outward from a seed. Explanations come from two sources: GAT attention weights as edge importance, and an input-gradient saliency map over the centre node's features. The Gradio app has five tabs: a case view with a rotatable 3-D neighbourhood, a ranked alert queue, a model performance page with per-step drift, a flow explorer laid out by hop distance, and a dataset-scale atlas of the learned embedding.

## Results

All numbers are on the held-out test window (steps 35 to 49, 16,670 labelled transactions), for the illicit class, at the threshold tuned on validation. Single run, seed 42, no hyperparameter search.

| Model | Parameters | Train time | Precision | Recall | F1 | Avg precision | ROC AUC |
|---|---|---|---|---|---|---|---|
| GAT (GATv2) | 180,738 | 18 min | 0.662 | 0.592 | **0.625** | 0.561 | 0.906 |
| GraphSAGE | 179,586 | 3 min | 0.464 | 0.596 | 0.522 | 0.418 | 0.866 |
| GCN | 125,698 | 4 min | 0.451 | 0.507 | 0.478 | 0.342 | 0.877 |
| MLP (no graph) | 71,810 | 24 s | 0.254 | 0.594 | 0.356 | 0.181 | 0.819 |

A few things stand out.

**The graph is worth between 0.12 and 0.27 F1.** Every graph model beats the MLP control, and the attention model beats it by a wide margin. That is the honest estimate of what message passing adds on top of features that already contain neighbourhood summaries.

**Recall is nearly identical across models, precision is not.** All four catch roughly 59% of illicit transactions (the GCN a bit less). What separates them is how many false alarms come with that. The GAT reaches its recall with 327 false positives. The MLP needs 1,884 to get the same number of true positives. For an investigator that is the whole story.

**Attention earns its cost.** The GAT took six times longer to train than GraphSAGE for 0.10 more F1. Laundering structure is about which counterparty a transaction touches rather than how many, and a model that can weight neighbours unequally has the right bias for that.

**Validation scores are much higher than test scores.** The GAT hits 0.907 F1 on validation and 0.625 on test. This is not overfitting in the usual sense. The test window spans a real-world event, explained next.

### Performance collapses after step 43

The dominant illicit marketplace in the data was shut down around time step 43. After that, the illicit transactions that remain look nothing like the ones the model trained on. Per-step F1 computed from the saved GAT checkpoint:

| Steps | Labelled | Illicit | Illicit share | GAT F1 | GraphSAGE F1 | GCN F1 | MLP F1 |
|---|---|---|---|---|---|---|---|
| 35 to 42 | 9,983 | 914 | 9.2% | 0.707 | 0.606 | 0.569 | 0.432 |
| 43 to 49 | 6,687 | 169 | 2.5% | 0.024 | 0.022 | 0.005 | 0.009 |

In step 35 the GAT scores 0.936. From step 43 on it is close to zero on every model, and often exactly zero. The headline 0.625 is an average over two regimes: one where the model works well and one where it does not work at all. A random split would have hidden this completely. The performance tab in the UI plots the drop rather than smoothing it away.

This is the most important finding in the project. It says the model learned the signature of a particular criminal operation, and when that operation disappeared the signature went with it. Any deployment would need retraining on fresh labels as the landscape shifts.

## What this means in practice

Think of the model as a triage tool, not a verdict. At a 6.5% base rate an analyst who picks transactions at random opens about fifteen dead ends for every real case. The GAT turns the 16,670-transaction test window into a queue of 968 alerts, of which 641 are genuinely illicit. That is a 66% hit rate, a tenfold improvement over the base rate, and it catches 59% of all illicit activity in the window while reviewing under 6% of it.

The threshold slider in the UI moves that trade-off live. Lower it and recall goes up at the cost of more dead ends. Raise it and the queue gets shorter and cleaner. The precision/recall curve on the performance tab shows the full range.

The explanation tools are there so an analyst does not have to take the score on faith. The case view shows the 2-hop or 3-hop neighbourhood that produced the prediction, with edge thickness driven by attention weight and the riskiest neighbours listed alongside. The flow explorer answers "where does this money go" on a few thousand nodes at once.

## Limitations

- Everything comes from one run at one seed. The numbers describe where this configuration lands, not an upper bound.
- The model is transductive. Scoring a new transaction means re-running the whole graph, which takes about a second rather than milliseconds.
- The 165 features are anonymised. Feature attribution can name a dimension but not say what it measures.
- The post-step-43 collapse is not solved here. It is measured and shown.
- The MLP remains badly calibrated even after temperature scaling (expected calibration error 0.23), so its probabilities should not be read as probabilities.
- Validation is only four time steps with a 17% illicit rate, well above either train or test. Thresholds and temperatures tuned on it carry that skew.
