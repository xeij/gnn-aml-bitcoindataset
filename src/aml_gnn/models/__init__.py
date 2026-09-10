"""Interchangeable node-classification architectures behind one interface."""

from .base import BaseNodeClassifier, ClassifierHead, ResidualProjection
from .baseline import MLP
from .gnn import GAT, GCN, GraphSAGE, MessagePassingClassifier
from .registry import available_models, build_model, get_model_class, register_model

__all__ = [
    "GAT",
    "GCN",
    "MLP",
    "BaseNodeClassifier",
    "ClassifierHead",
    "GraphSAGE",
    "MessagePassingClassifier",
    "ResidualProjection",
    "available_models",
    "build_model",
    "get_model_class",
    "register_model",
]
