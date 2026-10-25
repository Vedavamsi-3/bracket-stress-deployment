"""New CSV-free preprocessing must equal training's features."""
import numpy as np


def test_same_surface_nodes(processed, training_features):
    _, features = processed
    assert features.node_features.shape == training_features["node_features"].shape


def test_same_edges(processed, training_features):
    _, features = processed
    assert np.array_equal(features.edge_index,
                          training_features["edge_index"].astype(np.int64))


def test_same_node_features(processed, training_features):
    _, features = processed
    np.testing.assert_allclose(features.node_features,
                               training_features["node_features"], atol=1e-5)


def test_same_edge_features(processed, training_features):
    _, features = processed
    np.testing.assert_allclose(features.edge_features,
                               training_features["edge_features"], atol=1e-5)