"""Raw .vtk + .fem -> preprocess -> bundle model -> MPa must reproduce training."""
import numpy as np
import torch


def test_reproduces_training_result(bundle, processed, training_features, expected):
    _, features = processed
    x = torch.from_numpy(bundle.scalers.node.apply(features.node_features))
    e = torch.from_numpy(bundle.scalers.edge.apply(features.edge_features))
    ei = torch.from_numpy(features.edge_index)
    with torch.no_grad():
        y = bundle.model(x, ei, e).numpy()
    stress = np.expm1(bundle.scalers.target.undo(y))

    mae = np.abs(stress - training_features["stress_mpa"]).mean()
    assert abs(mae - expected["mae_mpa"]) < 0.01
    assert abs(stress.max() - expected["peak_predicted_mpa"]) < 0.1