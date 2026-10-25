"""Load a model bundle: weights + scaling stats + config, as one unit."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from surrogate.model import MeshGraphNet
from surrogate.scaling import Scalers

BUNDLE_ROOT = Path(__file__).resolve().parents[1] / "bundles"


@dataclass
class Bundle:
    model: MeshGraphNet
    scalers: Scalers
    reference_landmarks: np.ndarray
    info: dict


def load_model_bundle(name="v2", root=BUNDLE_ROOT):
    folder = Path(root) / name
    info = json.loads((folder / "bundle.json").read_text())
    model = MeshGraphNet(**info["architecture"])
    checkpoint = torch.load(folder / "best_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])     # the file holds {"model", "epoch", "val_loss"}
    model.eval()                                   # inference mode
    return Bundle(
        model=model,
        scalers=Scalers.load(folder / "scaling.npz"),
        reference_landmarks=np.load(folder / "reference_landmarks.npy"),
        info=info,
    )