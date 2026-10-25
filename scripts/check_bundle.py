"""Does the bundle reproduce the original v2 results? Run on 5 test brackets."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from surrogate.bundle import load_model_bundle

old = Path(sys.argv[1])
bundle = load_model_bundle("v2")
expected = pd.read_csv(old / "runs/v2/per_bracket.csv").set_index("model_id")

for model_id in [21, 29, 94, 95, 120]:
    f = np.load(old / "features" / f"{model_id}.npz")
    x  = torch.from_numpy(bundle.scalers.node.apply(f["node_features"]))
    e  = torch.from_numpy(bundle.scalers.edge.apply(f["edge_features"]))
    ei = torch.from_numpy(f["edge_index"].astype(np.int64))
    with torch.no_grad():
        y = bundle.model(x, ei, e).numpy()
    stress = np.expm1(bundle.scalers.target.undo(y))
    mae = np.abs(stress - f["stress_mpa"]).mean()
    row = expected.loc[model_id]
    print(f"{model_id:>4}  MAE {mae:8.2f} vs {row.mae_mpa:8.2f}   "
          f"peak {stress.max():9.1f} vs {row.peak_predicted_mpa:9.1f}")