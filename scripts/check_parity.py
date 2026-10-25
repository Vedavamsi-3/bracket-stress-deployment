"""Parity: new CSV-free preprocessing == training features, for test brackets."""
import json, sys, time
from pathlib import Path
import numpy as np
from surrogate.preprocess import preprocess

old = Path(sys.argv[1])          # training repo's output folder
raw = Path(sys.argv[2])          # folder with Input_files/ (.fem) and Output_files/ (.vtk)
n = int(sys.argv[3]) if len(sys.argv) > 3 else 5

test_ids = json.loads((old / "split.json").read_text())["test"][:n]
all_ok = True
for model_id in test_ids:
    start = time.perf_counter()
    graph, new = preprocess(raw / "Output_files" / f"{model_id}.vtk",
                            raw / "Input_files" / f"{model_id}.fem")
    seconds = time.perf_counter() - start
    saved = np.load(old / "features" / f"{model_id}.npz")

    same_shape = new.node_features.shape == saved["node_features"].shape
    same_edges = same_shape and np.array_equal(new.edge_index, saved["edge_index"].astype(np.int64))
    node_diff = np.abs(new.node_features - saved["node_features"]).max() if same_shape else np.inf
    edge_diff = np.abs(new.edge_features - saved["edge_features"]).max() if same_edges else np.inf
    ok = same_edges and node_diff < 1e-4 and edge_diff < 1e-4
    all_ok &= ok
    print(f"{model_id:>4}  nodes {new.node_features.shape[0]:>6} vs {saved['node_features'].shape[0]:>6}  "
          f"edges same: {same_edges}  max diff node {node_diff:.1e} edge {edge_diff:.1e}  "
          f"{seconds:5.1f}s  {'PASS' if ok else 'FAIL'}")

print("\nPARITY", "PASSED" if all_ok else "FAILED")