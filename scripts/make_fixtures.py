"""Copy one small test bracket + its training reference into tests/fixtures/."""
import json, shutil, sys
from pathlib import Path
import pandas as pd

old, raw = Path(sys.argv[1]), Path(sys.argv[2])
bracket = int(sys.argv[3]) if len(sys.argv) > 3 else 448
out = Path("tests/fixtures")
out.mkdir(parents=True, exist_ok=True)

shutil.copy(raw / "Output_files" / f"{bracket}.vtk", out / f"{bracket}.vtk")
shutil.copy(raw / "Input_files" / f"{bracket}.fem", out / f"{bracket}.fem")
shutil.copy(old / "features" / f"{bracket}.npz", out / f"{bracket}_features.npz")

row = pd.read_csv(old / "runs/v2/per_bracket.csv").set_index("model_id").loc[bracket]
expected = {"model_id": bracket, "split": row.split,
            "mae_mpa": float(row.mae_mpa),
            "peak_predicted_mpa": float(row.peak_predicted_mpa)}
(out / f"{bracket}_expected.json").write_text(json.dumps(expected, indent=2))

for path in sorted(out.iterdir()):
    print(f"{path.name:<24} {path.stat().st_size / 1e6:6.1f} MB")