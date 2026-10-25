"""Run validation on good and bad inputs, and show the messages."""
import sys
from pathlib import Path
import pandas as pd
from surrogate.bundle import load_model_bundle
from surrogate.preprocess import preprocess, PreprocessError
from surrogate.validate import (ValidationError, check_files,
                                check_load_case, check_landmarks)

old, raw = Path(sys.argv[1]), Path(sys.argv[2])
bundle = load_model_bundle("v2")


def run(label, model_id, load_case="ver"):
    vtk = raw / "Output_files" / f"{model_id}.vtk"
    fem = raw / "Input_files" / f"{model_id}.fem"
    try:
        check_files(vtk, fem)
        check_load_case(load_case, bundle)
        graph, _ = preprocess(vtk, fem)
        offset = check_landmarks(graph, bundle)
        print(f"[ACCEPTED] {label}: landmarks within {offset:.2f} mm")
    except (ValidationError, PreprocessError) as error:
        print(f"[REJECTED] {label}: {error}")


run("test bracket 21", 21)
run("test bracket 21, horizontal load", 21, load_case="hor")
run("bracket 293 (empty .fem)", 293)

excluded = pd.read_csv(old / "reports" / "excluded.csv")
print("\nbrackets the training run excluded:")
print(excluded.head(8).to_string(index=False))
for model_id in excluded.iloc[:3, 0]:
    run(f"excluded bracket {model_id}", int(model_id))