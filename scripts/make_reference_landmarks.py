"""Median interface landmarks of the training brackets -> bundles/v2/reference_landmarks.npy"""
import json, sys
from pathlib import Path
import numpy as np

old = Path(sys.argv[1])
train = json.loads((old / "split.json").read_text())["train"]
landmarks = [np.load(old / "graphs" / f"{i}.npz")["landmarks"] for i in train]
reference = np.median(np.stack(landmarks), axis=0)
np.save("bundles/v2/reference_landmarks.npy", reference)
print(reference.round(2))