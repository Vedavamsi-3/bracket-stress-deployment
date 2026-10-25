"""Checks that run before prediction. Each failure gives a clear, customer-facing reason."""
from pathlib import Path

import numpy as np

MAX_FILE_MB = 300


class ValidationError(Exception):
    """The input cannot be predicted on; the message says why."""


def check_files(vtk_path, fem_path):
    """Before reading: right type, exists, not empty, not huge."""
    for path, suffix in ((Path(vtk_path), ".vtk"), (Path(fem_path), ".fem")):
        if path.suffix.lower() != suffix:
            raise ValidationError(f"expected a {suffix} file, got {path.name}")
        if not path.is_file():
            raise ValidationError(f"{path.name} not found")
        size_mb = path.stat().st_size / 1e6
        if size_mb == 0:
            raise ValidationError(f"{path.name} is empty")
        if size_mb > MAX_FILE_MB:
            raise ValidationError(f"{path.name} is {size_mb:.0f} MB; the limit is {MAX_FILE_MB} MB")


def check_load_case(load_case, bundle):
    """Only load cases the model was trained on."""
    supported = bundle.info["load_cases"]
    if load_case not in supported:
        raise ValidationError(
            f"load case {load_case!r} is not supported by model {bundle.info['name']}; "
            f"supported: {', '.join(supported)}")


def check_landmarks(graph, bundle):
    """The template check: is this bracket laid out like the training brackets?"""
    expected = bundle.reference_landmarks
    if graph.landmarks.shape != expected.shape:
        raise ValidationError(
            f"expected 4 bolt holes and 1 load lug, found {len(graph.landmarks) - 1} "
            f"bolt holes and 1 load lug")
    offset = float(np.abs(graph.landmarks - expected).max())
    tolerance = bundle.info["landmark_tolerance_mm"]
    if offset > tolerance:
        raise ValidationError(
            f"bolt holes or load lug sit {offset:.1f} mm from a typical bracket "
            f"(limit {tolerance} mm); this layout is outside what model "
            f"{bundle.info['name']} was trained on")
    return offset