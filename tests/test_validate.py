"""Validation must accept normal input and reject bad input with a clear reason."""
from dataclasses import replace

import pytest

from surrogate.validate import (ValidationError, check_files,
                                check_load_case, check_landmarks)


def test_normal_bracket_accepted(processed, bundle):
    graph, _ = processed
    assert check_landmarks(graph, bundle) < 5.0


def test_shifted_bracket_rejected(processed, bundle):
    graph, _ = processed
    shifted = graph.landmarks.copy()
    shifted[0] += [10.0, 0.0, 0.0]                     # move one bolt hole by 10 mm
    moved = replace(graph, landmarks=shifted)          # a copy of the graph with that change
    with pytest.raises(ValidationError, match="outside what model"):
        check_landmarks(moved, bundle)


def test_unsupported_load_case_rejected(bundle):
    with pytest.raises(ValidationError, match="not supported"):
        check_load_case("hor", bundle)


def test_supported_load_case_accepted(bundle):
    check_load_case("ver", bundle)                     # no error means pass


def test_empty_file_rejected(tmp_path, raw_files):
    empty = tmp_path / "empty.fem"
    empty.write_bytes(b"")
    with pytest.raises(ValidationError, match="empty"):
        check_files(raw_files[0], empty)


def test_wrong_file_type_rejected(tmp_path, raw_files):
    with pytest.raises(ValidationError, match="expected a .fem"):
        check_files(raw_files[0], tmp_path / "deck.txt")