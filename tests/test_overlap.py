"""Tests for shaker/overlap.py — overlap matrix text summary."""

from itertools import combinations

import numpy as np

from shaker.overlap import _overlap_report


def _results():
    names = ["A", "B", "C"]
    pairs = list(combinations(names, 2))  # A-B, A-C, B-C
    overlaps = np.array([0.9, 0.5, 0.7])
    matrix = np.eye(3)
    for (a, b), v in zip(pairs, overlaps):
        i, j = names.index(a), names.index(b)
        matrix[i, j] = matrix[j, i] = v
    return {"matrix": matrix, "overlaps": overlaps, "pairs": pairs, "bead_names": names}


def test_summary_by_default_matrix_on_request():
    summary = _overlap_report(_results())
    assert "3 beads, 3 pairs" in summary
    assert "pairs: 1 ✓  1 ⚠  1 ✗" in summary
    assert summary.splitlines()[-2].split() == ["A-C", "0.50", "✗", "B-C", "0.70", "⚠"]
    assert summary.splitlines()[1].startswith("mean OC")  # no matrix by default

    full = _overlap_report(_results(), print_matrix=True)
    assert full.splitlines()[2].split() == ["A", "-", ".90", ".50", "0.70"]
