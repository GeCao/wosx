"""Load an optional, independently generated case-8 FEM capacitance matrix."""

import csv
import json
from pathlib import Path

import numpy as np

from .full_case8 import make_case


def load_fem_matrix(path):
    """Return an 8x8 active-terminal matrix or a full 9x9 matrix, in fF."""
    path = Path(path)
    expected = [item["id"] for item in make_case()["conductors"]]
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text())
        if isinstance(value, list):
            value = value[-1]
        matrix = np.asarray(value["C_fF"], dtype=float)
        labels = value.get("terminals")
    else:
        with path.open(newline="") as stream:
            rows = list(csv.reader(stream))
        labels = rows[0][1:]
        if [row[0] for row in rows[1:]] != labels:
            raise ValueError("FEM CSV row labels differ from its columns")
        matrix = np.asarray([[float(value) for value in row[1:]]
                             for row in rows[1:]], dtype=float)
    target_labels = expected if matrix.shape == (9, 9) else expected[1:]
    if labels != target_labels:
        raise ValueError("FEM terminal order differs from the case-8 deck")
    if matrix.shape not in ((8, 8), (9, 9)) or not np.isfinite(matrix).all():
        raise ValueError("FEM reference must be a finite 8x8 or 9x9 matrix")
    return matrix
