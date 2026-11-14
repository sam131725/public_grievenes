"""Class-imbalance helpers (numpy only, so they are testable without torch)."""
from __future__ import annotations

import numpy as np


def balanced_class_weights(y: list[int], n_classes: int) -> np.ndarray:
    """Inverse-frequency weights: n / (k_present * count_c). Absent classes get 0.

    sklearn's compute_class_weight raises when a class is missing from `y`;
    in multilingual/small data some urgency levels can be absent from train.
    """
    counts = np.bincount(np.asarray(y, dtype=int), minlength=n_classes).astype(float)
    present = counts > 0
    w = np.zeros(n_classes)
    w[present] = counts.sum() / (present.sum() * counts[present])
    return w
