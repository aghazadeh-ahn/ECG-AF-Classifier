"""Keep the first model's call unless it said Normal and the second model says Other."""

from __future__ import annotations

import numpy as np

from ecgaf.eval.metrics import CLASS_TO_INDEX


def review_normal_calls(base_pred: np.ndarray, call_other: np.ndarray) -> np.ndarray:
    """Class indices in, class indices out. Only a Normal prediction can change, and only to Other."""
    revised = np.asarray(base_pred, dtype=int).copy()
    flip = (revised == CLASS_TO_INDEX["N"]) & np.asarray(call_other, dtype=bool)
    revised[flip] = CLASS_TO_INDEX["O"]
    return revised
