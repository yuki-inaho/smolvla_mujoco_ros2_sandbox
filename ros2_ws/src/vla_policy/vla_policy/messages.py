from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class ObservationFrame:
    """Latest synchronized-enough observation used by a policy backend."""

    joint_names: list[str]
    joint_positions: np.ndarray
    images: Dict[str, np.ndarray]
    task_instruction: str
