"""Les modes décrivent le fonctionnement, sans étiqueter les défaillances."""

import numpy as np
import pandas as pd
import pytest

from metropt_maintenance.ingestion.modes import MODE_LABELS, infer_modes


def test_modes_respect_electrical_signals_and_unknown_states():
    frame = pd.DataFrame({
        "Motor_current": [0.04, 4, 7, 9, np.nan, -1, 0, 4, np.inf],
        "COMP": [1, 1, 0, 0, 1, 1, 0, 2, 1],
        "DV_eletric": [0, 0, 1, 1, 0, 0, 1, 0, 0],
    })
    assert infer_modes(frame).tolist() == [
        MODE_LABELS[0], MODE_LABELS[1], MODE_LABELS[2], MODE_LABELS[2],
        *([MODE_LABELS[3]] * 5),
    ]
    assert not any("défaillance" in name or "panne" in name for name in MODE_LABELS)


def test_current_threshold_is_configurable():
    frame = pd.DataFrame({"Motor_current": [0.5], "COMP": [1], "DV_eletric": [0]})
    assert infer_modes(frame, stopped_current_max=0.1).iloc[0] == MODE_LABELS[1]
    with pytest.raises(ValueError):
        infer_modes(frame, stopped_current_max=-1)
