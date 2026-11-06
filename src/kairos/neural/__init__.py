"""NeuralProphet models and the recursion dial κ.

Submodules are imported on demand so ``import kairos.neural`` does not load torch.
"""

from kairos.neural.model import (
    NeuralConfig,
    build_model,
    fit_model,
    predict,
    quiet_neuralprophet,
    to_np_frame,
    trainer_config,
)
from kairos.neural.synthetic import DEFAULT_DRIVER_LAGS, TARGET_ID, synthetic_series

__all__ = [
    "DEFAULT_DRIVER_LAGS",
    "TARGET_ID",
    "NeuralConfig",
    "build_model",
    "fit_model",
    "predict",
    "quiet_neuralprophet",
    "synthetic_series",
    "to_np_frame",
    "trainer_config",
]
