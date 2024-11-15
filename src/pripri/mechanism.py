from typing import Any
import numpy as _np
from .util import DPError
from .prisoner import Prisoner

def laplace_mechanism(prisoner: Prisoner[Any], epsilon: float) -> Any:
    sensitivity = prisoner.distance.max()

    if sensitivity <= 0:
        raise DPError(f"Invalid sensitivity ({sensitivity})")

    if epsilon <= 0:
        raise DPError(f"Invalid epsilon ({epsilon})")

    prisoner.consume_privacy_budget(epsilon)

    # TODO: type check for values, otherwise sensitive values can be leaked via errors (e.g., string)
    return _np.random.laplace(loc=prisoner._value, scale=sensitivity / epsilon)
