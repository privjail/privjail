import numpy as _np
from .util import DPError
from .prisoner import Prisoner

def laplace_mechanism(prisoner: Prisoner[int] | Prisoner[float], epsilon: float) -> float:
    if prisoner.sensitivity <= 0:
        raise DPError(f"Invalid sensitivity ({prisoner.sensitivity})")

    if epsilon <= 0:
        raise DPError(f"Invalid epsilon ({epsilon})")

    prisoner.consume_privacy_budget(epsilon)

    return _np.random.laplace(loc=prisoner._value, scale=prisoner.sensitivity / epsilon)
