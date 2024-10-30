import numpy as _np
from .prisoner import Prisoner

def laplace_mechanism(prisoner: Prisoner[int | float], epsilon: float) -> float:
    assert prisoner.sensitivity > 0
    return _np.random.laplace(loc=prisoner._value, scale=prisoner.sensitivity / epsilon)
