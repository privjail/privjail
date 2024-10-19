import numpy as _np

def laplace_mechanism(prisoner, epsilon):
    assert prisoner.sensitivity > 0
    return _np.random.laplace(loc=prisoner._value, scale=prisoner.sensitivity / epsilon)
