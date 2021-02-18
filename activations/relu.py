import numpy as np


def relu(z):
    """
    z : array
    return : 0 if z <= 0
             z if z > 0

    """
    return np.maximum(0, z)


def relu_prime(z):
    """
    z : array
    return : 0 if z <= 0
             1 if z > 0

    """
    z[z > 0] = 1
    z[z < 0] = 0
    return z
