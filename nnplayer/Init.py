import numpy as np


def normal(shape, mean=0, std=1):
    weights = np.random.normal(mean, std, shape)
    return weights


def zero(shape):
    weights = np.zeros(shape)
    return weights


def constant(shape, const):
    weights = np.ones(shape) * const
    return weights
