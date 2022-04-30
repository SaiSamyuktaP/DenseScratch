import numpy as np

def sigmoid(val):
    return 1/(1 + np.exp(-val))

def tanh(val):
    return (2/(1 + np.exp(-(2*val)))) - 1

def relu(val):
    return np.maximum(0, val)
