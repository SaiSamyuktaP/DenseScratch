import numpy as np
from activation_func import sigmoid

def derv_sigmoid(z):
    a = sigmoid(z)
    return np.multiply(a, (1 - a))

def derv_relu(z):
    z[z<=0] = 0
    z[z>0] = 1
    return z

def derv_tanh(z):
    a = sigmoid(z)
    return 1 - np.power(a,2)
