from model import mymodel
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)
sam = mymodel(alpha = 0.01, n_iter = 1000)
sam.myDense(3, "relu", 2)
sam.myDense(2, "relu")
sam.myDense(1, "sigmoid")
X1 = np.array([0, 0, 1, 1]).reshape(1, 4)
X2 = np.array([0, 1, 0, 1]).reshape(1, 4)
X = np.vstack((X1, X2))
Y = np.array([0, 1, 1, 1]).reshape(1, 4)
cost = sam.fit(X, Y)
print("Y:", Y)
print(sam.A[3])
# plt.plot(cost)
# plt.show()
#print(sam.layer_no)
