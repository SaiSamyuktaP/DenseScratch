import numpy as np
from activation_func import sigmoid, tanh, relu
from derv_activation_function import derv_sigmoid, derv_relu, derv_tanh

class mymodel:
    weights = {}
    biases = {}
    prev_output = 0
    layer_no = 0
    Z = {}
    A = {}
    dZ = {}
    dA = {}
    dW = {}
    db = {}
    m = 0
    activation_functions = {}

    def __init__(self, alpha = 0.02, n_iter = 100):
        self.alpha = alpha
        self.n_iter = n_iter

    def initialize_parameters(self, input_units, output_units):
        self.weights[self.layer_no] = np.random.randn(input_units, output_units)*0.01
        self.biases[self.layer_no] = np.random.randn(output_units, 1)*0.01

    def forward_prop(self):
        for i in range(1, (self.layer_no + 1)):
            self.Z[i] = np.dot(self.weights[i].T, self.A[i-1]) + self.biases[i]
            self.A[i] = eval(self.activation_functions[i]+'(self.Z[i])')

    def compute_cost(self, Y):
        pred = self.A[self.layer_no]
        logprobs = np.multiply(Y ,np.log(pred)) + np.multiply((1-Y), np.log(1-pred))
        cost = (-1/self.m) * np.sum(logprobs)

        cost = float(np.squeeze(cost))  # makes sure cost is the dimension we expect.
                                        # E.g., turns [[17]] into 17
        return cost

    def backward_prop(self, Y):
        i = self.layer_no
        print(i)
        while(i > 0):
            print(i)
            if i == self.layer_no:
                self.dA[i] = np.divide((self.A[i] - Y), np.multiply(self.A[i], (1 - self.A[i])))
                #print("When i = layer_no. done")
            else:
                self.dA[i] = np.dot(self.weights[i+1], self.dZ[i+1])
            #print("dA:")
            #print(self.dA)
            #print(eval('derv_'+self.activation_functions[i]+'(self.Z[i])'))
            self.dZ[i] = self.dA[i] * eval('derv_'+self.activation_functions[i]+'(self.Z[i])')
            #print("dZ:")
            #print(self.dZ)
            self.dW[i] = (1/self.m) * np.dot(self.A[i-1], self.dZ[i].T)
            #print("dW:")
            #print(self.dW)
            self.db[i] = (1/self.m) * np.sum(self.dZ[i], axis = 1, keepdims = True)
            #print("db:")
            #print(self.db)
            #print("backward_prop of", i, "done")
            i = i - 1

    def update_parameters(self):
        for i in range(1, (self.layer_no + 1)):
            self.weights[i] -= self.alpha * self.dW[i]
            self.biases[i] -= self.alpha * self.db[i]

    def myDense(self, num_of_neurons, activation_function, input_neurons = 0):
        self.layer_no += 1
        if not input_neurons:
            input_neurons = self.prev_output
        self.initialize_parameters(input_neurons, num_of_neurons)
        self.prev_output = num_of_neurons
        self.activation_functions[self.layer_no] = activation_function

    def fit(self, X, Y):
        self.A[0] = X
        self.m = X.shape[1]
        cost = []
        for i in range(self.n_iter):
            print("Iteration no: ", i)
            self.forward_prop()
            #print("Forward Propagation done")
            # print("Weights:")
            # print(self.weights)
            # print("Biases:")
            # print(self.biases)
            cost.append(self.compute_cost(Y))
            print("Cost:", self.compute_cost(Y))
            self.backward_prop(Y)
            #print("Backward Propagation done")
            # print(self.dW)
            # print(self.db)
            # print(self.A)
            self.update_parameters()
            #print("Update Parameters done")
        return cost
