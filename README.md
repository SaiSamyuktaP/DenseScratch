# DenseScratch
Created a Dense Layer from scratch using libraries - Numpy, Matplotlib only. 
The files contain following information - 
1. activation_func.py: 
  It contains 3 activation functions namely Sigmoid, Tanh and Relu which can be used during Forward Propagation.
2. derv_activation_function.py:
  It contains the derivations of all those above mentioned activation function which is necessary for Backward Propagation.
3. model.py:
  Here is where the model is created, it has functions
    a. To initialise parameters i.e., weights and biases, 
    b. For forward propagation (using activation_func)
    c. To compute cost by using Binary Cross Entropy/Log loss formula 
    d. For Backward propagation (using derv_activation_function)
    e. For Updating parameters 
    f. Lastly for fitting the curve by making use of above mentioned functions
  4. At the end using the above created functions to fit for a certain X and Y example using main.py file.
    