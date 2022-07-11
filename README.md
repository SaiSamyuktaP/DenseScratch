# DenseScratch
Created a Dense Layer from scratch using libraries - Numpy, Matplotlib only. 
The files contain following information - 
1. activation_func.py: 
  It contains 3 activation functions namely Sigmoid, Tanh and Relu which can be used during Forward Propagation.
2. derv_activation_function.py:
  It contains the derivations of all those above mentioned activation function which is necessary for Backward Propagation.
3. model.py:
  Here is where the model is created, it has functions
    1. To initialise parameters i.e., weights and biases,
    2. For forward propagation (using activation_func)
    3. To compute cost by using Binary Cross Entropy/Log loss formula
    4. For Backward propagation (using derv_activation_function)
    5. For Updating parameters
    6. Lastly for fitting the curve by making use of above mentioned functions
  4. At the end using the above created functions to fit for a certain X and Y example using main.py file.
    
