from activation_func import sigmoid
from derv_activation_function import derv_sigmoid

s = self.activation_functions[i]
z = 1
print(eval('derv_'+s+'(z)'))
