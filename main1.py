import numpy as np
from neuron import LayerDense

inputs = np.array([
    [1,2,3],
    [2,5,1],
    [4,2,1]
])

layer1 = LayerDense(3,4)  # 3 inputs, 4 neurons
layer1.forward(inputs)  
print(layer1.output)

print(layer1.weights.shape)
print(layer1.biases.shape)
print(layer1.output.shape)  
print (np.random.randn(4,3))  # Random weights for 4 neurons and 3 inputs
