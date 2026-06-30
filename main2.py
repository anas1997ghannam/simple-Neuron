import numpy as np
from neuron import LayerDense

inputs=np.array([
    [1,2,3],
    [2,5,1],
    [4,2,1]
])

layer1=LayerDense(3,4)
layer2=LayerDense(4,3)

layer1.forward(inputs)
layer2.forward(layer1.output)
print("Layer 1 output:")
print(layer1.output)
print("Layer 2 output:")
print(layer2.output)
print()
print(inputs.shape)
print(layer1.output.shape)
print(layer2.output.shape)