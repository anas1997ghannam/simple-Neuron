#simple neural network with 3 inputs, 1 neuron, and 1 output
# import numpy as np

# inputs = np.array([1, 2, 3])
# weights = np.array([0.2, 0.8, -0.5])
# bias = 2

# output = np.dot(inputs, weights) + bias

# print(output)

#simple neural network with 3 inputs, 3 neurons, and 1 output
# import numpy as np

# inputs = np.array([1, 2, 3])

# weights = np.array([
#     [0.2, 0.8, -0.5],
#     [0.5, -0.91, 0.26],
#     [-0.26, -0.27, 0.17]
# ])

# biases = np.array([2, 3, 0.5])

# outputs = np.dot(weights, inputs) + biases

# print(outputs)
# print(np.dot(weights[0], inputs)+biases[0])
# print(np.dot(weights[1], inputs)+biases[1])
# print(np.dot(weights[2], inputs)+biases[2])

import numpy as np

inputs = np.array([
    [1,2,3],
    [2,5,1],
    [4,2,1]
])

weights = np.array([
    [0.2, 0.8, -0.5],
    [0.5, -0.91, 0.26],
    [-0.26, -0.27, 0.17]
])

biases = np.array([2, 3, 0.5])

outputs = np.dot(inputs, weights.T) + biases
print(outputs)

print(inputs.shape)
print(weights.shape)
print(weights.T.shape)
print(outputs.shape)