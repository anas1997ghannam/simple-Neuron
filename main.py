import numpy as np

inputs = np.array([1, 2, 3])

weights = np.array([
    [0.2, 0.8, -0.5],
    [0.5, -0.91, 0.26],
    [-0.26, -0.27, 0.17]
])

biases = np.array([2, 3, 0.5])

outputs = np.dot(weights, inputs) + biases

print(outputs)
print(np.dot(weights[0], inputs)+biases[0])
print(np.dot(weights[1], inputs)+biases[1])
print(np.dot(weights[2], inputs)+biases[2])