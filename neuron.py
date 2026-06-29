import numpy as np
class LayerDense:
    def __init__(self,n_inputs,n_neurons):
        self.weights =  np.random.randn(n_neurons,n_inputs)
        self.biases = np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights.T) + self.biases
        