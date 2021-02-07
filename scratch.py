import numpy as np
from spiral_data_generator import spiral_data

np.random.seed(0)

X, y = spiral_data(100,3)

class Layer_Dense:
    def __init__(self,n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLu:
    def forward(self,inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self,inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims = True))
        probablities = exp_values / np.sum(exp_values, axis=1, keepdims = True)
        self.outputs = probablities

layer1 = Layer_Dense(2, 20)
activation1 = Activation_ReLu()
layer2 = Layer_Dense(20, 3)
softmax_layer = Activation_Softmax()

layer1.forward(X)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
softmax_layer.forward(layer2.output)


print(softmax_layer.outputs)

'''
X = [[1, 2, 3, 2.5],
     [2, 5, -1, 2],
     [-1.5, 2.7, 3.3, -0.8]]

weights1 = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases1 = [2, 3, 0.5] 

weights2 = [[0.1, -0.14, 0.5],
           [-0.5, 0.12, -0.33],
           [-0.44, 0.73, -0.13]]

biases2 = [-1, 2, -0.5] 

layer1_outputs = np.dot(np.array(inputs), np.array(weights1).T) + biases1
layer2_outputs = np.dot(np.array(layer1_outputs), np.array(weights2).T) + biases2
print(layer2_outputs)
'''