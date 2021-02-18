import sys
sys.path.append("layers")
from .layer import Layer
import numpy as np


class HiddenLayer(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weight = np.random.rand(input_shape[1], output_shape[1]) - 0.5
        self.bias = np.random.rand(1, output_shape[1]) - 0.5

    def forward_propagation(self, input):
        self.input_layer = input
        self.output = np.dot(input, self.weight) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        dweight = np.dot(self.input_layer.T, output_error)
        
        self.weight -= dweight*learning_rate
        self.bias -= learning_rate*output_error

        return np.dot(output_error, self.weight.T) 
