from abc import abstractmethod
import numpy as np


class Layer:
    def __init__(self):
        self.input_layer = None
        self.output_layer = None
        self.input_layer_shape = None
        self.output_layer_shape = None
        raise NotImplementedError

    @abstractmethod
    def input(self):
        return self.input_layer

    @abstractmethod
    def output(self):
        return self.output_layer

    @abstractmethod
    def input_shape(self):
        return self.input_layer_shape

    @abstractmethod
    def output_shape(self):
        return self.output_layer_shape

    @abstractmethod
    def forward_propagation(self, input):
        raise NotImplementedError

    @abstractmethod
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError
