from networks.network import Network
from layers.hidden_layer import HiddenLayer
from layers.activation_layer import ActivationLayer
from activations.relu import relu, relu_prime
from loss.loss import loss_funtion, loss_funtion_prime
import numpy as np

x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
y_train = np.array([[[0]], [[0]], [[0]], [[1]]])
"""
net = Network()
net.add(HiddenLayer((1, 2), (1, 3)))
net.add(ActivationLayer((1, 3), (1, 3), relu, relu_prime))
net.add(HiddenLayer((1, 3), (1, 1)))
net.add(ActivationLayer((1, 1), (1, 1), relu, relu_prime))
net.setup_loss(loss_funtion, loss_funtion_prime)
net.fit(x_train, y_train, 1000, 0.01)
"""
net = Network()
net.add(HiddenLayer((1, 2), (1, 1)))
net.add(ActivationLayer((1, 1), (1, 1), relu, relu_prime))
net.setup_loss(loss_funtion, loss_funtion_prime)
net.fit(x_train, y_train, 1000, 0.01)
result = net.predict(x_train)
print(result)