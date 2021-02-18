class Network:
    def __init__(self):
        self.layers = []
        self.loss_funtion = None
        self.loss_funtion_prime = None

    def add(self, layer):
        self.layers.append(layer)

    def setup_loss(self, loss_funtion, loss_funtion_prime):
        self.loss_funtion = loss_funtion
        self.loss_funtion_prime = loss_funtion_prime

    def predict(self, input):
        len_input = len(input)
        result = []
        for i in range(len_input):
            output = input[i]

            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    def fit(self, x_train, y_train, epoch, learning_rate):
        len_x_train = len(x_train)
        sum_loss = 0
        for i in range(epoch):
            for j in range(len_x_train):
                # Forward Propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                sum_loss += self.loss_funtion(output, y_train[j])

                # Backward Propagation
                output_error = self.loss_funtion_prime(output, y_train[j])
                for layer in reversed(self.layers):
                    output_error = layer.backward_propagation(output_error, learning_rate)
        
            loss = sum_loss / len_x_train    

            print('epoch : %d/%d ---- loss : %f'%(i, epoch, loss))
