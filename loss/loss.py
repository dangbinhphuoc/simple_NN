def loss_funtion(y_predict, y_true):
    return 0.5*(y_predict-y_true)**2


def loss_funtion_prime(y_predict, y_true):
    return y_predict-y_true
