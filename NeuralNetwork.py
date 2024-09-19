from Backpropagation import *
from UtilsFunctions import *


def model(X, y, num_neurons, learning_rate, epochs, activation_fn, lambda_r, regularization, minibatch_size):

    # get number of examples
    m = X.shape[1]

    # to get consistents output
    np.random.seed(1)

    # initialize parameters
    parameters, previous_parameters = initialize_parameters(num_neurons, activation_fn)

    # intialize cost and metric list
    metric_list = []
    cost_list = []

    for i in range(epochs):
        diminishing_stepsize = learning_rate / (1 + 0.01 * i)
        # Creare mini-batch
        mini_batches = create_mini_batches(X, y, minibatch_size)

        for mini_batch in mini_batches:
            (mini_batch_X, mini_batch_y) = mini_batch

            # Forward propagation
            AL, caches = L_model_forward(mini_batch_X, parameters, activation_fn)

            # Backward propagation
            grads = L_model_backward_reg(AL, mini_batch_y, caches, activation_fn, lambda_r, regularization)

            # Update parameters
            parameters, previous_parameters = update_parameters(parameters, grads, diminishing_stepsize, previous_parameters)


        AL, caches = L_model_forward(X, parameters, activation_fn)
        reg_cost = compute_cost_reg(AL, y, parameters, lambda_r, regularization)

        cost_list.append(reg_cost)

    return parameters, reg_cost, cost_list


# metodo per calcolare la funzione costo regolarizzata
def compute_cost_reg(AL, y, parameters, lambda_r=0, regularization=0):
    # number of examples
    m = y.shape[1]
    # compute mean squared error (MSE)
    mse_cost = (1 / (2 * m)) * np.sum(np.square(AL - y))
    # convert parameters dictionary to vector
    parameters_vector = dictionary_to_vector(parameters)

    # compute the regularization penalty
    if regularization == 2:
        regularization_penalty = (lambda_r / (2 * m)) * np.sum(np.square(parameters_vector))
    elif regularization == 1:
        regularization_penalty = (lambda_r / (2 * m)) * np.sum(np.abs(parameters_vector))
    else:
        regularization_penalty = 0

    # compute the total cost (MSE + regularization)
    cost = mse_cost + regularization_penalty
    return cost


def evaluate_model_rmse(X, parameters, y, activation_fn):
    # Forward propagate to get predictions
    predictions, caches = L_model_forward(X, parameters, activation_fn)
    # Compute Root Mean Squared Error (RMSE)
    rmse = np.sqrt(np.mean(np.square(predictions - y)))
    return rmse


def initialize_parameters(num_neurons, activation_fn):
    assert activation_fn == "relu" or activation_fn == "tanh"
    np.random.seed(1)
    parameters = {}
    previous_parameters = {}
    L = len(num_neurons)

    for l in range(1,L):
        if activation_fn == "relu":
            #He initialization for relu
            parameters["W" + str(l)] = np.random.randn(num_neurons[l], num_neurons[l - 1]) * np.sqrt(2 / num_neurons[l - 1])
        else:
            #Xavier/Glorot initialization for tanh
            parameters["W" + str(l)] = np.random.randn(num_neurons[l], num_neurons[l - 1]) * np.sqrt(2 / (num_neurons[l - 1] + num_neurons[l]))

        parameters["b" + str(l)] = np.zeros((num_neurons[l], 1))
        previous_parameters["W" + str(l)] = np.zeros((num_neurons[l], num_neurons[l - 1]))
        previous_parameters["b" + str(l)] = np.zeros((num_neurons[l], 1))

        assert parameters["W" + str(l)].shape == (num_neurons[l], num_neurons[l - 1])
        assert parameters["b" + str(l)].shape == (num_neurons[l], 1)

    return parameters, previous_parameters


def update_parameters(parameters, grads, learning_rate, previous_parameters, beta=0.9):
    L = len(parameters) // 2
    prev_parameters = parameters

    for l in range(1, L + 1):
        # Gradient clipping for dW and db
        grads["dW" + str(l)] = np.clip(grads["dW" + str(l)], -5, 5)
        grads["db" + str(l)] = np.clip(grads["db" + str(l)], -5, 5)

        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)] + beta * (parameters["W" + str(l)] - previous_parameters["W" + str(l)])
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)] + beta * (parameters["b" + str(l)] - previous_parameters["b" + str(l)])

    return parameters, prev_parameters


