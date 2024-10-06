from Backpropagation import *
from UtilsFunctions import *


def model(X, y, num_neurons, learning_rate, epochs, activation_fn, lambda_r, regularization, minibatch_size):
    np.random.seed(1)

    # Initialize parameters
    parameters, previous_parameters = initialize_parameters(num_neurons, activation_fn)

    # Initialize cost list
    cost_list = []

    for i in range(epochs):
        diminishing_stepsize = learning_rate / (1 + 0.01 * i)
        # Create mini-batch
        mini_batches = create_mini_batches(X, y, minibatch_size)

        for mini_batch in mini_batches:
            (mini_batch_X, mini_batch_y) = mini_batch

            # Backpropagation algorithm
            AL, caches = forward_propagation(mini_batch_X, parameters, activation_fn)
            gradient = backward_propagation(AL, mini_batch_y, caches, parameters, activation_fn, lambda_r, regularization)

            parameters, previous_parameters = update_parameters(parameters, gradient, diminishing_stepsize, previous_parameters)

        # Evaluation of the trained model on training set
        AL, caches = forward_propagation(X, parameters, activation_fn)
        reg_cost = compute_cost_reg(AL, y, parameters, lambda_r, regularization)
        cost_list.append(reg_cost)

    return parameters, cost_list


# Function used to calculated regularized cost function
def compute_cost_reg(AL, y, parameters, lambda_r=0, regularization=0):
    # Number of examples
    m = y.shape[1]
    # Compute mean squared error (MSE)
    mse_cost = (1 / (2 * m)) * np.sum(np.square(AL - y))
    # Convert parameters dictionary to vector
    parameters_vector = dictionary_to_vector(parameters)

    # Compute the regularization penalty
    if regularization == 2:
        regularization_penalty = (lambda_r / (2 * m)) * np.sum(np.square(parameters_vector))
    elif regularization == 1:
        regularization_penalty = (lambda_r / (2 * m)) * np.sum(np.abs(parameters_vector))
    else:
        regularization_penalty = 0

    # Compute the total cost function (MSE + regularization)
    cost = mse_cost + regularization_penalty
    return cost


def evaluate_model_rmse(X, parameters, y, activation_fn):
    # Forward propagate to get predictions
    predictions, caches = forward_propagation(X, parameters, activation_fn)
    # Compute Root Mean Squared Error (RMSE)
    rmse = np.sqrt(np.mean(np.square(predictions - y)))
    return rmse

def evaluate_model_mae(X, parameters, y, activation_fn):
    # Forward propagate to get predictions
    predictions, caches = forward_propagation(X, parameters, activation_fn)
    # Compute Mean Absolute Error (MAE)
    mae = np.mean(np.abs(predictions - y))
    return mae

def initialize_parameters(num_neurons, activation_fn):
    assert activation_fn == "relu" or activation_fn == "tanh"
    np.random.seed(1)
    parameters = {}
    previous_parameters = {}
    L = len(num_neurons)

    for l in range(1,L):
        if activation_fn == "relu":
            #He initialization for relu
            factor = np.sqrt(2 / num_neurons[l - 1])
        else:
            #Xavier/Glorot initialization for tanh
            factor = np.sqrt(2 / (num_neurons[l - 1] + num_neurons[l]))
        parameters["W" + str(l)] = np.random.randn(num_neurons[l], num_neurons[l - 1]) * factor

        parameters["b" + str(l)] = np.zeros((num_neurons[l], 1))
        previous_parameters["W" + str(l)] = np.zeros((num_neurons[l], num_neurons[l - 1]))
        previous_parameters["b" + str(l)] = np.zeros((num_neurons[l], 1))

        assert parameters["W" + str(l)].shape == (num_neurons[l], num_neurons[l - 1])
        assert parameters["b" + str(l)].shape == (num_neurons[l], 1)

    return parameters, previous_parameters


def update_parameters(parameters, gradient, learning_rate, previous_parameters, beta=0.9):
    L = len(parameters) // 2
    prev_parameters = parameters

    for l in range(1, L + 1):
        # Gradient clipping for dW and db
        gradient["dW" + str(l)] = np.clip(gradient["dW" + str(l)], -5, 5)
        gradient["db" + str(l)] = np.clip(gradient["db" + str(l)], -5, 5)

        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * gradient["dW" + str(l)] + beta * (parameters["W" + str(l)] - previous_parameters["W" + str(l)])
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * gradient["db" + str(l)] + beta * (parameters["b" + str(l)] - previous_parameters["b" + str(l)])

    return parameters, prev_parameters


