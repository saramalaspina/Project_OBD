
# metodo per calcolare la funzione costo regolarizzata
def compute_cost_reg(AL, y, parameters, lambda_rega_rega_reg=0, reg_type):

    # number of examples
    m = y.shape[1]

    # compute mean squared error (MSE)
    mse_cost = (1 / (2 * m)) * np.sum(np.square(AL - y))

    # convert parameters dictionary to vector
    parameters_vector = dictionary_to_vector(parameters)

    # compute the regularization penalty
    if reg_type == 2:
        regularization_penalty = (lambda_rega_rega_reg / (2 * m)) * np.sum(np.square(parameters_vector))
    elif reg_type == 1:
        regularization_penalty = (lambda_rega_rega_reg / (2 * m)) * np.sum(np.abs(parameters_vector))
    else:
        regularization_penalty = 0

    # compute the total cost (MSE + regularization)
    cost = mse_cost + regularization_penalty

    return cost


# forward propagation con funzione di attivazione lineare
def L_model_forward(X, parameters, activation_fn="relu"):
    A = X
    caches = []
    L = len(parameters) // 2

    # Forward propagate through hidden layers
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], activation_fn=activation_fn)
        caches.append(cache)

    # Output layer: no activation function (linear)
    AL, cache = linear_forward(A, parameters["W" + str(L)], parameters["b" + str(L)])
    caches.append(cache)

    assert AL.shape == (1, X.shape[1])

    return AL, caches


# calcolo Z nella forward propagation
def linear_forward(A_prev, W, b):

    Z = np.dot(W, A_prev) + b
    cache = (A_prev, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation_fn):

    assert activation_fn == "relu" or activation_fn == "tanh" 

    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation_fn == "relu":
        A, activation_cache = relu(Z)
    elif activation_fn == "tanh":
        A, activation_cache = tanh(Z)

    assert A.shape == (W.shape[0], A_prev.shape[1])

    cache = (linear_cache, activation_cache)

    return A, cache


def linear_backward_reg(dZ, cache, lambda_reg=0, reg_type):

    A_prev, W, b = cache
    m = A_prev.shape[1]

    if(reg_type):
        dW = (1 / m) * np.dot(dZ, A_prev.T) + (lambda_reg / m) * W
    else:
        dW = (1 / m) * np.dot(dZ, A_prev.T) + (lambda_reg / m) * np.sign(W)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def linear_activation_backward_reg(dA, cache, activation_fn="relu", lambda_reg=0, reg_type):

    linear_cache, activation_cache = cache

    if activation_fn == "relu":
        dZ = relu_gradient(dA, activation_cache)
    elif activation_fn == "tanh":
        dZ = tanh_gradient(dA, activation_cache)
    elif activation_fn == "linear":
        dZ = dA
    
    dA_prev, dW, db = linear_backward_reg(dZ, linear_cache, lambda_reg, reg_type)

    return dA_prev, dW, db


def L_model_backward_reg(AL, y, caches, activation_fn="relu",lambda_reg=0, reg_type):

    y = y.reshape(AL.shape)
    L = len(caches)
    grads = {}

    dAL = AL - y

    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward_reg(dAL, caches[L - 1], "linear", lambda_reg, reg_type)

    for l in range(L - 1, 0, -1):
        current_cache = caches[l - 1]
        grads["dA" + str(l - 1)], grads["dW" + str(l)], grads["db" + str(l)] = linear_activation_backward_reg( grads["dA" + str(l)], current_cache, activation_fn, lambda_reg, reg_type)
    return grads



# calcolo dell'errore quadratico medio
def evaluate_model(X, parameters, y, activation_fn):

    # Forward propagate to get predictions
    predictions, caches = L_model_forward(X, parameters, activation_fn)

    # Compute Mean Squared Error (MSE)
    mse = np.mean(np.square(predictions - y))

    return mse


def model_regularization(X, y, layers_dims, dir, learning_rate=0.01, num_epochs=50, activation_fn="relu", lambda_rega_reg=0, momentum=True, reg_type, mini_batch_size=64):

    # get number of examples
    m = X.shape[1]

    # to get consistents output
    np.random.seed(1)

    # initialize parameters
    parameters, previous_parameters = initialize_parameters(layers_dims, activation_fn)

    # intialize cost list
    cost_list = []

    for i in range(num_epochs):

        diminishing_stepsize = learning_rate / (1 + 0.01 * i)

        # Creare mini-batch
        mini_batches = create_mini_batches(X, y, mini_batch_size)

        for mini_batch in mini_batches:
            (mini_batch_X, mini_batch_y) = mini_batch

            # Forward propagation
            AL, caches = L_model_forward(mini_batch_X, parameters, activation_fn)

            # Calculate regularized cost (MSE + regularization)
            reg_cost = compute_cost_reg(AL, mini_batch_y, parameters, lambda_rega_reg, reg_type)

            # Backward propagation
            grads = L_model_backward_reg(AL, mini_batch_y, caches, activation_fn, lambda_rega_reg, reg_type)

            # Update parameters
            parameters, previous_parameters = update_parameters(parameters, grads, diminishing_stepsize, previous_parameters, momentum)


        AL, caches = L_model_forward(X, parameters, activation_fn)
        reg_cost = compute_cost_reg(AL, y, parameters, lambda_rega_reg, reg_type)

        # Evaluate model on full training set
        mse = evaluate_model(X, parameters, y, activation_fn)
        cost_list.append(mse)

    return parameters, reg_cost, cost_list


