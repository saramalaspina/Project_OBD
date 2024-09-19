from UtilsFunctions import *

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
    Z = np.dot(W, A_prev) + b

    if activation_fn == "relu":
        A = np.maximum(0, Z)
    elif activation_fn == "tanh":
        A = np.tanh(Z)

    assert A.shape == (W.shape[0], A_prev.shape[1])

    linear_cache = (A_prev, W, b)
    cache = (linear_cache, Z)

    return A, cache


def L_model_backward_reg(AL, y, caches, activation_fn="relu", lambda_r=0, regularization = 0):

    y = y.reshape(AL.shape)
    L = len(caches)
    grads = {}

    dAL = AL - y

    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_backward_reg(dAL, caches[L-1], lambda_r, regularization)

    for l in range(L - 1, 0, -1):
        current_cache = caches[l - 1]
        grads["dA" + str(l - 1)], grads["dW" + str(l)], grads["db" + str(l)] = linear_activation_backward_reg(grads["dA" + str(l)], current_cache, activation_fn, lambda_r, regularization)
    return grads

def linear_activation_backward_reg(dA, cache, activation_fn="relu", lambda_r=0, regularization=0):

    linear_cache, activation_cache = cache

    if activation_fn == "relu":
        dZ = relu_gradient(dA, activation_cache)
    elif activation_fn == "tanh":
        dZ = tanh_gradient(dA, activation_cache)

    dA_prev, dW, db = linear_backward_reg(dZ, linear_cache, lambda_r, regularization)

    return dA_prev, dW, db

def linear_backward_reg(dZ, cache, lambda_r=0, regularization=0):

    A_prev, W, b = cache
    m = A_prev.shape[1]

    if(regularization == 2):
        dW = (1 / m) * np.dot(dZ, A_prev.T) + (lambda_r / m) * W
    else:
        dW = (1 / m) * np.dot(dZ, A_prev.T) + (lambda_r / m) * np.sign(W)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db