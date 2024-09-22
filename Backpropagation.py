from UtilsFunctions import *

# forward propagation con funzione di attivazione lineare
def forward_propagation(X, parameters, activation_fn="relu"):
    Z = X
    caches = []
    L = len(parameters) // 2

    assert activation_fn == "relu" or activation_fn == "tanh"
    if activation_fn == "relu":
        def activation_function(A):
            return np.maximum(0, A)
    elif activation_fn == "tanh":
        def activation_function(A):
            return np.tanh(A)

    # Forward propagate through hidden layers
    for l in range(1, L):
        Z_prev = Z
        A = np.dot(parameters["W" + str(l)], Z_prev) + parameters["b" + str(l)]
        Z = activation_function(A)
        cache = ((Z_prev, parameters["W" + str(l)], parameters["b" + str(l)]), A)
        caches.append(cache)

    # Output layer: no activation function (linear)
    AL = np.dot(parameters["W" + str(L)], Z) + parameters["b" + str(L)]
    cache = (Z, parameters["W" + str(L)], parameters["b" + str(L)])
    caches.append(cache)

    assert AL.shape == (1, X.shape[1])

    return AL, caches

def backward_propagation(AL, y, caches, activation_fn="relu", lambda_r=0, regularization=0):
    #y = y.reshape(AL.shape) FUNZIONA ANCHE SE LO TOLGO QUINDI A CHE CAZZO SERVE
    L = len(caches)
    grads = {}

    #dAL = AL - y
    dA = AL - y

    #grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_backward_reg(dAL, caches[L - 1],lambda_r, regularization)
    dA, grads["dW" + str(L)], grads["db" + str(L)] = linear_backward_reg(dA, caches[L - 1],lambda_r, regularization)

    for l in range(L - 1, 0, -1):
        current_cache = caches[l - 1]
        dA, grads["dW" + str(l)], grads["db" + str(l)] = linear_activation_backward_reg(dA, current_cache, activation_fn, lambda_r, regularization)
    return grads


def linear_activation_backward_reg(dA, cache, activation_fn="relu", lambda_r=0, regularization=0):
    # l-1 quindi ho Z invece di A
    linear_cache, A = cache

    if activation_fn == "relu":
        dEdgda = dA * np.int64(A > 0)

    elif activation_fn == "tanh":
        dEdgda = dA * (1 - np.square(np.tanh(A)))

    dZ_prev, dW, db = linear_backward_reg(dEdgda, linear_cache, lambda_r, regularization)

    return dZ_prev, dW, db


def linear_backward_reg(dEdgda, cache, lambda_r=0, regularization=0):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    if (regularization == 2):
        dW = (1 / m) * np.dot(dEdgda, A_prev.T) + (lambda_r / m) * W
    else:
        dW = (1 / m) * np.dot(dEdgda, A_prev.T) + (lambda_r / m) * np.sign(W)
    db = (1 / m) * np.sum(dEdgda, axis=1, keepdims=True)
    dZ_prev = np.dot(W.T, dEdgda)

    assert (dZ_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dZ_prev, dW, db