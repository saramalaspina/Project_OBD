from UtilsFunctions import *

# forward propagation con funzione di attivazione lineare
def forward_propagation(X, parameters, activation_fn):
    Z = X
    caches = []
    L = len(parameters) // 2

    assert activation_fn == "relu" or activation_fn == "tanh"
    if activation_fn == "relu":
        activation_function = lambda A: np.maximum(0, A)
    elif activation_fn == "tanh":
        activation_function = lambda A: np.tanh(A)

    # Forward propagate through hidden layers
    for l in range(1, L):
        Z_prev = Z
        A = np.dot(parameters["W" + str(l)], Z_prev) + parameters["b" + str(l)]
        Z = activation_function(A)
        cache = (Z_prev, A)
        caches.append(cache)

    # Output layer: no activation function (linear)
    AL = np.dot(parameters["W" + str(L)], Z) + parameters["b" + str(L)]
    caches.append(Z)    #NOTA: l'ultima cache non ha A

    assert AL.shape == (1, X.shape[1])

    return AL, caches

def backward_propagation(AL, y, caches, parameters, activation_fn="relu", lambda_r=0, regularization=0):
    y = y.reshape(AL.shape)
    L = len(caches)
    gradient = {}

    assert activation_fn == "relu" or activation_fn == "tanh"
    # Define activation gradient function outside the loop
    if activation_fn == "relu":
        activation_gradient = lambda A: np.int64(A > 0)
    elif activation_fn == "tanh":
        activation_gradient = lambda A: 1 - np.square(np.tanh(A))

    assert regularization == 0 or regularization == 1 or regularization == 2
    # Define regularization function outside the loop
    if regularization == 2:
        reg = lambda W: lambda_r * W
    elif regularization == 1:
        reg = lambda W: lambda_r * np.sign(W)
    else:
        reg = lambda W: 0

    dEda = AL - y
    m = caches[L - 1].shape[1]    #NOTA: Z_prev è caches[L - 1]
    gradient["dW" + str(L)] = (np.dot(dEda, caches[L - 1].T) + reg(parameters["W" + str(L)])) / m
    gradient["db" + str(L)] = np.sum(dEda, axis=1, keepdims=True) / m

    for l in range(L - 1, 0, -1):
        Z_prev, A = caches[l - 1]
        dEda = np.dot(parameters["W" + str(l+1)].T, dEda) * activation_gradient(A)
        m = Z_prev.shape[1]
        gradient["dW" + str(l)] = (np.dot(dEda, Z_prev.T) + reg(parameters["W" + str(l)])) / m
        gradient["db" + str(l)] = np.sum(dEda, axis=1, keepdims=True) / m
    return gradient