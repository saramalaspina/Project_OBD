import numpy as np
import matplotlib.pyplot as plt

# Roll a dictionary into a single vector.
def dictionary_to_vector(params_dict):
    count = 0
    for key in params_dict.keys():
        new_vector = np.reshape(params_dict[key], (-1, 1))
        if count == 0:
            theta_vector = new_vector
        else:
            theta_vector = np.concatenate((theta_vector, new_vector))
        count += 1
    return theta_vector


def create_mini_batches(X, y, mini_batch_size):
    m = X.shape[1]
    mini_batches = []

    permutation = np.random.permutation(m)
    shuffled_X = X[:, permutation]
    shuffled_y = y[:, permutation]

    num_complete_minibatches = m // mini_batch_size
    for k in range(num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch_y = shuffled_y[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batches.append((mini_batch_X, mini_batch_y))

    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size:]
        mini_batch_y = shuffled_y[:, num_complete_minibatches * mini_batch_size:]
        mini_batches.append((mini_batch_X, mini_batch_y))

    return mini_batches


def plotError(error_list, num_iterations, dir, model_name="final_model_error"):
    iterations = list(range(0, num_iterations))
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, error_list, marker='', linestyle='-', color='b', linewidth=1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss through training')
    plt.grid(True)
    plt.savefig('plots/' + dir + '/result/' + model_name + '.png')
    plt.close()


