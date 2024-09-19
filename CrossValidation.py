import time

from NeuralNetwork import *
from UtilsFunctions import *
import concurrent.futures

from UtilsFile import add_csv_line


def cross_validation(X_train, Y_train, X_val, Y_val, num_neurons_list, lambda_list, dir, activation_fn_list, momentum=True, learning_rate=0.1, num_epochs=50, print_debug=True, mini_batch_size=64):
    best_parameters = None
    best_rmse = float('inf')
    best_neurons = None
    best_lambda = None
    best_error = None
    error_list_final_model = None
    best_activation_fn = None
    best_regularization = 0
    regularization_list = ["None", "L1", "L2"]

    start = time.time()
    print("\nStarting cross validation")

    def evaluate_model_CV(num_neurons, lambda_r, regularization, activation_fn):

        parameters, error, error_list = model(X_train, Y_train, num_neurons, learning_rate, num_epochs, activation_fn, lambda_r, momentum, regularization, mini_batch_size)
        rmse = evaluate_model_rmse(X_val, parameters, Y_val, activation_fn)

        if print_debug:
            if regularization == 0:
                text = f"RMSE for model {num_neurons} with no regularization and hidden activation function {activation_fn}: "+str(rmse)
            else:
                text = f"RMSE for model {num_neurons} with regularization {regularization_list[regularization]}, lambda {lambda_r} and hidden activation function {activation_fn}: "+str(rmse)
            print(text)

        return {
            'activation_fn': activation_fn,
            'parameters': parameters,
            'rmse': rmse,
            'num_neurons': num_neurons,
            'lambda': lambda_r,
            'error': error,
            'error_list': error_list,
            'regularization': regularization
        }

    def update_best_model(result):
        nonlocal best_rmse, best_parameters, best_neurons, best_lambda, best_error, error_list_final_model, best_activation_fn, best_regularization
        if result['rmse'] < best_rmse:
            best_rmse = result['rmse']
            best_parameters = result['parameters']
            best_neurons = result['num_neurons']
            best_lambda = result['lambda']
            best_error = result['error']
            error_list_final_model = result['error_list']
            best_activation_fn = result['activation_fn']
            best_regularization = result['regularization']

    results = []

    # Parallel processing using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for activation_fn in activation_fn_list:
            for num_neurons in num_neurons_list:
                futures.append(executor.submit(evaluate_model_CV, num_neurons, 0, 0, activation_fn))
                for lambda_r in lambda_list:
                    futures.append(executor.submit(evaluate_model_CV, num_neurons, lambda_r, 1, activation_fn))
                    futures.append(executor.submit(evaluate_model_CV, num_neurons, lambda_r, 2, activation_fn))

        # Aspetta che tutti i thread siano completati
        concurrent.futures.wait(futures)

        # Raccolta dei risultati da tutti i thread completati
        for future in futures:
            result = future.result()  # Estrai il risultato da ciascun future
            results.append(result)

    for result in results:
        add_csv_line(result['num_neurons'], regularization_list[result['regularization']], result['lambda'], result['error'], result['rmse'], result['activation_fn'], dir)
        update_best_model(result)

    end = time.time()
    min, sec = divmod(end - start, 60)
    print(f"End cross validation. Time spent for cross validation is {int(min)}:{sec:.2f} min\n")

    if best_regularization == 0:
        text = f"Best configuration is {best_neurons} using no regularization, with activation function {best_activation_fn}"
    else:
        text = f"Best configuration is {best_neurons} using {regularization_list[best_regularization]} with lambda {best_lambda} and activation function {best_activation_fn}"

    print(text)
    print("The RMSE on validation set is: "+str(best_rmse))

    with open(f'plots/{dir}/result/final_result', "w") as file:
        file.write(f"Time spent for cross validation is {int(min)}:{sec:.2f} min\n\n")
        file.write(text + "\n\n")

    plotError(error_list_final_model, len(error_list_final_model), dir)

    return best_parameters, best_activation_fn
