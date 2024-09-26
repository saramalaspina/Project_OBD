import time

from NeuralNetwork import *
from UtilsFunctions import *
import concurrent.futures

def cross_validation(X_train, Y_train, X_val, Y_val, num_neurons_list, lambda_list, activation_fn_list, num_epochs_list, minibatch_size, dir, print_debug=True):
    best_parameters = None
    best_rmse = float('inf')
    best_neurons = None
    best_epochs = None
    best_lambda = None
    final_mae = None
    error_list_final_model = None
    best_activation_fn = None
    best_regularization = 0
    regularization_list = ["None", "L1", "L2"]


    start = time.time()
    print("\nStarting cross validation")

    def evaluate_model_CV(num_neurons, epochs, minibatch_size, lambda_r, regularization, activation_fn):

        parameters, error_list = model(X_train, Y_train, num_neurons, 0.01, epochs, activation_fn, lambda_r, regularization, minibatch_size)
        rmse = evaluate_model_rmse(X_val, parameters, Y_val, activation_fn)
        mae = evaluate_model_mae(X_val, parameters, Y_val, activation_fn)

        if print_debug:
            if regularization == 0:
                text = f"Model {num_neurons} with no regularization, hidden activation function {activation_fn}, {epochs} epochs and minibatch of size {minibatch_size}: RMSE "+str(rmse)+" MAE "+str(mae)
            else:
                text = f"Model {num_neurons} with regularization {regularization_list[regularization]}, lambda {lambda_r}, hidden activation function {activation_fn}, {epochs} epochs and minibatch of size {minibatch_size}:  RMSE "+str(rmse)+" MAE "+str(mae)
            print(text)

        return {
            'activation_fn': activation_fn,
            'parameters': parameters,
            'rmse': rmse,
            'mae': mae,
            'num_neurons': num_neurons,
            'epochs': epochs,
            'minibatch_size': minibatch_size,
            'lambda': lambda_r,
            'error_list': error_list,
            'regularization': regularization
        }

    def update_best_model(result):
        nonlocal best_rmse, best_parameters, best_neurons, best_epochs, best_lambda, final_mae, error_list_final_model, best_activation_fn, best_regularization
        if result['rmse'] < best_rmse:
            best_rmse = result['rmse']
            final_mae = result['mae']
            best_parameters = result['parameters']
            best_neurons = result['num_neurons']
            best_epochs = result['epochs']
            best_lambda = result['lambda']
            error_list_final_model = result['error_list']
            best_activation_fn = result['activation_fn']
            best_regularization = result['regularization']

    results = []

    # Parallel processing using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for epochs in num_epochs_list:
            for activation in activation_fn_list:
                for num_neurons in num_neurons_list:
                    futures.append(executor.submit(evaluate_model_CV, num_neurons, epochs, minibatch_size, 0, 0, activation))
                    for lambda_r in lambda_list:
                        futures.append(executor.submit(evaluate_model_CV, num_neurons, epochs, minibatch_size, lambda_r, 1, activation))
                        futures.append(executor.submit(evaluate_model_CV, num_neurons, epochs, minibatch_size, lambda_r, 2, activation))

        # Aspetta che tutti i thread siano completati
        concurrent.futures.wait(futures)

        # Raccolta dei risultati da tutti i thread completati
        for future in futures:
            result = future.result()  # Estrai il risultato da ciascun future
            results.append(result)

    for result in results:
        add_csv_line(result['num_neurons'], regularization_list[result['regularization']], result['lambda'], result['rmse'], result['mae'], result['activation_fn'], result['epochs'], result['minibatch_size'], dir)
        update_best_model(result)

    end = time.time()
    min, sec = divmod(end - start, 60)
    print(f"End cross validation. Time spent for cross validation is {int(min)}:{sec:.2f} min\n")

    if best_regularization == 0:
        text = f"Best configuration is {best_neurons} using no regularization, with activation function {best_activation_fn}, {best_epochs} epochs and minibatch of size {minibatch_size}\n"
    else:
        text = f"Best configuration is {best_neurons} using {regularization_list[best_regularization]} with lambda {best_lambda}, activation function {best_activation_fn}, {best_epochs} epochs and minibatch of size {minibatch_size}\n"

    print(text)
    text2 = "The RMSE on validation set is: "+str(best_rmse)+"\nThe MAE on validation set is: "+str(final_mae)
    print(text2)

    with open(f'plots/{dir}/result/final_result', "w") as file:
        file.write(f"Time spent for cross validation is {int(min)}:{sec:.2f} min\n\n")
        file.write(text + "\n\n")
        file.write(text2 + "\n\n")

    plotError(error_list_final_model, len(error_list_final_model), dir)

    return best_parameters, best_activation_fn