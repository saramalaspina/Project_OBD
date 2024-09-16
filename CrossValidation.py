import numpy as np
import time

from NeuralNetwork import *
from PlotUtils import *
from FileUtils import *
import concurrent.futures


def cross_validation(X_train, Y_train, X_valid, Y_valid, layers_neurons_list, lambda_list, dir, activation_function="relu", momentum=True, learning_rate=0.1, num_epochs=50, print_debug=True, mini_batch_size=64):

    #cambia nomi
    best_parameters = None
    best_rmse float = 0.0
    best_neurons = None
    best_lambda = None
    error_list_final_model = None
    reg_type = 0
    regularization_list = ["None", "L1", "L2"]

    start = time.time()
    print("\nStarting cross validation")

    def evaluate_model_CV(layers_neurons, lambda_reg, reg_type):

            # accuracy list va riadattato al nostro caso

        parameters, error, error_list, accuracy_list = model_with_regularization(
            X_train, Y_train, layers_neurons, dir, learning_rate, num_epochs, activation_function,
            lambda_reg, momentum, reg_type, mini_batch_size=mini_batch_size)

        rmse = evaluate_model(X_valid, parameters, Y_valid, activation_function)

        if print_debug:
            print(f"The RMSE for model {layers_neurons} with {regularization_list[reg_type]} regularization and lambda {lambda_reg}: ", rmse)

        return {
            'parameters': parameters,
            'rmse': rmse,
            'layers_neurons': layers_neurons,
            'lambda': lambda_reg,
            'error_list': error_list,
            'accuracy_list': accuracy_list, #cambiare accuracy list
            'reg_type': reg_type
        }

    def update_best_model(result):
        nonlocal best_rmse, best_parameters, best_neurons, best_lambda, error_list_final_model, accuracy_list_final_model, reg_type
        if result['rmse'] > best_rmse:
            best_rmse = result['rmse']
            best_parameters = result['parameters']
            best_neurons = result['layers_neurons']
            best_lambda = result['lambda']
            error_list_final_model = result['error_list']
            accuracy_list_final_model = result['accuracy_list'] #cambiare
            reg_type = result['reg_type']

    results = []

    # Parallel processing using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for layers_neurons in layers_neurons_list:
            futures.append(executor.submit(evaluate_model_CV, layers_neurons, 0, 0))
            for lambda_reg in lambda_list:
                futures.append(executor.submit(evaluate_model_CV, layers_neurons, lambda_reg, 1))
                futures.append(executor.submit(evaluate_model_CV, layers_neurons, lambda_reg, 2))

        # Aspetta che tutti i thread siano completati
        concurrent.futures.wait(futures)

        # Raccolta dei risultati da tutti i thread completati
        for future in futures:
            result = future.result()  # Estrai il risultato da ciascun future
            results.append(result)

    for result in results:
        add_csv_line(result['layers_neurons'], regularization_list[result['reg_type']], result['lambda'], result['rmse'], dir, activation_function)
        update_best_model(result)

    end = time.time()
    min, sec = divmod(end - start, 60)
    print(f"End cross validation. Time spent for cross validation is {int(min)}:{sec:.2f} min\n")

    if reg_type == 0:
        text = f"Best configuration is {best_neurons} using no regularization"
    else:
        text = f"Best configuration is {best_neurons} using {reg_type} with lambda {best_lambda}"

    print(text)
    with open(f'plots/{dir}/{activation_function}/final_result', "w") as file:
        file.write(f"Time spent for cross validation is {int(min)}:{sec:.2f} min\n\n")
        file.write(text + "\n\n")

    plotError(error_list_final_model, len(error_list_final_model), dir, activation_fn=activation_function)

    return best_parameters

