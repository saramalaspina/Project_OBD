from scipy.stats import moment

from AirQualityPreprocessing import *
from CrossValidation import *
from SeoulBikePreprocessing import SeoulBikeData


def main():
    np.random.seed(123456)
    np.set_printoptions(suppress=True)

    dataset = menu(
        "Which dataset do you want to use?\n[1] Air Quality\n[2] Seoul Bike Sharing Demand\n[3] Metro Interstate Traffic Volume",
        ["1", "2", "3"]
    )

    if dataset == "1":
        X_train, X_val, X_test, y_train, y_val, y_test = AirQualityData()
        dir = "airquality"
    elif dataset == "2":
        X_train, X_val, X_test, y_train, y_val, y_test = SeoulBikeData()
        dir = "seoulbike"
    else:
        return 0

    # definisci possibili scelte per cross validation
    input_layer = X_train.shape[0]
    num_neurons_list = [[input_layer, 32, 1], [input_layer, 64, 1], [input_layer, 128, 1], [input_layer, 32, 64, 1], [input_layer, 64, 128, 1], [input_layer, 32, 128, 1], [input_layer, 32, 64, 128, 1], [input_layer, 32, 32, 128, 1], [input_layer, 64, 128, 32, 1]]
    lambda_list = [1e-5, 1e-3, 1e-1]
    activation_fn_list = ["relu", "tanh"]
    minibatch_size_list = [64]
    num_epochs_list = [50]
    best_parameters, activation_function = cross_validation(X_train, y_train, X_val, y_val, lambda_list = lambda_list, num_neurons_list = num_neurons_list, activation_fn_list=activation_fn_list, num_epochs_list= num_epochs_list, minibatch_size_list= minibatch_size_list, dir = dir)

    # print the test rmse
    test_rmse = evaluate_model_rmse(X_test, best_parameters, y_test, activation_function)
    text = "The RMSE on test set is: " + str(test_rmse)
    print(text)
    with open('plots/' + dir + '/result/final_result', "a") as file:
        file.write(text + "\n")



def menu(message, options):
    while True:
        print(message)
        choice = input(f"Select a number between {options}: ").strip().lower()
        if choice in options:
            return choice
        else:
            print(f"Non valid choice. Try again.\n")

if __name__ == "__main__":
    main()