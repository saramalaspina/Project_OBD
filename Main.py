from AirQualityPreprocessing import *
from CrossValidation import *


def main():
    np.random.seed(123456)
    np.set_printoptions(suppress=True)

    dataset = menu(
        "Which dataset do you want to use?\n[1] Air Quality\n[2] Seoul Bike Sharing Demand\n[3] Metro Interstate Traffic Volume",
        ["1", "2", "3"]
    )

    if dataset == "1":
        X_train, X_val, X_test, y_train, y_val, y_test = AirQualityData()
    elif dataset == "2":
        return 0
    else:
        return 0

    # definisci possibili scelte per cross validation
    input_layer = X_train.shape[0]
    num_neurons_list = [[input_layer, 10, 20, 1], [input_layer, 50, 30, 1]]  # da scegliere
    lambda_list = [0.01, 0.1, 0.5]
    activation_fn_list = ["relu", "tanh"]
    best_parameters, activation_function = cross_validation(X_train, y_train, X_val, y_val, lambda_list = lambda_list, num_neurons_list = num_neurons_list, activation_fn_list=activation_fn_list, dir = dir)

    # print the test accuracy
    rmse = evaluate_model_rmse(X_test, best_parameters, y_test, activation_function)
    text = "The test rmse is: " + str(rmse)
    print(text)
    with open('plots/' + dir + '/' + activation_function + '/final_result', "a") as file:
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