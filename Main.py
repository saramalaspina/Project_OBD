from AirQualityPreprocessing import *
from CrossValidation import *
from HousingPreprocessing import HousingData
from SeoulBikePreprocessing import SeoulBikeData


def main():
    np.random.seed(123456)
    np.set_printoptions(suppress=True)

    dataset = menu(
        "Which dataset do you want to use?\n1 --> Air Quality\n2 --> Seoul Bike Sharing Demand\n3 --> Median House Value",
        ["1", "2", "3"]
    )

    if dataset == "1":
        X_train, X_val, X_test, y_train, y_val, y_test = AirQualityData()
        dir = "airquality"
        minibatch_size = 32
    elif dataset == "2":
        X_train, X_val, X_test, y_train, y_val, y_test = SeoulBikeData()
        dir = "seoulbike"
        minibatch_size = 64
    else:
        X_train, X_val, X_test, y_train, y_val, y_test = HousingData()
        dir = "housing"
        minibatch_size = 512

    # Possible configurations for cross validation
    input_layer = X_train.shape[0]
    num_neurons_list = [[input_layer, 128, 1], [input_layer, 128, 64, 1], [input_layer, 64, 32, 1], [input_layer, 128, 64, 64, 1], [input_layer, 128, 64, 32, 1]]
    lambda_list = [1e-3, 1e-2, 1e-1]
    activation_fn_list = ["relu","tanh"]
    num_epochs_list = [80]
    best_parameters, activation_function = cross_validation(X_train, y_train, X_val, y_val, num_neurons_list, lambda_list, activation_fn_list, num_epochs_list, minibatch_size, dir)

    # Print the test rmse and mae
    test_rmse = evaluate_model_rmse(X_test, best_parameters, y_test, activation_function)
    test_mae = evaluate_model_mae(X_test, best_parameters, y_test, activation_function)
    text = "\nThe RMSE on test set is: " + str(test_rmse)+"\nThe MAE on test set is: " + str(test_mae)
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