from CrossValidation import *
from AirQualityPreprocessing import *

def main():
    np.random.seed(123456)
    np.set_printoptions(suppress=True)

    dataset = menu(
        "What dataset do you want to use?\n[1] Air Quality\n[2] Seoul Bike Sharing Demand\n[3] Metro Interstate Traffic Volume",
        ["1", "2", "3"]
    )

    #definisci possibili scelte per cross validation

    if dataset == 1:
        X_train, X_val, X_test, y_train, y_val, y_test = AirQualityData()
        best_parameters = cross_validation(X_train,y_train,X_val,y_val, )
    elif dataset == 2:
        #TODO
    else:
        #TODO


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