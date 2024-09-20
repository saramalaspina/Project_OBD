import os
from UtilsPreprocessing import *

def HousingData():
    # read the data
    DATASET_FILENAME = "Housing.csv"
    DATASET_DIR = "./dataset"
    data = pd.read_csv(os.path.join(DATASET_DIR, DATASET_FILENAME))

    data = data.dropna(subset=['total_bedrooms'])

    X = data.copy()
    y = X.pop('median_house_value')  # target column

    #split the dataset into training, validation and test set
    train_data, X_test, y_test, X_val, y_val = train_val_test_split(X, y)

    #removal of duplicated rows of training set
    if train_data.duplicated().sum() != 0:
        train_data = train_data.drop_duplicates()

    numerical = train_data.select_dtypes(include=['number']).columns
    categorical = train_data.select_dtypes(include=['object', 'category']).columns

    plot_features(train_data, numerical, dir = "housing")
    plot_features(train_data, categorical, dir="housing")

    numerical_features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']
    categorical_features = ['ocean_proximity']
    X_train = train_data.copy()
    y_train = X_train.pop("median_house_value")

    #encoding and standardization
    X_train, X_val, X_test = encoding_and_standardization(X_train, X_val, X_test, numerical = numerical_features, categorical=categorical_features)

    y_train = y_train.to_numpy()
    y_val = y_val.to_numpy()
    y_test = y_test.to_numpy()

    X_train = X_train.T
    y_train = y_train.reshape(1, -1)
    X_val = X_val.T
    y_val = y_val.reshape(1, -1)
    X_test = X_test.T
    y_test = y_test.reshape(1, -1)

    return X_train, X_val, X_test, y_train, y_val, y_test
