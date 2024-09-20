import os
from UtilsPreprocessing import *

def SeoulBikeData():
    # read the data
    DATASET_FILENAME="SeoulBikeData.csv"
    DATASET_DIR="./dataset"
    data = pd.read_csv(os.path.join(DATASET_DIR, DATASET_FILENAME), encoding='ISO-8859-1')

    #date transformation
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
    data['Month'] = data['Date'].dt.month.astype(int)
    data.drop('Date', axis=1, inplace=True)

    # handling 0 target values
    data = data[(data["Rented Bike Count"] != 0)]

    X = data.copy()
    y = X.pop('Rented Bike Count')  # target column

    #split the dataset into training, validation and test set
    train_data, X_test, y_test, X_val, y_val = train_val_test_split(X, y)

    #removal of duplicated rows of training set
    if train_data.duplicated().sum() != 0:
        train_data = train_data.drop_duplicates()

    numerical = train_data.select_dtypes(include=['number']).columns
    categorical = train_data.select_dtypes(include=['object', 'category']).columns

    train_data = train_data[(train_data["Humidity"] >= 10)]
    train_data = train_data[(train_data["Rainfall"] <= 5)]
    train_data = train_data[(train_data["Snowfall"] <= 4)]

    plot_features(train_data, numerical, dir = "seoulbike")
    plot_features(train_data, categorical, dir="seoulbike")

    numerical_features = ['Hour', 'Temperature', 'Humidity', 'Wind speed', 'Visibility', 'Dew point temperature', 'Solar Radiation', 'Rainfall', 'Snowfall', 'Month']
    categorical_features = ['Seasons', 'Holiday', 'Functioning Day']
    X_train = train_data.copy()
    y_train = X_train.pop("Rented Bike Count")

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
