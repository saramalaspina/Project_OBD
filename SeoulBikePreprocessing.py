import os
from UtilsPreprocessing import *

def SeoulBikeData():
    # Read the data
    DATASET_FILENAME="SeoulBikeData.csv"
    DATASET_DIR="./dataset"
    data = pd.read_csv(os.path.join(DATASET_DIR, DATASET_FILENAME), encoding='ISO-8859-1')

    # Date transformation
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
    data['Month'] = data['Date'].dt.month.astype(int)
    data.drop('Date', axis=1, inplace=True)

    X = data.copy()
    y = X.pop('Rented Bike Count')  # target column

    # Split the dataset into training, validation and test set
    train_data, X_test, y_test, X_val, y_val = train_val_test_split(X, y)

    # Removal of duplicated rows of training set
    if train_data.duplicated().sum() != 0:
        train_data = train_data.drop_duplicates()

    numerical = train_data.select_dtypes(include=['number']).columns
    categorical = train_data.select_dtypes(include=['object', 'category']).columns

    plot_features(train_data, numerical, dir = "seoulbike")
    plot_features(train_data, categorical, dir="seoulbike")

    numerical_features = ['Hour', 'Temperature', 'Humidity', 'Wind speed', 'Visibility', 'Dew point temperature', 'Solar Radiation', 'Rainfall', 'Snowfall', 'Month']
    categorical_features = ['Seasons', 'Holiday', 'Functioning Day']
    X_train = train_data.copy()
    y_train = X_train.pop("Rented Bike Count")

    # Encoding and standardization
    X_train, X_val, X_test = encoding_and_standardization(X_train, X_val, X_test, numerical = numerical_features, categorical=categorical_features)

    X_train, y_train, X_val, y_val, X_test, y_test = reshape_dataset(X_train, y_train, X_val, y_val, X_test, y_test)

    return X_train, X_val, X_test, y_train, y_val, y_test
