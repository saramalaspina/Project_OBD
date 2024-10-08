import os
import numpy as np
from UtilsPreprocessing import *


def AirQualityData():
    # Read the data
    DATASET_FILENAME="AirQualityUCI.csv"
    DATASET_DIR="./dataset"
    data = pd.read_csv(os.path.join(DATASET_DIR, DATASET_FILENAME), sep = ';', decimal = ',')

    # Handling missing values
    data.replace(-200, np.nan, inplace = True) #-200 corresponds to a null value
    data.drop(['NMHC(GT)','Unnamed: 15','Unnamed: 16'] , axis = 1, inplace = True) #removal of features with few non-null values
    #removal of null rows
    data = data.dropna(how='all') 
    data = data.dropna(subset=['PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']) 

    # Replacement of null values with the median
    for feature in ["CO(GT)", "NOx(GT)", "NO2(GT)"]:
        median = data[feature].median()  
        data[feature] = data[feature].fillna(median)

    # Date and time transformation
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
    data['Month'] = data['Date'].dt.month.astype(int)
    data.drop('Date', axis = 1, inplace = True)

    data['Time'] = pd.to_datetime(data['Time'], format='%H.%M.%S')
    data['Hour'] = data['Time'].dt.hour.astype(int)
    data.drop('Time', axis = 1, inplace = True)

    X = data.copy()
    y = X.pop('C6H6(GT)') # target column

    # Split the dataset into training, validation and test set
    train_data, X_test, y_test, X_val, y_val = train_val_test_split(X, y)

    # Removal of duplicated rows of training set
    if train_data.duplicated().sum() != 0:
        train_data = train_data.drop_duplicates()

    numerical = train_data.loc[:, (train_data.dtypes == int) | (train_data.dtypes == float)].columns.tolist()

    plot_features(train_data, numerical, dir = "airquality")

    numerical_features = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH', 'CO(GT)', 'NOx(GT)', 'NO2(GT)']

    X_train = train_data.copy()
    y_train = X_train.pop("C6H6(GT)")

    # Encoding and standardization
    X_train, X_val, X_test = encoding_and_standardization(X_train, X_val, X_test, numerical = numerical_features)

    X_train, y_train, X_val, y_val, X_test, y_test = reshape_dataset(X_train, y_train, X_val, y_val, X_test, y_test)

    return X_train, X_val, X_test, y_train, y_val, y_test