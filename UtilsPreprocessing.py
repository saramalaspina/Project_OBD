import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

plt.rc("axes", titleweight = "bold", titlesize = 18, titlepad = 10)

def train_val_test_split(X, y):
    # Split the dataset into training (65%), validation (15%) and test set (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.12, random_state=42)

    # Training set
    train_data = pd.concat([X_train, y_train], axis = 1)

    return train_data, X_test, y_test, X_val, y_val


def encoding_and_standardization(X_train, X_val, X_test, numerical = None, categorical = None):
    if numerical is None and categorical is None:
        return None

    # Numerical features standardization
    transformer_num = make_pipeline(
        SimpleImputer(strategy="constant"), # default fill value=0
        StandardScaler(),
    )      

    # One hot encoding for categorical features
    transformer_cat = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="NA"),
        OneHotEncoder(handle_unknown='ignore'),
    )

    if numerical is not None and categorical is not None:
        preprocessor = make_column_transformer(
            (transformer_num, numerical), 
            (transformer_cat, categorical)
        )
    elif numerical is not None:
        preprocessor = make_column_transformer(
            (transformer_num, numerical)
        )
    else:
        preprocessor = make_column_transformer(
            (transformer_cat, categorical)
        )

    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)
    X_test = preprocessor.transform(X_test)

    return X_train, X_val, X_test

# Function to draw a histogram to represent the data distribution of the features
def plot_features(data, features, dir):
    for feature in features:
        plt.figure(figsize  = (8,6))
        plt.hist(data[feature], bins = 30, color = "tab:blue", edgecolor = "black", alpha = 0.7)
        plt.title(feature, fontsize = 12)
        plt.xlabel(feature, fontsize = 10)
        plt.ylabel("count", fontsize = 10)
        plt.grid(linewidth = 0.3)
        plt.savefig('plots/' + dir + '/features/' + feature + '.png')

# Function to transpose the matrix X and the vector y
def reshape_dataset(X_train, y_train, X_val, y_val, X_test, y_test):
    y_train = y_train.to_numpy()
    y_val = y_val.to_numpy()
    y_test = y_test.to_numpy()

    X_train = X_train.T
    y_train = y_train.reshape(1, -1)
    X_val = X_val.T
    y_val = y_val.reshape(1, -1)
    X_test = X_test.T
    y_test = y_test.reshape(1, -1)

    return X_train, y_train, X_val, y_val, X_test, y_test
