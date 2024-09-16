import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc("axes", titleweight = "bold", titlesize = 18, titlepad = 10)

def train_val_test_split(X, y):
    #split the dataset into training, validation and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.12, random_state=42)

    #training set
    train_data = pd.concat([X_train, y_train], axis = 1)

    return train_data, X_test, y_test, X_val, y_val


def encoding_and_standardization(X_train, X_val, X_test, numerical = None, categorical = None):

    if (numerical == None and categorical == None):
        return None

    preprocessor = None

    #numerical features standardization
    transformer_num = make_pipeline(
        SimpleImputer(strategy="constant"), # default fill value=0
        StandardScaler(),
    )      

    #one hot encoding
    transformer_cat = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="NA"),
        OneHotEncoder(handle_unknown='ignore'),
    )

    if (numerical != None and categorical != None):
        preprocessor = make_column_transformer(
            (transformer_num, numerical), 
            (transformer_cat, categorical)
        )
    elif numerical != None:
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


def plot_features(data, features, dir):
    for feature in features:
        plt.figure(figsize  = (8,6))
        plt.hist(data[feature], bins = 30, color = "tab:blue", edgecolor = "black", alpha = 0.7)
        plt.title(feature, fontsize = 12)
        plt.xlabel(feature, fontsize = 10)
        plt.ylabel("count", fontsize = 10)
        plt.grid(linewidth = 0.3)
        plt.savefig('Plots/' + dir + '/' + feature + '.png')