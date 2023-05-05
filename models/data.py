import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
np.random.seed(42)

def get_data(train_csv_path, test_csv_path):
    # Cargo los datos
    df_train = pd.read_csv(train_csv_path)
    df_test = pd.read_csv(test_csv_path)

    # Separo la data en train y validation, dropeando columnas
    X_data = df_train.drop(["Label"],axis=1,inplace=False)
    y_data = (df_train["Label"] == "s")*1
    X_train, X_val, y_train , y_val = train_test_split(X_data, y_data, test_size=0.1,random_state=42)

    weights_train, weights_val = X_train["Weight"], X_val["Weight"]

    X_train.drop(["EventId","Weight"],axis=1,inplace=True)
    X_val.drop(["EventId","Weight"],axis=1,inplace=True)
    X_test = df_test.drop(["EventId"],axis=1)

    s_weight = np.sum(weights_train[y_train==1])
    b_weight = np.sum(weights_train[y_train==0])
    

    return X_train, y_train, X_val, y_val, X_test, s_weight, b_weight,weights_train ,weights_val
    #para RF:
    weights_train_RF = weights_train.copy()
    indices_signal = (y_train==1)
    weights_train_RF[indices_signal] *= b_weight/s_weight