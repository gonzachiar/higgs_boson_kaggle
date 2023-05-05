import os
import sys

sys.path.insert(0, os.getcwd()) #noqa

from models.data import get_data
from models.random_forest.random_forest import RandomForest
import pandas as pd

def main():
    train_csv_path = "training.csv" #Elegir el path
    test_csv_path = "test.csv" #Elegir el path

    X_train, y_train, X_val, y_val, X_test, s_weight, b_weight,weights_train, weights_val = get_data(train_csv_path, test_csv_path)

    weights_train_RF = weights_train.copy()
    indices_signal = (y_train==1)
    weights_train_RF[indices_signal] *= b_weight/s_weight

    model = RandomForest(max_depth=49, sample_weight=weights_train_RF)

    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)

    ams_val = model.score(weights_val, y_val, y_val_pred)

    print(f"AMS_VALIDATION: {ams_val}")
    model.test(X_test,df_test=pd.read_csv(test_csv_path),csv_path="rf.csv")

if __name__ == "__main__":
    main()