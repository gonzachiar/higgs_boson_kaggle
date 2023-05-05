import os
import sys

sys.path.insert(0, os.getcwd()) #noqa

from models.data import get_data
from models.xgboost.xgboost import XGboost


def main():
    train_csv_path = "training.csv" #Elegir el path
    test_csv_path = "test.csv" #Elegir el path

    X_train, y_train, X_val, y_val, X_test, s_weight, b_weight,weights_train, weights_val = get_data(train_csv_path, test_csv_path)

    angle_cols = (20,25,28)
    inv_log_cols = (0,1,2,3,4,5,7,8,9,10,12,13,16,19,21,23,26)
    todas=tuple(range(X_train.shape[1]))

    model = XGboost(max_depth=3, n_estimators=170, sample_weight=weights_train,scale_pos_weight=((b_weight/s_weight)**(53/100)),angle_cols=angle_cols,inv_log_cols=inv_log_cols, todas=todas)

    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)

    ams_val = model.score(weights_val, y_val, y_val_pred)

    print(f"AMS_VALIDATION: {ams_val}")
    model.test(X_test,"xgboost.csv")

if __name__ == "__main__":
    main()