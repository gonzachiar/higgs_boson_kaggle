import os
import sys

sys.path.insert(0, os.getcwd()) #noqa

from models.data import get_data
from models.hybrid_model.hybrid_model import NueralNetwork, XGboost
import pandas as pd

def main():
    train_csv_path = "training.csv" #Elegir el path
    test_csv_path = "test.csv" #Elegir el path

    X_train, y_train, X_val, y_val, X_test, s_weight, b_weight,weights_train, weights_val = get_data(train_csv_path, test_csv_path)

    # Declaro y entreno la red
    net = NueralNetwork(shape=X_train.shape[1:],batch_size=225000, epochs=1000,saving_path="model.h5")
    net.fit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

    # Calculo embeddings
    embs = net.predict_embeddings(X_train)

    # Entreno XGboost a partir de los embeddings
    model = XGboost(max_depth=3, n_estimators=120, sample_weight=weights_train,scale_pos_weight=(b_weight/s_weight)**(53/100))
    model.preprocessor = None
    model.fit(X_train=embs, y_train=y_train)

    embs_val = net.predict_embeddings(X_val)
    y_val_pred = model.predict(embs_val)
    ams_val = model.score(weights_val, y_val, y_val_pred)

    print(f"AMS_VALIDATION: {ams_val}")
    model.test(X_test,df_test=pd.read_csv(test_csv_path),csv_path="rf.csv")

if __name__ == "__main__":
    main()