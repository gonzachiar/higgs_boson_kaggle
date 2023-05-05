from tensorflow import keras
from keras.models import load_model
import numpy as np
import pandas as pd

class Model:
  """
  Base class for all models.

  Subclasses must define the following attributes:
  - self.predictor: the main model
  - self.preprocessor: a pipeline for preprocessing the input data
  - self.train_params: a dictionary of hyperparameters for training the model
  """
  @staticmethod
  def get_ams_score(weights, y_true, y_pred):
    def ams(s, b):
      return np.sqrt(2 * ((s + b + 10) * np.log(1.0 + s/(b + 10)) - s))
    s = np.sum(weights * (y_true == 1) * (y_pred == 1))
    b = np.sum(weights * (y_true == 0) * (y_pred == 1))
    return ams(s, b)

  def __init__(self):
    self.model_weights_path = None

  def fit(self, X_train, y_train, X_val=None, y_val=None):

    if self.preprocessor is not None:
      print("Fitting preprocessor")
      X_train_transformed = self.preprocessor.fit_transform(X_train) #, y_train) 
    else:
      X_train_transformed = X_train

    if X_val is not None:
      X_val_transformed = self.preprocessor.transform(X_val)
      self.train_params["validation_data"] = (X_val_transformed, y_val)

    print("Fitting model")
    if self.model_weights_path is not None:
      self.predictor = load_model(self.model_weights_path)
    else:
      self.predictor.fit(X_train_transformed, y_train, **self.train_params)

  def predict(self, X):
    if self.preprocessor is not None:
      X_transformed = self.preprocessor.transform(X)
    else:
      X_transformed = X

    y_pred = self.predictor.predict(X_transformed)

    return np.round(y_pred)
  
  def score(self, weights, y, y_pred):

    return self.get_ams_score(weights, y, y_pred)
  
  def test(self, X_test,df_test, csv_path="model.csv"):
    y_test_pred = self.predict(X_test)
    y_test_pred = ["s" if i == 1 else "b" for i in y_test_pred]
    submission_df = pd.DataFrame({"EventId" : df_test["EventId"], "RankOrder": [i+1 for i in range(550000)],"Class": y_test_pred})
    submission_df.to_csv(csv_path, index=False)