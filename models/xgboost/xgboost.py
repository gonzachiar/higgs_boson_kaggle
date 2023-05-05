import numpy as np
import xgboost as xgb
from models.base_wrapper import Model
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def trig_func(X):
    Z = np.radians(X)
    sin = np.sin(X)
    cos = np.cos(X)
    tan = np.tan(X)
    return np.column_stack((Z,sin, cos, tan))
def ilog_func(X):
    return np.log(1/(1+X))
def neutro(X):
    return X


class XGboost(Model):
  def __init__(self, max_depth, n_estimators, scale_pos_weight,sample_weight,angle_cols=None,inv_log_cols=None, todas=None):
    super().__init__()
    self.preprocessor = Pipeline([
                ('impute', SimpleImputer(missing_values=-999,strategy='median')),
                ('add_cols', ColumnTransformer([
                            ('numpy', FunctionTransformer(neutro),todas), # no es lo mas prolijo del mundo pero lo hago para que no cambie el orden de las columnas (no encontre ninguna forma)
                            ('angle', FunctionTransformer(trig_func, validate=False),(angle_cols)),
                            
                            ('logs', FunctionTransformer(ilog_func,validate=False),(inv_log_cols)),

                            ]))
            ])

    self.predictor = xgb.XGBRegressor(objective="binary:logistic", max_depth=max_depth, n_estimators=n_estimators,scale_pos_weight=scale_pos_weight)

    self.train_params = {"sample_weight" : sample_weight}