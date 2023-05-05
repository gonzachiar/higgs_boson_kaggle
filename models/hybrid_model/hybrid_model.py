import tensorflow as tf
import xgboost as xgb
from models.base_wrapper import Model
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras import layers, regularizers
from keras.callbacks import ModelCheckpoint

tf.random.set_seed(42)

class NueralNetwork(Model):
  def __init__(self, shape, batch_size=225000, epochs=1000, saving_path="model.h5", model_weights_path=None):
    super().__init__()
    self.saving_path = saving_path
    self.model_weights_path = model_weights_path

    self.preprocessor = Pipeline([
    ('filler', SimpleImputer()),
    ('standard_scaler', StandardScaler())
    ])

    reg = regularizers.L1L2(l1=1e-5, l2=1e-4)
    self.predictor = keras.models.Sequential([
      layers.Dense(32, activation="relu",input_shape=shape, kernel_regularizer=reg),
      layers.BatchNormalization(),
      keras.layers.Dropout(0.2),
      layers.Dense(64, activation="relu",input_shape=shape, kernel_regularizer=reg),
      layers.BatchNormalization(),
      layers.Dense(128, activation="relu",input_shape=shape, kernel_regularizer=reg),
      layers.BatchNormalization(),
      layers.Dense(128, activation="relu",input_shape=shape, kernel_regularizer=reg),
      layers.BatchNormalization(),
      layers.Dense(64, activation="relu",input_shape=shape, kernel_regularizer=reg),
      layers.BatchNormalization(),
      layers.Dense(1, activation="sigmoid"),
    ])
    self.predictor.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.001))
    
    sch_callback = tf.keras.callbacks.ReduceLROnPlateau(patience=5,min_delta=0.000001,monitor='loss', min_lr=0.00001)
    checkpoint = ModelCheckpoint(
        self.saving_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min')
    callbacks = [sch_callback,checkpoint]
    self.train_params = {"epochs" : epochs, "callbacks": callbacks, "batch_size" : batch_size}

  def predict_embeddings(self, X):
    predictor = keras.models.Sequential(self.predictor.layers[:-1])
    X_transformed = self.preprocessor.transform(X)
    y_pred = predictor.predict(X_transformed, batch_size=225000)

    return y_pred

class XGboost(Model):
  def __init__(self, max_depth, n_estimators, scale_pos_weight,sample_weight):
    super().__init__()
    self.preprocessor = None

    self.predictor = xgb.XGBRegressor(objective="binary:logistic", max_depth=max_depth, n_estimators=n_estimators,scale_pos_weight=scale_pos_weight)

    self.train_params = {"sample_weight" : sample_weight}