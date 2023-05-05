from models.base_wrapper import Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


class RandomForest(Model):
  def __init__(self, max_depth, sample_weight):
    super().__init__()
    self.preprocessor = Pipeline([
            ('impute', SimpleImputer(missing_values=-999,strategy='median'))])

    self.predictor = RandomForestClassifier(max_depth = max_depth)

    self.train_params = {"sample_weight" : sample_weight}