from models.base_wrapper import Model
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression


class LogReg(Model):
  def __init__(self, C=19.3):
    super().__init__()
    drop_candidates_70 = ["DER_deltaeta_jet_jet","DER_mass_jet_jet" , "DER_prodeta_jet_jet", "DER_lep_eta_centrality", "PRI_jet_subleading_pt","PRI_jet_subleading_eta","PRI_jet_subleading_phi"]

    drop_candidates_40 = ["PRI_jet_leading_pt", "PRI_jet_leading_eta", "PRI_jet_leading_phi"]

    drop_candidates_15 = ["DER_mass_MMC"]

    drop_log_reg = drop_candidates_70 + drop_candidates_40 + drop_candidates_15
   
    self.preprocessor = ColumnTransformer([
        ('dropear','drop',drop_log_reg)
    ],remainder='passthrough')
    self.predictor = LogisticRegression(C=C)

    self.train_params = {}