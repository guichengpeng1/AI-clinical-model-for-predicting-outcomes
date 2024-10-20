from sklearn.linear_model import Lasso, ElasticNet
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from pysurvival.models.semi_parametric import CoxPHModel as PysurvivalCoxPH
from pysurvival.models.non_parametric import ConditionalSurvivalForestModel
from lifelines import WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter

import numpy as np
from deephit import DeepHitSingle


class sksurvSurvival:
    def __init__(self):     
        pass
    
    def fit(self, X, T, Y):
        y = [(Y.iloc[i, 0], T.iloc[i, 0]) for i in range(len(Y))]
        y = np.array(y, dtype=[('status', 'bool'), ('time', '<f8')])
        self.model.fit(X, y)

    def predict(self, X, time_horizons):
        preds_ = self.model.predict(X)
        return preds_


# ====================== Lasso =========================
class LassoModel(sksurvSurvival):
    def __init__(self, alpha=1.0):
        super(LassoModel, self).__init__()
        self.name = 'Lasso'
        self.model = Lasso(alpha=alpha)
        self.direction = 1
        self.prob_FLAG = True
        
    def get_hyperparameter_space(self):
        return [{'name': 'Lasso.alpha', 'type': 'continuous', 'domain': (0.001, 1.0), 'dimensionality': 1}]


# ====================== Elastic Net ====================
class ElasticNetModel(sksurvSurvival):
    def __init__(self, alpha=1.0, l1_ratio=0.5):
        super(ElasticNetModel, self).__init__()
        self.name = 'ElasticNet'
        self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        self.direction = 1
        self.prob_FLAG = True
        
    def get_hyperparameter_space(self):
        return [
            {'name': 'ElasticNet.alpha', 'type': 'continuous', 'domain': (0.001, 1.0), 'dimensionality': 1},
            {'name': 'ElasticNet.l1_ratio', 'type': 'continuous', 'domain': (0, 1), 'dimensionality': 1}
        ]


# ====================== SuperPC =======================
class SuperPC(sksurvSurvival):
    def __init__(self, n_components=5):
        super(SuperPC, self).__init__()
        self.name = 'SuperPC'
        self.model = make_pipeline(PCA(n_components=n_components), CoxPHSurvivalAnalysis())
        self.direction = 1
        self.prob_FLAG = True

    def get_hyperparameter_space(self):
        return [{'name': 'SuperPC.n_components', 'type': 'discrete', 'domain': range(2, 20)}]


# ====================== SVM ===========================
class SVMModel(sksurvSurvival):
    def __init__(self, C=1.0, kernel='linear'):
        super(SVMModel, self).__init__()
        self.name = 'SVM'
        self.model = SVC(C=C, kernel=kernel, probability=True)
        self.direction = 1
        self.prob_FLAG = True

    def get_hyperparameter_space(self):
        return [{'name': 'SVM.C', 'type': 'continuous', 'domain': (0.001, 10.0), 'dimensionality': 1}]


# ====================== GBM ===========================
class GBMModel(sksurvSurvival):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        super(GBMModel, self).__init__()
        self.name = 'GBM'
        self.model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
        self.direction = 1
        self.prob_FLAG = True

    def get_hyperparameter_space(self):
        return [
            {'name': 'GBM.n_estimators', 'type': 'discrete', 'domain': range(50, 500, 50)},
            {'name': 'GBM.learning_rate', 'type': 'continuous', 'domain': (0.01, 1.0), 'dimensionality': 1},
            {'name': 'GBM.max_depth', 'type': 'discrete', 'domain': range(1, 10)}
        ]


# ====================== DeepHit =======================
class DeepHitModel(sksurvSurvival):
    def __init__(self, num_layers=3, num_units=128, learning_rate=1e-4):
        super(DeepHitModel, self).__init__()
        self.name = 'DeepHit'
        self.model = DeepHitSingle(num_layers=num_layers, num_units=num_units, learning_rate=learning_rate)
        self.direction = 1
        self.prob_FLAG = True

    def get_hyperparameter_space(self):
        return [
            {'name': 'DeepHit.num_layers', 'type': 'discrete', 'domain': range(2, 10)},
            {'name': 'DeepHit.num_units', 'type': 'discrete', 'domain': range(32, 256)},
            {'name': 'DeepHit.learning_rate', 'type': 'continuous', 'domain': (1e-5, 1e-2), 'dimensionality': 1}
        ]




# ====================== Conditional Inference Survival Forest (CIF) =======================
class ConditionalInferenceSurvForestModel(sksurvSurvival):
    def __init__(self, n_estimators=100, max_features='sqrt'):
        super(ConditionalInferenceSurvForestModel, self).__init__()
        self.name = 'ConditionalInferenceSurvForest'
        self.model = ConditionalSurvivalForestModel(n_estimators=n_estimators, max_features=max_features)
        self.direction = 1
        self.prob_FLAG = True

    def get_hyperparameter_space(self):
        return [
            {'name': 'CIF.n_estimators', 'type': 'discrete', 'domain': range(50, 500, 50)},
            {'name': 'CIF.max_features', 'type': 'discrete', 'domain': ['sqrt', 'log2']}
        ]


# ====================== Other models =========
class CoxPH(sksurvSurvival):
    def __init__(self):
        super(CoxPH, self).__init__()
        self.name = 'CoxPH'
        self.model = CoxPHSurvivalAnalysis()
        self.direction = 1
        self.prob_FLAG = True


class CoxPHRidge(sksurvSurvival):
    def __init__(self, alpha=10.0):
        super(CoxPHRidge, self).__init__()
        self.name = 'CoxPHRidge'
        self.model = CoxPHSurvivalAnalysis(alpha=alpha)
        self.direction = 1
        self.prob_FLAG = True


class Weibull(sksurvSurvival):
    def __init__(self):
        super(Weibull, self).__init__()
        self.name = 'Weibull'
        self.model = WeibullAFTFitter()
        self.direction = 1
        self.prob_FLAG = True


class LogNormal(sksurvSurvival):
    def __init__(self):
        super(LogNormal, self).__init__()
        self.name = 'LogNormal'
        self.model = LogNormalAFTFitter()
        self.direction = 1
        self.prob_FLAG = True


class LogLogistic(sksurvSurvival):
    def __init__(self):
        super(LogLogistic, self).__init__()
        self.name = 'LogLogistic'
        self.model = LogLogisticAFTFitter()
        self.direction = 1
        self.prob_FLAG = True


class RandomSurvForest(sksurvSurvival):
    def __init__(self, n_estimators=100):
        super(RandomSurvForest, self).__init__()
        self.name = 'RandomSurvForest'
        self.model = RandomSurvivalForest(n_estimators=n_estimators)
        self.direction = 1
        self.prob_FLAG = True
