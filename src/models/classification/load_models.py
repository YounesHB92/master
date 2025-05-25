from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

class LoadModels:
    def __init__(self, *args, **kwargs):
        self.models_all = {
            "random_forest": RandomForestClassifier(max_depth=5, n_estimators=100),
            # "gradient_boosting": GradientBoostingClassifier(max_depth=3, n_estimators=100),
            # "logistic_regression": LogisticRegression(max_iter=1000),
        }
        super().__init__(*args, **kwargs)