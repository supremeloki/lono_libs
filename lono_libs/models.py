from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Lasso, Ridge, ElasticNet, LassoLars,
    SGDClassifier, HuberRegressor, BayesianRidge,
)
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier,
    GradientBoostingRegressor, AdaBoostClassifier, BaggingClassifier,
)
from sklearn.svm import (
    SVC, SVR
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from typing import Any, Dict, Literal, Optional
_CLASSIFIER_CLASSES = {
    "LogisticRegression": LogisticRegression,
    "RandomForestClassifier": RandomForestClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "XGBoostClassifier": XGBClassifier,
    "LightGBMClassifier": LGBMClassifier,
    "CatBoostClassifier": CatBoostClassifier,
    "SVC": SVC,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "KNeighborsClassifier": KNeighborsClassifier,
    "NaiveBayes": GaussianNB,
    "AdaBoostClassifier": AdaBoostClassifier,
    "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis,
    "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis,
    "MLPClassifier": MLPClassifier,
    "BaggingClassifier": BaggingClassifier,
    "StochasticGradientDescent": SGDClassifier,
    "DummyClassifier": DummyClassifier,
}
_REGRESSOR_CLASSES = {
    "LinearRegression": LinearRegression,
    "RandomForestRegressor": RandomForestRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "XGBoostRegressor": XGBRegressor,
    "LightGBMRegressor": LGBMRegressor,
    "CatBoostRegressor": CatBoostRegressor,
    "SVR": SVR,
    "DecisionTreeRegressor": DecisionTreeRegressor,
    "KNeighborsRegressor": KNeighborsRegressor,
    "Lasso": Lasso,
    "Ridge": Ridge,
    "ElasticNet": ElasticNet,
    "LassoLars": LassoLars,
    "HuberRegressor": HuberRegressor,
    "BayesianRidge": BayesianRidge,
}
_MODEL_SPECIFIC_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "LogisticRegression": {"max_iter": 1000},
    "XGBoostClassifier": {"use_label_encoder": False, "eval_metric": 'logloss'},
    "CatBoostClassifier": {"silent": True}, # iterations will be mapped from n_estimators
    "SVC": {"kernel": 'rbf'},
    "KNeighborsClassifier": {"n_neighbors": 5},
    "AdaBoostClassifier": {"n_estimators": 50},
    "MLPClassifier": {"hidden_layer_sizes": (100,), "max_iter": 1000},
    "BaggingClassifier": {"n_estimators": 50},
    "StochasticGradientDescent": {"max_iter": 1000, "tol": 1e-3},
    "DummyClassifier": {"strategy": 'stratified'},
    "Lasso": {"alpha": 0.1},
    "Ridge": {"alpha": 1.0},
    "ElasticNet": {"alpha": 0.1, "l1_ratio": 0.7},
    "LassoLars": {"alpha": 0.1},
    "KNeighborsRegressor": {"n_neighbors": 5},
    "CatBoostRegressor": {"silent": True},
}
def create_model_instance(
    model_name: str,
    model_type: Literal["classification", "regression"],
    random_state: Optional[int] = None,
    n_estimators: Optional[int] = None,
    **kwargs
) -> Any:
    model_class = None
    if model_type == "classification":
        model_class = _CLASSIFIER_CLASSES.get(model_name)
    elif model_type == "regression":
        model_class = _REGRESSOR_CLASSES.get(model_name)
    if model_class is None:
        raise ValueError(f"Model '{model_name}' of type '{model_type}' not found.")
    final_params = _MODEL_SPECIFIC_DEFAULTS.get(model_name, {}).copy()
    if random_state is not None and 'random_state' in model_class.__init__.__annotations__:
        final_params['random_state'] = random_state
    if n_estimators is not None:
        if model_name.startswith("CatBoost"):
            final_params['iterations'] = n_estimators
        elif 'n_estimators' in model_class.__init__.__annotations__:
            final_params['n_estimators'] = n_estimators
    final_params.update(kwargs)
    return model_class(**final_params)
def get_classification_models(random_state: int = 42, n_estimators: int = 100) -> Dict[str, Any]:
    models_dict = {}
    for name in _CLASSIFIER_CLASSES.keys():
        try:
            models_dict[name] = create_model_instance(name, "classification", random_state=random_state, n_estimators=n_estimators)
        except Exception:
            pass
    return models_dict
def get_regression_models(random_state: int = 42, n_estimators: int = 100) -> Dict[str, Any]:
    models_dict = {}
    for name in _REGRESSOR_CLASSES.keys():
        try:
            models_dict[name] = create_model_instance(name, "regression", random_state=random_state, n_estimators=n_estimators)
        except Exception:
            pass
    return models_dict
