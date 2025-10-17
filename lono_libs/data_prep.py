import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from typing import Optional
def create_polynomial_features(X_train: pd.DataFrame, X_test: pd.DataFrame, degree: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly_transformed = poly.fit_transform(X_train)
    X_test_poly_transformed = poly.transform(X_test)
    feature_names = poly.get_feature_names_out(X_train.columns)
    X_train_poly = pd.DataFrame(X_train_poly_transformed, columns=feature_names, index=X_train.index)
    X_test_poly = pd.DataFrame(X_test_poly_transformed, columns=feature_names, index=X_test.index)
    return X_train_poly, X_test_poly
def apply_standard_scaler(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_df_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_df_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    return X_train_df_scaled, X_test_df_scaled
def preprocess_data(X_train: pd.DataFrame, X_test: pd.DataFrame, degree: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train_poly, X_test_poly = create_polynomial_features(X_train, X_test, degree)
    X_train_scaled, X_test_scaled = apply_standard_scaler(X_train_poly, X_test_poly)
    return X_train_scaled, X_test_scaled
def get_preprocessing_pipeline(
    impute_strategy: Optional[str] = None,
    polynomial_degree: Optional[int] = None,
    scale_features: bool = True
) -> Pipeline:
    steps = []
    if impute_strategy:
        steps.append(('imputer', SimpleImputer(strategy=impute_strategy)))
    if polynomial_degree is not None and polynomial_degree > 1:
        steps.append(('poly', PolynomialFeatures(degree=polynomial_degree, include_bias=False)))
    if scale_features:
        steps.append(('scaler', StandardScaler()))
    if not steps:
        raise ValueError("Pipeline needs at least one preprocessing step (imputation, polynomial features, or scaling).")
    return Pipeline(steps)
