"""
Smart Mandi - Regression Models for Price Forecasting
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def get_models():
    """Return dict of models to train and evaluate."""
    models = {
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42),
    }
    if HAS_XGB:
        models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
        )
    return models


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, R2."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def train_and_evaluate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_cols: list[str],
    scaler: StandardScaler | None = None,
) -> tuple[dict, dict, StandardScaler]:
    """
    Train multiple models and return results.
    Returns: (results_dict, best_model_name, scaler)
    """
    X_tr = X_train[feature_cols].fillna(X_train[feature_cols].median())
    X_te = X_test[feature_cols].fillna(X_train[feature_cols].median())

    if scaler is None:
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
    else:
        X_tr_scaled = scaler.transform(X_tr)
    X_te_scaled = scaler.transform(X_te)

    results = {}
    for name, model in get_models().items():
        model.fit(X_tr_scaled, y_train)
        pred = model.predict(X_te_scaled)
        results[name] = evaluate_model(y_test.values, pred)
        results[name]["model"] = model

    return results, scaler


def save_model(model, scaler: StandardScaler, path: Path):
    """Save model and scaler to disk."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path / "price_model.joblib")
    joblib.dump(scaler, path / "scaler.joblib")


def load_model(path: Path) -> tuple:
    """Load model and scaler from disk."""
    path = Path(path)
    return joblib.load(path / "price_model.joblib"), joblib.load(path / "scaler.joblib")
