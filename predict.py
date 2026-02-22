"""
Smart Mandi - Price Prediction Interface
Predicts next-day (24-hour) price per kg for a vegetable.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from .data_loader import prepare_for_modeling, get_vegetable_category
from .features import build_features, get_feature_columns, SEASONS
from .models import load_model


def predict_price(
    vegetable: str,
    month: int,
    year: int,
    recent_prices: list[float] | None = None,
    model_path: Path | str = "models",
) -> float:
    """
    Predict price per kg for a vegetable given month, year, and optional recent prices.

    Args:
        vegetable: e.g., "Potato", "Onion", "Brinjal"
        month: 1-12
        year: e.g., 2024
        recent_prices: Optional list of recent daily prices (most recent last)
        model_path: Path to saved model directory

    Returns:
        Predicted price per kg (₹)
    """
    model, scaler = load_model(Path(model_path))
    feature_cols = get_feature_columns()

    # Build minimal feature row
    lag_1 = recent_prices[-1] if recent_prices else None
    lag_7 = recent_prices[-7] if recent_prices and len(recent_prices) >= 7 else lag_1
    lag_14 = recent_prices[-14] if recent_prices and len(recent_prices) >= 14 else lag_7

    rolling_mean_7 = np.mean(recent_prices[-7:]) if recent_prices and len(recent_prices) >= 7 else lag_1
    rolling_mean_14 = np.mean(recent_prices[-14:]) if recent_prices and len(recent_prices) >= 14 else rolling_mean_7
    rolling_mean_30 = np.mean(recent_prices[-30:]) if recent_prices and len(recent_prices) >= 30 else rolling_mean_14

    # Defaults when no history
    default_price = 30.0  # fallback
    lag_1 = lag_1 or default_price
    lag_7 = lag_7 or lag_1
    lag_14 = lag_14 or lag_7
    rolling_mean_7 = rolling_mean_7 or lag_1
    rolling_mean_14 = rolling_mean_14 or rolling_mean_7
    rolling_mean_30 = rolling_mean_30 or rolling_mean_14

    season = SEASONS.get(month, "Winter")
    season_cols = [f"season_{s}" for s in ["Winter", "Summer", "Monsoon", "Autumn"]]

    row = {
        "month": month,
        "year": year,
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
        "lag_1": lag_1,
        "lag_7": lag_7,
        "lag_14": lag_14,
        "rolling_mean_7": rolling_mean_7,
        "rolling_mean_14": rolling_mean_14,
        "rolling_mean_30": rolling_mean_30,
        "rolling_std_7": 0.0,
        "rolling_std_14": 0.0,
        "rolling_std_30": 0.0,
    }
    for col in season_cols:
        row[col] = 1 if col == f"season_{season}" else 0

    X = pd.DataFrame([row])[feature_cols]
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    return max(0.0, round(pred, 2))
