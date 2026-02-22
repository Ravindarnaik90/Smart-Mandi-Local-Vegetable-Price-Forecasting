"""
Smart Mandi - Feature Engineering
Creates month, season, and historical trend features for regression.
"""

import pandas as pd
import numpy as np


SEASONS = {
    1: "Winter", 2: "Winter", 3: "Summer", 4: "Summer", 5: "Summer",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Autumn",
    10: "Autumn", 11: "Autumn", 12: "Winter",
}


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add month, year, and season from date."""
    df = df.copy()
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["season"] = df["month"].map(SEASONS)
    return df


def add_lag_and_rolling_features(
    df: pd.DataFrame,
    veg_col: str = "vegetable",
    price_col: str = "price_per_kg",
    lags: list[int] = [1, 7, 14],
    windows: list[int] = [7, 14, 30],
) -> pd.DataFrame:
    """
    Add lagged prices and rolling statistics per vegetable.
    """
    df = df.copy()
    result_rows = []

    for vegetable in df[veg_col].unique():
        veg_df = df[df[veg_col] == vegetable].sort_values("date").reset_index(drop=True)

        for i, lag in enumerate(lags):
            veg_df[f"lag_{lag}"] = veg_df[price_col].shift(lag)

        for window in windows:
            veg_df[f"rolling_mean_{window}"] = veg_df[price_col].rolling(window).mean().shift(1)
            veg_df[f"rolling_std_{window}"] = veg_df[price_col].rolling(window).std().shift(1)

        result_rows.append(veg_df)

    result = pd.concat(result_rows, ignore_index=True)
    return result.sort_values([veg_col, "date"]).reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    df = add_date_features(df)
    df = add_lag_and_rolling_features(df)

    # One-hot encode season
    season_dummies = pd.get_dummies(df["season"], prefix="season")
    df = pd.concat([df, season_dummies], axis=1)

    # Cyclical encoding for month (captures seasonality)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def get_feature_columns() -> list[str]:
    """Return list of feature column names for modeling."""
    base = ["month", "month_sin", "month_cos", "year"]
    lags = ["lag_1", "lag_7", "lag_14"]
    rolling = ["rolling_mean_7", "rolling_mean_14", "rolling_mean_30"]
    seasons = [f"season_{s}" for s in ["Winter", "Summer", "Monsoon", "Autumn"]]
    return base + lags + rolling + seasons
