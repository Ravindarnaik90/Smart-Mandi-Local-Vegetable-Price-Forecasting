"""
Smart Mandi - Data Loader
Loads and preprocesses vegetable price data from Kaggle dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# Map specific item names to standardized vegetable categories
VEGETABLE_MAPPING = {
    "potato": ["Potato", "potato"],
    "onion": ["Onion", "onion"],
    "brinjal": ["Brinjal", "brinjal"],
    "tomato": ["Tomato", "tomato", "Thakkali"],
    "tomatoes": ["Tomato", "tomato"],
}


def load_data(csv_path: str | Path) -> pd.DataFrame:
    """Load the vegetable prices CSV."""
    df = pd.read_csv(csv_path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and filter the dataset."""
    df = df.copy()

    # Parse dates
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"])

    # Keep only rows with valid prices
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.dropna(subset=["Price"])
    df = df[df["Price"] > 0]

    # Drop empty or invalid item names
    df = df[df["Item_Name"].notna() & (df["Item_Name"].str.strip() != "")]

    return df[["date", "Item_Name", "Price"]].reset_index(drop=True)


def get_vegetable_category(item_name: str) -> str:
    """
    Map item name to a base vegetable category for aggregation.
    E.g., 'Potato(M)', 'Potato(B)' -> 'Potato'
    """
    if pd.isna(item_name):
        return "Other"
    item_lower = str(item_name).strip().lower()
    for category, keywords in VEGETABLE_MAPPING.items():
        for kw in keywords:
            if kw.lower() in item_lower:
                return category.title()
    # First word is often the vegetable (e.g., "Potato pack" -> "Potato")
    first_word = item_lower.split()[0] if item_lower else ""
    for category in ["potato", "onion", "brinjal", "tomato"]:
        if category in first_word:
            return category.title()
    return "Other"


def aggregate_by_vegetable_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily prices by vegetable category (median price per kg per day).
    """
    df = df.copy()
    df["vegetable"] = df["Item_Name"].apply(get_vegetable_category)
    df = df[df["vegetable"] != "Other"]

    agg = (
        df.groupby(["date", "vegetable"])
        .agg(price_per_kg=("Price", "median"), count=("Price", "count"))
        .reset_index()
    )
    return agg


def prepare_for_modeling(
    csv_path: str | Path,
    min_samples_per_vegetable: int = 100,
) -> pd.DataFrame:
    """
    Full pipeline: load, clean, aggregate. Returns data ready for modeling.
    """
    df = load_data(csv_path)
    df = clean_data(df)
    agg = aggregate_by_vegetable_date(df)

    # Filter vegetables with enough history
    veg_counts = agg.groupby("vegetable")["price_per_kg"].count()
    valid_vegetables = veg_counts[veg_counts >= min_samples_per_vegetable].index
    agg = agg[agg["vegetable"].isin(valid_vegetables)]

    return agg.sort_values(["vegetable", "date"]).reset_index(drop=True)
