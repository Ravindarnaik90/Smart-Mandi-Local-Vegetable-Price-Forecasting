# Smart Mandi: Local Vegetable Price Forecasting

Predict **Price per KG** of vegetables in India based on month, season, and historical price trends. Helps farmers decide when to sell for better returns.

## Data Source

- **Kaggle:** [Vegetable and Fruit Prices in India](https://www.kaggle.com/datasets/anshtanwar/current-daily-price-of-various-commodities-india)
- Place `Vegetable and Fruits Prices in India.csv` in the parent folder (`d:\VIbe Coding\`)

## Setup

```bash
cd smart_mandi
pip install -r requirements.txt
```

## Usage

1. **Run the Jupyter notebook** `Smart_Mandi_Price_Forecast.ipynb` to:
   - Load and clean data
   - Engineer features (month, season, lags, rolling averages)
   - Train Ridge, Random Forest, and XGBoost models
   - Evaluate and save the best model
   - Get 24-hour price predictions

2. **Programmatic prediction:**
   ```python
   from smart_mandi.predict import predict_price
   pred = predict_price("Potato", month=3, year=2024, recent_prices=[25,27,26,28,29,30,31])
   print(f"Predicted: ₹{pred}/kg")
   ```

## Features Used

| Feature | Description |
|---------|-------------|
| `month` | 1–12 |
| `season` | Winter, Summer, Monsoon, Autumn |
| `lag_1`, `lag_7`, `lag_14` | Previous day, 7 days, 14 days ago |
| `rolling_mean_7/14/30` | Rolling average over 7, 14, 30 days |
| `month_sin`, `month_cos` | Cyclical encoding for seasonality |

## Note on State

The provided dataset (`Vegetable and Fruits Prices in India.csv`) appears to be from a single market (likely Karnataka). To forecast by **state**, you would need a dataset with a state/market column (e.g., the multi-state Agricultural Commodities dataset on Kaggle).
