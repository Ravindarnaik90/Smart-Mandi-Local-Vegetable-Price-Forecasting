# Deploy Smart Mandi on Streamlit Cloud

## Quick Deploy

1. **Push to GitHub** — Ensure your repo includes:
   - `app.py` (main Streamlit app)
   - `data_loader.py`, `features.py`, `models.py`, `predict.py`
   - `requirements.txt`
   - `Vegetable and Fruits Prices in India.csv` (optional, for live data)
   - `models/` folder with trained `price_model.joblib` and `scaler.joblib` (optional)

2. **Go to [share.streamlit.io](https://share.streamlit.io)**

3. **Sign in** with GitHub

4. **New app** → Select repo and branch

5. **Main file path:** `smart_mandi/app.py`  
   (If your app is at repo root, use `app.py`)

6. **Advanced settings** (optional):
   - Python version: 3.11
   - `requirements.txt` path: `smart_mandi/requirements.txt`

7. **Deploy**

---

## Local Run

```bash
cd smart_mandi
pip install -r requirements.txt
streamlit run app.py
```

---

## Tips

- **Without CSV/model:** App runs in demo mode with fallback predictions and sample 3D viz
- **With CSV only:** Real 3D price surface; predictions use seasonality fallback
- **Full setup:** Train model in notebook, add `models/` to repo; get ML predictions + real data
