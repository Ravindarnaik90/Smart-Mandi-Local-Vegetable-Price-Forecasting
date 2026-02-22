"""
Smart Mandi - Streamlit App
Agriculture-themed UI with 3D visualizations for vegetable price forecasting.
Deploy: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
import sys

# Add parent for imports
ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
sys.path.insert(0, str(PARENT))

# Page config - must be first Streamlit command
st.set_page_config(
    page_title="Smart Mandi | Agriculture Price Forecast",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============ CUSTOM CSS - Agriculture Theme & 3D Animated Environment ============
st.markdown("""
<style>
    /* Import distinctive font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    /* Root variables - Earth & harvest palette */
    :root {
        --soil: #3D2914;
        --leaf: #2D5A27;
        --golden: #E8B923;
        --wheat: #F5E6C8;
        --forest: #1A3D1A;
        --sky: #0F1F0A;
    }
    
    /* Animated gradient background - flowing fields */
    .stApp {
        background: linear-gradient(135deg, #0F1F0A 0%, #1A3D1A 25%, #2D5A27 50%, #1A3D1A 75%, #0F1F0A 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    /* Floating leaf animation - subtle depth */
    .floating-leaf {
        position: fixed;
        width: 30px;
        height: 30px;
        opacity: 0.15;
        animation: float 8s ease-in-out infinite;
        z-index: 0;
        pointer-events: none;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0) rotate(0deg); }
        50% { transform: translateY(-20px) rotate(10deg); }
    }
    
    /* Main container - glassmorphism card */
    .main-card {
        background: rgba(26, 47, 20, 0.85);
        backdrop-filter: blur(12px);
        border-radius: 24px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 1px solid rgba(45, 90, 39, 0.5);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.05);
    }
    
    /* Hero header */
    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-size: 3.2rem;
        font-weight: 700;
        color: #F5E6C8;
        text-align: center;
        margin-bottom: 0.25rem;
        text-shadow: 0 2px 20px rgba(0,0,0,0.4);
    }
    
    .hero-subtitle {
        font-family: 'Outfit', sans-serif;
        font-size: 1.2rem;
        color: #B8D4A8;
        text-align: center;
        margin-bottom: 2rem;
        letter-spacing: 0.5px;
    }
    
    /* Prediction card - highlight */
    .prediction-card {
        background: linear-gradient(145deg, #2D5A27, #1A3D1A);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        border: 2px solid #E8B923;
        box-shadow: 0 0 40px rgba(232, 185, 35, 0.2);
    }
    
    .prediction-value {
        font-family: 'Outfit', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        color: #E8B923;
    }
    
    .metric-card {
        background: rgba(45, 90, 39, 0.6);
        border-radius: 16px;
        padding: 1.25rem;
        text-align: center;
        border: 1px solid rgba(232, 185, 35, 0.3);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1A2F14, #0F1F0A);
    }
    
    /* Divider with agriculture motif */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #E8B923, transparent);
        opacity: 0.6;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============ HELPER: Load data & model ============
def get_csv_path():
    paths = [
        PARENT / "Vegetable and Fruits Prices in India.csv",
        ROOT / "Vegetable and Fruits Prices in India.csv",
        Path("Vegetable and Fruits Prices in India.csv"),
    ]
    for p in paths:
        if p.exists():
            return p
    return None

def get_model_path():
    paths = [ROOT / "models", PARENT / "smart_mandi" / "models", Path("models")]
    for p in paths:
        if (p / "price_model.joblib").exists():
            return p
    return None

def predict_fallback(vegetable: str, month: int, year: int) -> float:
    """Fallback prediction when model not trained - uses simple seasonality."""
    base_prices = {"Potato": 28, "Onion": 32, "Brinjal": 35}
    base = base_prices.get(vegetable, 30)
    season_factor = {1: 1.1, 2: 1.05, 3: 0.95, 4: 0.9, 5: 0.95, 6: 1.0, 7: 1.15, 8: 1.2, 9: 1.0, 10: 0.95, 11: 1.0, 12: 1.1}
    return round(base * season_factor.get(month, 1.0), 2)

# ============ MAIN APP ============
def main():
    st.markdown('<h1 class="hero-title">🌾 Smart Mandi</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">24-Hour Price Forecast • Potato • Onion • Brinjal • Better decisions for Indian farmers</p>',
        unsafe_allow_html=True,
    )

    csv_path = get_csv_path()
    model_path = get_model_path()
    has_model = model_path is not None
    has_data = csv_path is not None

    # Load data if available
    df_agg = None
    if has_data:
        try:
            from smart_mandi.data_loader import prepare_for_modeling
            df_agg = prepare_for_modeling(csv_path, min_samples_per_vegetable=200)
        except Exception as e:
            st.warning(f"Could not load data: {e}")
            has_data = False

    # Sidebar - Prediction inputs
    with st.sidebar:
        st.markdown("### 📊 Forecast Settings")
        vegetable = st.selectbox(
            "Select Vegetable",
            options=["Potato", "Onion", "Brinjal"],
            index=0,
        )
        col1, col2 = st.columns(2)
        with col1:
            month = st.selectbox("Month", range(1, 13), index=2, format_func=lambda x: datetime(2000, x, 1).strftime("%b"))
        with col2:
            year = st.number_input("Year", min_value=2020, max_value=2030, value=2024)
        
        st.markdown("---")
        st.markdown("### 📈 Recent Prices (Optional)")
        st.caption("Enter last 7 days' prices (₹/kg) for better accuracy")
        recent = []
        for i in range(7):
            v = st.number_input(f"Day {7-i} ago", min_value=0.0, value=25.0 + i * 0.5, key=f"price_{i}", step=0.5)
            recent.append(v)
        recent_prices = [float(p) for p in recent if p > 0] if any(p > 0 for p in recent) else None

    # Prediction
    if has_model:
        try:
            from smart_mandi.predict import predict_price
            pred = predict_price(vegetable, month, year, recent_prices, model_path)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            pred = predict_fallback(vegetable, month, year)
    else:
        pred = predict_fallback(vegetable, month, year)
        st.info("💡 **Demo mode** — Train the model in the notebook first for ML predictions.")

    # Main layout: Prediction card + 3D viz
    col_pred, col_space = st.columns([1, 1])
    with col_pred:
        st.markdown("""
        <div class="prediction-card">
            <p style="color: #B8D4A8; font-size: 1rem; margin-bottom: 0.5rem;">24-Hour Price Forecast</p>
            <p class="prediction-value">₹ {:.2f}</p>
            <p style="color: #8FA87E; font-size: 0.95rem;">per kg • {}</p>
        </div>
        """.format(pred, vegetable), unsafe_allow_html=True)

    # 3D Animated Visualization
    st.markdown("---")
    st.markdown("### 🌍 3D Price Surface — Season & Time Trends")
    st.caption("Rotate, zoom, and explore price patterns across months and vegetables")

    if df_agg is not None and len(df_agg) > 100:
        df_agg["date"] = pd.to_datetime(df_agg["date"])
        df_agg["month"] = df_agg["date"].dt.month
        df_agg["year"] = df_agg["date"].dt.year
        
        # 3D Surface: Month × Vegetable × Price
        pivot = df_agg.pivot_table(values="price_per_kg", index="month", columns="vegetable", aggfunc="median")
        vegetables = pivot.columns.tolist()
        months = pivot.index.tolist()
        Z = pivot.values

        fig_3d = go.Figure(data=[go.Surface(
            z=Z, x=months, y=vegetables,
            colorscale=[[0, "#1A3D1A"], [0.5, "#2D5A27"], [1, "#E8B923"]],
            contours=dict(z=dict(show=True, usecolormap=True))
        )])
        fig_3d.update_layout(
            title="Price per KG (₹) — Month × Vegetable",
            scene=dict(
                xaxis_title="Month",
                yaxis_title="Vegetable",
                zaxis_title="Price (₹/kg)",
                bgcolor="#0F1F0A",
                xaxis=dict(gridcolor="rgba(45,90,39,0.3)"),
                yaxis=dict(gridcolor="rgba(45,90,39,0.3)"),
                zaxis=dict(gridcolor="rgba(45,90,39,0.3)"),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E8F0E4"),
            margin=dict(l=0, r=0, t=50, b=0),
            height=500,
        )
        st.plotly_chart(fig_3d, use_container_width=True)

        # 3D Scatter: Year × Month × Price for selected vegetable
        veg_df = df_agg[df_agg["vegetable"] == vegetable]
        fig_scatter = go.Figure(data=[go.Scatter3d(
            x=veg_df["year"],
            y=veg_df["month"],
            z=veg_df["price_per_kg"],
            mode="markers",
            marker=dict(size=4, color=veg_df["price_per_kg"], colorscale="Viridis"),
            hovertemplate="Year: %{x}<br>Month: %{y}<br>Price: ₹%{z:.1f}/kg<extra></extra>",
        )])
        fig_scatter.update_layout(
            title=f"{vegetable} — Price History (Year × Month × ₹/kg)",
            scene=dict(
                xaxis_title="Year",
                yaxis_title="Month",
                zaxis_title="Price (₹/kg)",
                bgcolor="#0F1F0A",
                xaxis=dict(gridcolor="rgba(45,90,39,0.3)"),
                yaxis=dict(gridcolor="rgba(45,90,39,0.3)"),
                zaxis=dict(gridcolor="rgba(45,90,39,0.3)"),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E8F0E4"),
            margin=dict(l=0, r=0, t=50, b=0),
            height=450,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        # Sample 3D when no data
        np.random.seed(42)
        months = list(range(1, 13))
        veggies = ["Potato", "Onion", "Brinjal"]
        Z = np.random.uniform(20, 50, (12, 3))
        fig = go.Figure(data=[go.Surface(z=Z, x=months, y=veggies, colorscale="Viridis")])
        fig.update_layout(
            title="Sample Price Surface (load data for real viz)",
            scene=dict(bgcolor="#0F1F0A", xaxis_title="Month", yaxis_title="Vegetable", zaxis_title="Price"),
            paper_bgcolor="rgba(0,0,0,0)",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Footer metrics
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Model Status", "Active" if has_model else "Demo", "")
    with c2:
        st.metric("Data Source", "Loaded" if has_data else "Sample", "")
    with c3:
        st.metric("Forecast Horizon", "24 Hours", "Next day")


if __name__ == "__main__":
    main()
