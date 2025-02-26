# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
import pickle
import os
from tensorflow.keras.models import load_model
import tensorflow as tf
import lightgbm
from prophet import Prophet

# Custom CSS for a sleek, luxurious UI
st.markdown("""
    <style>
    .main {
        background: linear-gradient(120deg, #0f2027, #203a43, #2c5364);
        font-family: 'Lora', serif;
        padding: 40px;
        border-radius: 25px;
        box-shadow: 0 15px 40px rgba(0,0,0,0.5);
        color: #f4f4f4;
        overflow: hidden;
    }
    .title {
        color: #f4f4f4;
        font-size: 50px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 15px;
        letter-spacing: 1.5px;
        background: linear-gradient(to right, #f4f4f4, #d4af37);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        color: #d4af37;
        font-size: 22px;
        text-align: center;
        margin-bottom: 40px;
        font-style: italic;
        opacity: 0.85;
    }
    .stButton>button {
        background: linear-gradient(to right, #d4af37, #b8860b);
        color: #0f2027;
        border: none;
        border-radius: 50px;
        padding: 15px 35px;
        font-size: 18px;
        font-weight: 600;
        box-shadow: 0 8px 15px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
        text-transform: uppercase;
    }
    .stButton>button:hover {
        background: linear-gradient(to right, #b8860b, #d4af37);
        box-shadow: 0 12px 20px rgba(0,0,0,0.4);
        transform: translateY(-2px);
        color: #fff;
    }
    .metric-box {
        background: rgba(244, 244, 244, 0.95);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        text-align: center;
        border: 1px solid #d4af37;
        transition: all 0.3s ease;
        margin: 20px auto;
        max-width: 400px;
    }
    .metric-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 25px rgba(0,0,0,0.3);
    }
    .metric-title {
        color: #0f2027;
        font-size: 24px;
        font-weight: 600;
        margin-bottom: 10px;
    }
    .metric-value {
        color: #d4af37;
        font-size: 36px;
        font-weight: 700;
    }
    .stExpander {
        background: rgba(244, 244, 244, 0.05);
        border-radius: 15px;
        border: 1px solid rgba(212, 175, 55, 0.3);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .stNumberInput input, .stSlider input, .stSelectbox select, .stTextInput input {
        background: rgba(244, 244, 244, 0.1);
        color: #f4f4f4;
        border-radius: 12px;
        border: 1px solid #d4af37;
        padding: 10px;
    }
    .stSlider .st-bx {
        background: #d4af37 !important;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .footer {
        color: #d4af37;
        font-size: 14px;
        text-align: center;
        margin-top: 50px;
        font-weight: 400;
        opacity: 0.7;
    }
    </style>
""", unsafe_allow_html=True)

# Load models silently
model_files = {
    'xgb_model.pkl': 'XGBoost',
    'lstm_model.h5': 'LSTM',
    'prophet_model.pkl': 'Prophet',
    'lgbm_model.pkl': 'LightGBM',
    'le_mach.pkl': 'Machinery Encoder',
    'le_region.pkl': 'Region Encoder'
}

for file, name in model_files.items():
    if not os.path.exists(file):
        st.error(f"Missing {name} file: '{file}'. Ensure it’s in the same directory.")
        st.stop()

try:
    xgb_model = joblib.load('xgb_model.pkl')
    lstm_model = load_model('lstm_model.h5', custom_objects={'mae': tf.keras.losses.MeanAbsoluteError()})
    with open('prophet_model.pkl', 'rb') as f:
        prophet_model = pickle.load(f)
    lgbm_model = joblib.load('lgbm_model.pkl')
    le_mach = joblib.load('le_mach.pkl')
    le_region = joblib.load('le_region.pkl')
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Prediction function for future sales only
def predict_future_sales(customer_id, date, daily_sales, market_share, political, marketing, budget, machinery_type, region):
    machinery_encoded = le_mach.transform([machinery_type])[0] if machinery_type in le_mach.classes_ else -1
    region_encoded = le_region.transform([region])[0] if region in le_region.classes_ else -1
    political_encoded = {"Low": 0, "Medium": 1, "High": 2}.get(political, 0)
    marketing_encoded = {"Low": 0, "Medium": 1, "High": 2}.get(marketing, 0)

    day_of_week = date.weekday()
    month = date.month
    quarter = (date.month - 1) // 3 + 1
    budget_market_share = budget * market_share
    political_marketing = political_encoded * marketing_encoded
    sales_lag_7 = sales_lag_14 = sales_lag_30 = 0

    input_data = np.array([[day_of_week, month, quarter, machinery_encoded, region_encoded,
                            market_share, political_encoded, marketing_encoded, budget,
                            budget_market_share, political_marketing, sales_lag_7,
                            sales_lag_14, sales_lag_30]])

    xgb_pred = xgb_model.predict(input_data)[0]
    input_lstm = np.repeat(input_data, 7, axis=0).reshape(1, 7, 14)
    lstm_pred = lstm_model.predict(input_lstm, verbose=0)[0][0]

    prophet_input = pd.DataFrame({
        'ds': [pd.to_datetime(date)],
        'Budget': [budget],
        'Market_Share': [market_share]
    })
    prophet_pred = prophet_model.predict(prophet_input)['yhat'].values[0]

    ensemble_input = np.array([[xgb_pred, lstm_pred, prophet_pred]])
    final_pred = lgbm_model.predict(ensemble_input)[0]

    return final_pred

# Streamlit App Layout
st.markdown('<div class="title">Future Sales Oracle</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict Tomorrow’s Success Today</div>', unsafe_allow_html=True)

# Input Section
with st.expander("Enter Future Sales Inputs", expanded=True):
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        customer_id = st.number_input("Customer ID", min_value=0, step=1, value=1, help="Your unique customer identifier")
        date = st.date_input("Future Date", value=datetime.today(), help="Select the future date for prediction")
        daily_sales = st.slider("Expected Daily Sales %", 0.0, 100.0, 50.0, step=0.1, help="Projected daily sales percentage")
        market_share = st.slider("Projected Market Share", 0.0, 1.0, 0.5, step=0.01, help="Anticipated market share fraction")
    with col2:
        political = st.selectbox("Political Influence", ["Low", "Medium", "High"], help="Expected political impact")
        marketing = st.selectbox("Marketing Effort", ["Low", "Medium", "High"], help="Planned marketing intensity")
        budget = st.number_input("Budget", min_value=0.0, value=100000.0, step=1000.0, help="Planned budget")
        machinery_type = st.text_input("Machinery Type", value="Backhoe Loader", help="Enter the type of machinery")
        region = st.selectbox("Region", le_region.classes_.tolist(), help="Target region")

# Predict Button
if st.button("Forecast Future Sales"):
    final_pred = predict_future_sales(
        customer_id, date, daily_sales, market_share, political, marketing, budget, machinery_type, region
    )
    st.markdown(f'<div class="metric-value">{final_pred:.2f}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
# Footer
st.markdown("""
    <hr style='border: 1px solid #d4af37; opacity: 0.3;'>
    <div class='footer'>
        Designed with Sophistication by ~ TEAM DATUM DAREDEVILS 
    </div>
""", unsafe_allow_html=True)