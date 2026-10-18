from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Olist Delivery Time Predictor",layout="centered")

@st.cache_resource
def load_model():
    with open("delivery_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

st.title(" Olist E-Commerce Delivery Time Predictor")

st.divider()

st.sidebar.header(" Order Parameters")

def user_input_features():
    n_items = st.sidebar.number_input("Number of Items", min_value=1, max_value=20, value=1)
    n_distinct_products = st.sidebar.number_input("Distinct Products", min_value=1, max_value=10, value=1)
    n_sellers = st.sidebar.number_input("Number of Sellers", min_value=1, max_value=5, value=1)
    n_distinct_categories = st.sidebar.number_input("Distinct Categories", min_value=1, max_value=3, value=1)
    
    total_weight_g = st.sidebar.number_input("Total Weight (g)", min_value=10.0, max_value=150000.0, value=1000.0)
    max_single_item_weight_g = st.sidebar.number_input("Max Single Item Weight (g)", min_value=10.0, max_value=150000.0, value=1000.0)
    total_volume_cm3 = st.sidebar.number_input("Total Volume (cm³)", min_value=10.0, max_value=500000.0, value=5000.0)
    
    max_distance_km = st.sidebar.slider("Max Distance (km)", min_value=1.0, max_value=4500.0, value=500.0)
    mean_distance_km = st.sidebar.slider("Mean Distance (km)", min_value=1.0, max_value=4500.0, value=500.0)
    
    all_same_state = st.sidebar.selectbox("Are Seller and Customer in the Same State?", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    
    order_hour = st.sidebar.slider("Order Hour (0-23)", 0, 23, 12)
    order_dow = st.sidebar.selectbox("Day of Week", options=[0, 1, 2, 3, 4, 5, 6], format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x])
    order_month = st.sidebar.slider("Order Month (1-12)", 1, 12, 6)

    data = {"n_items": n_items,
        "n_distinct_products": n_distinct_products,
        "n_sellers": n_sellers,
        "n_distinct_categories": n_distinct_categories,
        "total_weight_g": total_weight_g,
        "max_single_item_weight_g": max_single_item_weight_g,
        "total_volume_cm3": total_volume_cm3,
        "max_distance_km": max_distance_km,
        "mean_distance_km": mean_distance_km,
        "all_same_state": all_same_state,
        "order_hour": order_hour,
        "order_dow": order_dow,
        "order_month": order_month}
    
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

st.subheader(" Selected Order Features")
st.dataframe(input_df)

if st.button(" Predict Delivery Time", type="primary"):
    prediction_hours = int(round(model.predict(input_df)[0]))
    prediction_days = int(round(prediction_hours / 24))
    
    estimated_date = datetime.now() + timedelta(hours=prediction_hours)
    formatted_date = estimated_date.strftime("%B %d, %Y")
    
    st.success(f"### Estimated Delivery Time: **{prediction_hours} hours** (~**{prediction_days} days**) \n\n"
        f" **Estimated Delivery Date:** {formatted_date}")
