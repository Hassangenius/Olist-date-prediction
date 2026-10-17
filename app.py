import streamlit as st
import pandas as pd
import joblib


PREDICTION_UNIT = "days"

st.set_page_config(
    page_title="Delivery Date Predictor",
    page_icon="📦",
    layout="centered"
)

st.title("📦 Delivery Date Predictor")
st.write("Enter your order details to estimate the delivery date.")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.subheader("Order details")

with st.form("delivery_form"):
    col1, col2 = st.columns(2)

    with col1:
        n_items = st.number_input("Number of items", min_value=1, value=2)
        n_distinct_products = st.number_input("Number of distinct products", min_value=1, value=2)
        total_weight_g = st.number_input("Total weight (g)", min_value=0.0, value=1000.0)
        n_sellers = st.number_input("Number of sellers", min_value=1, value=1)
        n_distinct_categories = st.number_input("Number of distinct categories", min_value=1, value=1)

    with col2:
        max_longest_side_cm = st.number_input("product dimension (cm)", min_value=0.0, value=30.0)
        max_distance_km = st.number_input("distance (km)", min_value=0.0, value=25.0)
        mean_distance_km = st.number_input("Mean distance (km)", min_value=0.0, value=25.0)
        all_same_state = st.selectbox("Are all sellers in the same state?", options=[True, False])
        distance_missing = st.selectbox("Is distance information missing?", options=[False, True])

    purchase_date = st.date_input("Purchase date")
    

    submitted = st.form_submit_button("Predict delivery date")

if submitted:
    purchase_timestamp = pd.Timestamp(purchase_date).replace(hour=12)

    input_data = pd.DataFrame([{
        "n_items": n_items,
        "n_distinct_products": n_distinct_products,
        "total_weight_g": total_weight_g,
        "n_sellers": n_sellers,
        "n_distinct_categories": n_distinct_categories,
        "max_longest_side_cm": max_longest_side_cm,
        "max_distance_km": max_distance_km,
        "mean_distance_km": mean_distance_km,
        "all_same_state": all_same_state,
        "distance_missing": distance_missing,
        "purchase_month": purchase_timestamp.month,
        "purchase_day": purchase_timestamp.day,
        "purchase_hour": purchase_timestamp.hour
    }])

    try:
        days = max(round(float(model.predict(input_data)[0])), 0)
        predicted_date = purchase_date + pd.Timedelta(days=days)

        st.success(f"Estimated delivery date: {predicted_date:%d %B %Y}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")