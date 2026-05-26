import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Supply Chain Delay Predictor",
    page_icon="🚚",
    layout="wide"
)

# Load model and data
@st.cache_resource
def load_model():
    return joblib.load('src/model.pkl')

@st.cache_data
def load_data():
    df = pd.read_csv('data/processed/features.csv')
    return df

model = load_model()
df    = load_data()

# ── Header ──────────────────────────────────────────────
st.title("🚚 Supply Chain Delay Predictor")
st.markdown("*Predicting delivery delays using machine learning on 100k+ real orders*")
st.divider()

# ── KPI Row ─────────────────────────────────────────────
total        = len(df)
delayed      = df['is_delayed'].sum()
delay_rate   = round((delayed / total) * 100, 1)
avg_leadtime = round(df['promised_lead_time'].mean(), 1)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Orders",    f"{total:,}")
col2.metric("Delayed Orders",  f"{delayed:,}")
col3.metric("Delay Rate",      f"{delay_rate}%")
col4.metric("Avg Lead Time",   f"{avg_leadtime} days")

st.divider()

# ── Two column layout ────────────────────────────────────
left, right = st.columns([2, 1])

with left:
    # Monthly delay trend
    st.subheader("Monthly Delay Trend")
    PATH = "data/raw/Brazilian E-Comm/"
    orders = pd.read_csv(PATH + "olist_orders_dataset.csv",
        parse_dates=['order_purchase_timestamp',
                     'order_estimated_delivery_date',
                     'order_delivered_customer_date'])
    orders_clean = orders[orders['order_status'] == 'delivered'].copy()
    orders_clean = orders_clean.dropna(subset=[
        'order_delivered_customer_date',
        'order_estimated_delivery_date'])
    orders_clean['is_delayed'] = (
        orders_clean['order_delivered_customer_date'] >
        orders_clean['order_estimated_delivery_date']
    ).astype(int)
    orders_clean['order_month'] = orders_clean[
        'order_purchase_timestamp'].dt.to_period('M').astype(str)

    monthly = orders_clean.groupby('order_month').agg(
        total=('is_delayed','count'),
        delayed=('is_delayed','sum')
    ).reset_index()
    monthly['delay_rate'] = round(
        (monthly['delayed'] / monthly['total']) * 100, 1)

    fig1 = px.line(monthly, x='order_month', y='delay_rate',
                   markers=True,
                   labels={'order_month':'Month',
                           'delay_rate':'Delay Rate (%)'},
                   color_discrete_sequence=['#e74c3c'])
    fig1.update_layout(height=300, margin=dict(t=10,b=10))
    st.plotly_chart(fig1, use_container_width=True)

    # Feature importance
    st.subheader("What drives delays?")
    feat_df = pd.DataFrame({
        'Feature':    df.drop('is_delayed',axis=1).columns.tolist(),
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)

    fig2 = px.bar(feat_df, x='Importance', y='Feature',
                  orientation='h',
                  color_discrete_sequence=['#e74c3c'])
    fig2.update_layout(height=350, margin=dict(t=10,b=10))
    st.plotly_chart(fig2, use_container_width=True)

with right:
    # Delay rate gauge
    st.subheader("Overall delay rate")
    fig3 = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = delay_rate,
        number= {'suffix': '%'},
        gauge = {
            'axis': {'range': [0, 30]},
            'bar':  {'color': '#e74c3c'},
            'steps': [
                {'range': [0,  10], 'color': '#2ecc71'},
                {'range': [10, 20], 'color': '#f39c12'},
                {'range': [20, 30], 'color': '#e74c3c'},
            ]
        }
    ))
    fig3.update_layout(height=250, margin=dict(t=10,b=10))
    st.plotly_chart(fig3, use_container_width=True)

    # Delay breakdown pie
    st.subheader("On time vs delayed")
    fig4 = px.pie(
        values=[total - delayed, delayed],
        names =['On Time', 'Delayed'],
        color_discrete_sequence=['#2ecc71','#e74c3c']
    )
    fig4.update_layout(height=280, margin=dict(t=10,b=10))
    st.plotly_chart(fig4, use_container_width=True)

st.divider()

# ── Live Predictor ───────────────────────────────────────
st.subheader("🔮 Live Delay Predictor")
st.markdown("Adjust the inputs below to predict whether an order will be delayed.")

c1, c2, c3 = st.columns(3)

with c1:
    promised_lead_time = st.slider("Promised lead time (days)", 1, 60, 14)
    order_month        = st.slider("Order month", 1, 12, 6)
    order_dayofweek    = st.slider("Day of week (0=Mon, 6=Sun)", 0, 6, 2)

with c2:
    order_hour         = st.slider("Order hour", 0, 23, 12)
    is_weekend         = st.selectbox("Weekend order?", [0, 1],
                                      format_func=lambda x: "Yes" if x else "No")
    total_items        = st.slider("Number of items", 1, 20, 1)

with c3:
    total_price        = st.slider("Order value (R$)", 10, 5000, 150)
    total_freight      = st.slider("Freight value (R$)", 5, 500, 20)
    seller_delay_rate  = st.slider("Seller delay rate", 0.0, 1.0, 0.08)
    product_weight_g   = st.slider("Product weight (g)", 50, 30000, 500)

input_data = pd.DataFrame([{
    'promised_lead_time': promised_lead_time,
    'order_month':        order_month,
    'order_dayofweek':    order_dayofweek,
    'order_hour':         order_hour,
    'is_weekend':         is_weekend,
    'total_items':        total_items,
    'total_price':        total_price,
    'total_freight':      total_freight,
    'seller_delay_rate':  seller_delay_rate,
    'product_weight_g':   product_weight_g
}])

proba      = model.predict_proba(input_data)[0][1]
prediction = model.predict(input_data)[0]
pct        = round(proba * 100, 1)

st.divider()
if prediction == 1:
    st.error(f"⚠️ High delay risk — {pct}% probability of delay")
else:
    st.success(f"✅ Low delay risk — {pct}% probability of delay")

st.progress(float(proba))
st.caption(f"Model: XGBoost + SMOTE | ROC-AUC: 0.80 | Delay recall: 69%")