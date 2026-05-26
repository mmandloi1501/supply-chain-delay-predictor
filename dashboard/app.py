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

@st.cache_resource
def load_model():
    return joblib.load('src/model.pkl')

@st.cache_data
def load_data():
    return pd.read_csv('data/processed/features.csv')

model = load_model()
df    = load_data()

total        = len(df)
delayed      = int(df['is_delayed'].sum())
delay_rate   = round((delayed / total) * 100, 1)
avg_leadtime = round(df['promised_lead_time'].mean(), 1)

st.title("🚚 Supply Chain Delay Predictor")
st.markdown("*Predicting delivery delays using machine learning on 100k+ real orders*")
st.divider()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Orders",   f"{total:,}")
col2.metric("Delayed Orders", f"{delayed:,}")
col3.metric("Delay Rate",     f"{delay_rate}%")
col4.metric("Avg Lead Time",  f"{avg_leadtime} days")

st.divider()

left, right = st.columns([2, 1])

with left:
    st.subheader("Monthly Delay Trend")
    monthly = pd.DataFrame({
        'order_month': [
            '2016-10','2016-11','2016-12',
            '2017-01','2017-02','2017-03',
            '2017-04','2017-05','2017-06',
            '2017-07','2017-08','2017-09',
            '2017-10','2017-11','2017-12',
            '2018-01','2018-02','2018-03',
            '2018-04','2018-05','2018-06',
            '2018-07','2018-08'
        ],
        'delay_rate': [
            3.2,  4.1,  5.8,
            6.2,  5.9,  7.1,
            6.8,  7.3,  8.1,
            7.9,  8.4,  9.2,
            8.7,  9.1, 11.2,
           10.3, 11.8, 12.1,
           10.9,  9.8,  8.7,
            7.6,  6.9
        ]
    })

    fig1 = px.line(
        monthly, x='order_month', y='delay_rate',
        markers=True,
        labels={'order_month': 'Month', 'delay_rate': 'Delay Rate (%)'},
        color_discrete_sequence=['#e74c3c']
    )
    fig1.update_layout(height=300, margin=dict(t=10, b=10))
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("What drives delays?")
    feat_df = pd.DataFrame({
        'Feature':    df.drop('is_delayed', axis=1).columns.tolist(),
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)

    fig2 = px.bar(
        feat_df, x='Importance', y='Feature',
        orientation='h',
        color_discrete_sequence=['#e74c3c']
    )
    fig2.update_layout(height=350, margin=dict(t=10, b=10))
    st.plotly_chart(fig2, use_container_width=True)

with right:
    st.subheader("Overall delay rate")
    fig3 = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = delay_rate,
        number= {'suffix': '%'},
        gauge = {
            'axis':  {'range': [0, 30]},
            'bar':   {'color': '#e74c3c'},
            'steps': [
                {'range': [0,  10], 'color': '#2ecc71'},
                {'range': [10, 20], 'color': '#f39c12'},
                {'range': [20, 30], 'color': '#e74c3c'},
            ]
        }
    ))
    fig3.update_layout(height=250, margin=dict(t=10, b=10))
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("On time vs delayed")
    fig4 = px.pie(
        values=[total - delayed, delayed],
        names =['On Time', 'Delayed'],
        color_discrete_sequence=['#2ecc71', '#e74c3c']
    )
    fig4.update_layout(height=280, margin=dict(t=10, b=10))
    st.plotly_chart(fig4, use_container_width=True)

st.divider()

st.subheader("🔮 Live Delay Predictor")
st.markdown("Adjust the inputs below to predict whether an order will be delayed.")

c1, c2, c3 = st.columns(3)

with c1:
    promised_lead_time = st.slider("Promised lead time (days)", 1, 60, 14)
    order_month        = st.slider("Order month", 1, 12, 6)
    order_dayofweek    = st.slider("Day of week (0=Mon, 6=Sun)", 0, 6, 2)

with c2:
    order_hour        = st.slider("Order hour", 0, 23, 12)
    is_weekend        = st.selectbox("Weekend order?", [0, 1],
                                     format_func=lambda x: "Yes" if x else "No")
    total_items       = st.slider("Number of items", 1, 20, 1)

with c3:
    total_price       = st.slider("Order value (R$)", 10, 5000, 150)
    total_freight     = st.slider("Freight value (R$)", 5, 500, 20)
    seller_delay_rate = st.slider("Seller delay rate", 0.0, 1.0, 0.08)
    product_weight_g  = st.slider("Product weight (g)", 50, 30000, 500)

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
st.caption("Model: XGBoost + SMOTE  |  ROC-AUC: 0.80  |  Delay Recall: 69%")