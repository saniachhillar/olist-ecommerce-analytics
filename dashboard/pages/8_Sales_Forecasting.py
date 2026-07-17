import streamlit as st
import pandas as pd
import plotly.express as px

from prophet import Prophet

from utils import run_query

st.set_page_config(page_title="Sales Forecasting", layout="wide")

st.title("📈 AI Sales Forecasting")
st.caption("Forecast future daily sales using Facebook Prophet.")

st.divider()

# --------------------------------------------------------
# Load Daily Sales
# --------------------------------------------------------

df = run_query("""
SELECT
DATE(order_purchase_timestamp) AS ds,
SUM(payment_value) AS y
FROM orders
JOIN payments
ON orders.order_id = payments.order_id
GROUP BY DATE(order_purchase_timestamp)
ORDER BY ds
""")

df["ds"] = pd.to_datetime(df["ds"])

# --------------------------------------------------------
# Forecast Horizon
# --------------------------------------------------------

periods = st.slider(
    "Forecast Days",
    30,
    365,
    90
)

# --------------------------------------------------------
# Train Model
# --------------------------------------------------------

model = Prophet()

model.fit(df)

future = model.make_future_dataframe(periods=periods)

forecast = model.predict(future)

# --------------------------------------------------------
# Plot
# --------------------------------------------------------

fig = px.line(
    forecast,
    x="ds",
    y="yhat",
    title="Predicted Sales"
)

fig.add_scatter(
    x=df["ds"],
    y=df["y"],
    mode="lines",
    name="Actual Sales"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# --------------------------------------------------------
# Forecast Table
# --------------------------------------------------------

st.subheader("Forecast")

st.dataframe(
    forecast[
        ["ds", "yhat", "yhat_lower", "yhat_upper"]
    ].tail(periods)
)