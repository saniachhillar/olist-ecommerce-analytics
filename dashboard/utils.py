import sqlite3
import pandas as pd
import os
import streamlit as st
from huggingface_hub import hf_hub_download

# ----------------------------------------------------
# Database Path
# ----------------------------------------------------

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH = os.path.join(
    DATA_DIR,
    "olist.db"
)

# ----------------------------------------------------
# Download Database from Hugging Face
# ----------------------------------------------------

@st.cache_resource
def download_database():

    if not os.path.exists(DB_PATH):

        hf_hub_download(
            repo_id="sania-chhillar/olist-ecommerce-data",
            filename="olist.db",
            repo_type="dataset",
            local_dir=DATA_DIR
        )

# ----------------------------------------------------
# Database Connection
# ----------------------------------------------------

@st.cache_resource
def get_connection():

    download_database()

    return sqlite3.connect(
        DB_PATH,
        check_same_thread=False
    )

# ----------------------------------------------------
# Run Query
# ----------------------------------------------------

@st.cache_data
def run_query(query):

    conn = get_connection()

    return pd.read_sql(query, conn)

# ----------------------------------------------------
# Filter Values
# ----------------------------------------------------

@st.cache_data
def get_filter_values():

    years = run_query("""
    SELECT DISTINCT
    strftime('%Y', order_purchase_timestamp) AS Year
    FROM orders
    ORDER BY Year
    """)

    states = run_query("""
    SELECT DISTINCT
    customer_state
    FROM customers
    ORDER BY customer_state
    """)

    payments = run_query("""
    SELECT DISTINCT
    payment_type
    FROM payments
    ORDER BY payment_type
    """)

    return years, states, payments

# ----------------------------------------------------
# Dynamic WHERE Clause
# ----------------------------------------------------

def build_where_clause(
    year="All",
    state="All",
    payment="All"
):

    conditions = []

    if year != "All":

        conditions.append(
            f"strftime('%Y', orders.order_purchase_timestamp) = '{year}'"
        )

    if state != "All":

        conditions.append(
            f"customers.customer_state = '{state}'"
        )

    if payment != "All":

        conditions.append(
            f"payments.payment_type = '{payment}'"
        )

    if len(conditions) == 0:

        return ""

    return "WHERE " + " AND ".join(conditions)

@st.cache_data
def get_monthly_revenue():

    query = """
    SELECT
        DATE(strftime('%Y-%m-01', orders.order_purchase_timestamp)) AS Month,
        SUM(payments.payment_value) AS Revenue
    FROM orders
    JOIN payments
        ON orders.order_id = payments.order_id
    GROUP BY Month
    ORDER BY Month
    """

    return run_query(query)

