"""RFM (Recency, Frequency, Monetary) feature engineering, one row per customer.

Each feature is added by its own function so a notebook can merge one feature at a time
and check for new NAs after each merge.
"""
import pandas as pd

RFM_COLUMNS = ["gross_revenue", "recency_days", "qtde_products", "frequency", "qtde_returns"]


def create_reference(df):
    """One row per customer_id, dropping transaction-level columns."""
    return (
        df.drop(columns=["invoice_no", "stock_code", "quantity", "invoice_date", "unit_price", "country"])
        .drop_duplicates(ignore_index=True)
    )


def add_gross_revenue(ref, purchases):
    purchases = purchases.copy()
    purchases["gross_revenue"] = purchases["quantity"] * purchases["unit_price"]
    monetary = purchases.groupby("customer_id")["gross_revenue"].sum().reset_index()
    return pd.merge(ref, monetary, on="customer_id", how="left")


def add_recency(ref, df, purchases):
    recency = purchases.groupby("customer_id")["invoice_date"].max().reset_index()
    recency["recency_days"] = (df["invoice_date"].max() - recency["invoice_date"]).dt.days
    return pd.merge(ref, recency[["customer_id", "recency_days"]], on="customer_id", how="left")


def add_qtde_products(ref, purchases):
    qtde_products = (
        purchases.groupby("customer_id")["stock_code"]
        .count()
        .reset_index()
        .rename(columns={"stock_code": "qtde_products"})
    )
    return pd.merge(ref, qtde_products, on="customer_id", how="left")


def add_frequency(ref, purchases):
    aux = (
        purchases[["customer_id", "invoice_no", "invoice_date"]]
        .drop_duplicates()
        .groupby("customer_id")
        .agg(
            max_=("invoice_date", "max"),
            min_=("invoice_date", "min"),
            days_=("invoice_date", lambda x: (x.max() - x.min()).days + 1),
            buy_=("invoice_no", "count"),
        )
        .reset_index()
    )
    aux["frequency"] = aux.apply(lambda x: x["buy_"] / x["days_"] if x["days_"] != 0 else 0, axis=1)
    return pd.merge(ref, aux[["customer_id", "frequency"]], on="customer_id", how="left")


def add_qtde_returns(ref, returns):
    qtde_returns = (
        returns.groupby("customer_id")["quantity"]
        .sum()
        .reset_index()
        .rename(columns={"quantity": "qtde_returns"})
    )
    qtde_returns["qtde_returns"] *= -1
    ref = pd.merge(ref, qtde_returns, how="left", on="customer_id")
    ref["qtde_returns"] = ref["qtde_returns"].fillna(0)
    return ref
