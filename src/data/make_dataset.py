"""Load and clean the raw e-commerce transactions dataset.

Each step below is exposed separately (not just the `clean()` wrapper) so a notebook can
call them one at a time and show diagnostics (NA counts, dtypes, shape) between steps --
`clean()` is a convenience for callers (e.g. a future scoring script) that just want the
end result.
"""
import zipfile

import numpy as np
import pandas as pd

RAW_DATA_PATH = "../data/raw/Ecommerce.zip"
RAW_CSV_NAME = "Ecommerce.csv"

COLUMN_NAMES = [
    "invoice_no", "stock_code", "description", "quantity",
    "invoice_date", "unit_price", "customer_id", "country",
]

# stock codes that are fees/postage/samples, not real products (identified by inspecting
# the alpha-only stock codes during EDA)
EXCLUDED_STOCK_CODES = [
    "POST", "D", "DOT", "M", "S", "AMAZONFEE", "m",
    "DCGSSBOY", "DCGSSGIRL", "PADS", "B", "CRUK",
]

EXCLUDED_COUNTRIES = ["European Community", "Unspecified"]

# outlier customer identified during EDA (extreme purchase volume distorting RFM stats)
EXCLUDED_CUSTOMER_IDS = [16446]


def load_raw(path=RAW_DATA_PATH):
    """Read the raw CSV straight out of the zip, no need to extract it manually."""
    with zipfile.ZipFile(path) as zf:
        with zf.open(RAW_CSV_NAME) as f:
            df = pd.read_csv(f, encoding="windows-1252")
    return df.drop(columns=["Unnamed: 8"])


def rename_columns(df):
    df = df.copy()
    df.columns = COLUMN_NAMES
    return df


def fill_missing_customer_id(df):
    """Assign synthetic ids (19000+) to invoices with no customer_id, keyed by invoice_no."""
    missing = df.loc[df["customer_id"].isna(), :]

    backup = pd.DataFrame(missing["invoice_no"].drop_duplicates())
    backup["customer_id"] = np.arange(19000, 19000 + len(backup))

    df = pd.merge(df, backup, on="invoice_no", how="left")
    df["customer_id"] = df["customer_id_x"].combine_first(df["customer_id_y"])
    return df.drop(columns=["customer_id_x", "customer_id_y"])


def fix_dtypes(df):
    df = df.copy()
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], format="%d-%b-%y")
    df["customer_id"] = df["customer_id"].astype(int)
    return df


def filter_records(df):
    """Drop invalid/outlier records: free-of-charge items, non-product stock codes,
    unusable country labels and the known outlier customer."""
    # numerical attributes
    df = df.loc[df["unit_price"] >= 0.04, :]

    # categorical attributes
    df = df[~df["stock_code"].isin(EXCLUDED_STOCK_CODES)]
    df = df.drop(columns="description")
    df = df[~df["country"].isin(EXCLUDED_COUNTRIES)]
    df = df[~df["customer_id"].isin(EXCLUDED_CUSTOMER_IDS)]

    return df.reset_index(drop=True)


def clean(df_raw):
    """Convenience wrapper chaining rename -> fill missing id -> fix dtypes -> filter."""
    df = rename_columns(df_raw)
    df = fill_missing_customer_id(df)
    df = fix_dtypes(df)
    df = filter_records(df)
    return df


def split_returns_purchases(df):
    """Split cleaned transactions into returns (quantity < 0) and purchases (quantity >= 0)."""
    returns = df.loc[df["quantity"] < 0, :].copy()
    purchases = df.loc[df["quantity"] >= 0, :].copy()
    return returns, purchases
