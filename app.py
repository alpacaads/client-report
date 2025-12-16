import os
import math
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# -----------------------------
# App config
# -----------------------------
st.set_page_config(
    page_title="Client Reporting Dashboard",
    page_icon="📊",
    layout="wide",
)

# -----------------------------
# Simple auth (optional)
# -----------------------------
# Set either:
# - STREAMLIT_APP_PASSWORD in env vars, OR
# - st.secrets["APP_PASSWORD"]
def _get_app_password() -> Optional[str]:
    if "APP_PASSWORD" in st.secrets:
        return str(st.secrets["APP_PASSWORD"])
    return os.getenv("STREAMLIT_APP_PASSWORD")

def require_password():
    password = _get_app_password()
    if not password:
        return  # auth disabled

    if "auth_ok" not in st.session_state:
        st.session_state["auth_ok"] = False

    if st.session_state["auth_ok"]:
        return

    st.title("🔒 Login")
    entered = st.text_input("Password", type="password")
    if st.button("Sign in"):
        if entered == password:
            st.session_state["auth_ok"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

require_password()

# -----------------------------
# Data model
# -----------------------------
@dataclass
class ShopifyMetrics:
    revenue_total: float
    revenue_by_channel: pd.DataFrame   # columns: channel, revenue
    product_sales: pd.DataFrame        # columns: product, sku, qty, revenue
    new_customers: int
    returning_customers: int
    conversion_rate: float             # 0-1
    sessions: int
    revenue_timeseries: pd.DataFrame   # columns: date, revenue

@dataclass
class AdsMetrics:
    spend: float
    roas: float
    cpa: float

@dataclass
class DashboardData:
    shopify: ShopifyMetrics
    meta: AdsMetrics
    google: AdsMetrics
    blended_roas: float
    blended_cpa: float
    blended_spend: float

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Controls")

# You can replace this with your real client registry later.
DEFAULT_CLIENTS = [
    {"client_id": "nuage", "name": "Nuage Interiors"},
    {"client_id": "jta", "name": "Japanese Tools Australia"},
    {"client_id": "bestabrasives", "name": "Best Abrasives"},
]

client_name_to_id = {c["name"]: c["client_id"] for c in DEFAULT_CLIENTS}
client_name = st.sidebar.selectbox("Client", list(client_name_to_id.keys()))
client_id = client_name_to_id[client_name]

today = dt.date.today()
default_start = today - dt.timedelta(days=30)

date_start, date_end = st.sidebar.date_input(
    "Date range",
    value=(default_start, today),
    max_value=today,
)

if isinstance(date_start, tuple) or isinstance(date_start, list):
    # Defensive: some Streamlit versions can return a tuple depending on usage
    date_start, date_end = date_start

if date_start > date_end:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

# Data mode toggle: makes the app usable immediately.
mock_mode = st.sidebar.toggle("Mock mode (use sample data)", value=True)

st.sidebar.divider()
st.sidebar.caption("Tip: turn off Mock mode once API connectors are added.")

# -----------------------------
# Utilities
# -----------------------------
def daterange(start: dt.date, end: dt.date) -> List[dt.date]:
    days = (end - start).days
    return [start + dt.timedelta(days=i) for i in range(days + 1)]

def safe_div(n: float, d: float) -> float:
    if d == 0:
        return 0.0
    return n / d

def money(x: float) -> str:
    return f"${x:,.2f}"

def pct(x: float) -> str:
    return f"{x*100:.2f}%"

# -----------------------------
# Mock data generators
# -----------------------------
def mock_shopify(client_id: str, start: dt.date, end: dt.date) -> ShopifyMetrics:
    dates = daterange(start, end)
    base = 2500 if client_id == "nuage" else 1800 if client_id == "jta" else 1400

    # Create a simple revenue curve
    rev = []
    for i, d in enumerate(dates):
        wave = 1 + 0.15 * math.sin(i / 3.0)
        weekday_boost = 1.1 if d.weekday() in (4, 5) else 1.0  # Fri/Sat
        rev.append(base * wave * weekday_boost)

    revenue_timeseries = pd.DataFrame({"date": dates, "revenue": rev})

    channels = ["Online Store", "Shop App", "POS", "Amazon", "Draft Orders"]
    channel_weights = [0.72, 0.06, 0.05, 0.12, 0.05]
    total_rev = float(sum(rev))

    revenue_by_channel = pd.DataFrame({
        "channel": channels,
        "revenue": [total_rev * w for w in channel_weights]
    }).sort_values("revenue", ascending=False)

    product_sales = pd.DataFrame([
        {"product": "Hero Product A", "sku": "SKU-A", "qty": 42, "revenue": total_rev * 0.22},
        {"product": "Hero Product B", "sku": "SKU-B", "qty": 35, "revenue": total_rev * 0.18},
        {"product": "Accessory C", "sku": "SKU-C", "qty": 64, "revenue": total_rev * 0.11},
        {"product": "Bundle D", "sku": "SKU-D", "qty": 18, "revenue": total_rev * 0.09},
        {"product": "Other", "sku": "-", "qty": 120, "revenue": total_rev * 0.40},
    ]).sort_values("revenue", ascending=False)

    sessions = int(18000 + (total_rev / 2.5))
    conversion_rate = 0.018 if client_id == "nuage" else 0.015 if client_id == "jta" else 0.013

    new_customers = int(total_rev / 220)  # rough
    returning_customers = int(total_rev / 340)

    return ShopifyMetrics(
        revenue_total=total_rev,
        revenue_by_channel=revenue_by_channel,
        product_sales=product_sales,
        new_customers=new_customers,
        returning_customers=returning_customers,
        conversion_rate=conversion_rate,
        sessions=sessions,
        revenue_timeseries=revenue_timeseries,
    )

def mock_ads(client_id: str, start: dt.date, end: dt.date, platform: str) -> AdsMetrics:
    days = (end - start).days + 1
    if platform == "meta":
        spend = 55 * days if client_id == "nuage" else 45 * days
        roas = 2.7 if client_id == "nuage" else 2.2
        cpa = 38 if client_id == "nuage" else 44
    else:
        spend = 28 * days if client_id == "nuage" else 25 * days
        roas = 3.1 if client_id == "nuage" else 2.6
        cpa = 41 if client_id == "nuage" else 49

    return AdsMetrics(spend=float(spend), roas=float(roas), cpa=float(cpa))

# -----------------------------
# Connectors (real implementations go here)
# -----------------------------
# These are structured so you can plug in real API calls without changing the UI.
# Keep the return types identical.

@st.cache_data(show_spinner=False, ttl=900)
def fetch_shopify_metrics(client_id: str, start: dt.date, end: dt.date, mock: bool) -> ShopifyMetrics:
    if mock:
        return mock_shopify(client_id, start, end)

    # REAL IMPLEMENTATION PLACE:
    # - Authenticate per client (token in secrets/DB)
    # - Query Shopify Admin API / Analytics
    # - Build the exact ShopifyMetrics object

    raise NotImplementedError("Shopify connector not implemented (turn on Mock mode for now).")

@st.cache_data(show_spinner=False, ttl=900)
def fetch_meta_metrics(client_id: str, start: dt.date, end: dt.date, mock: bool) -> AdsMetrics:
    if mock:
        return mock_ads(client_id, start, end, "meta")

    raise NotImplementedError("Meta connector not implemented (turn on Mock mode for now).")

@st.cache_data(show_spinner=False, ttl=900)
def fetch_google_metrics(client_id: str, start: dt.date, end: dt.date, mock: bool) -> AdsMetrics:
    if mock:
        return mock_ads(client_id, start, end, "google")

    raise NotImplementedError("Google Ads connector not implemented (turn on Mock mode for now).")

def compute_blended(shopify: ShopifyMetrics, meta: AdsMetrics, google: AdsMetrics) -> Tuple[float, float, float]:
    blended_spend = meta.spend + google.spend

    # Spec you gave:
    # Blended ROAS = Shopify revenue / blended spend
    blended_roas = safe_div(shopify.revenue_total, blended_spend)

    # Blended CPA = blended spend / new customers
    blended_cpa = safe_div(blended_spend, float(shopify.new_customers))

    return blended_roas, blended_cpa, blended_spend

def get_dashboard_data(client_id: str, start: dt.date, end: dt.date, mock: bool) -> DashboardData:
    shopify = fetch_shopify_metrics(client_id, start, end, mock)
    meta = fetch_meta_metrics(client_id, start, end, mock)
    google = fetch_google_metrics(client_id, start, end, mock)

    blended_roas, blended_cpa, blended_spend = compute_blended(shopify, meta, google)

    return DashboardData(
        shopify=shopify,
        meta=meta,
        google=google,
        blended_roas=blended_roas,
        blended_cpa=blended_cpa,
        blended_spend=blended_spend,
    )

# -----------------------------
# Page header
# -----------------------------
st.title("📊 Client Reporting Dashboard")
st.caption(f"{client_name} • {date_start.isoformat()} → {date_end.isoformat()}")

# -----------------------------
# Load data
# -----------------------------
with st.spinner("Loading data…"):
    try:
        data = get_dashboard_data(client_id, date_start, date_end, mock_mode)
    except NotImplementedError as e:
        st.error(str(e))
        st.stop()

# -----------------------------
# Shopify filters (channel)
# -----------------------------
channels = list(data.shopify.revenue_by_channel["channel"].unique())
selected_channels = st.multiselect(
    "Shopify sales channels (filter revenue breakdown)",
    options=channels,
    default=channels,
)

filtered_rev_by_channel = data.shopify.revenue_by_channel[
    data.shopify.revenue_by_channel["channel"].isin(selected_channels)
].copy()

# Note: In real Shopify data, you'd also filter revenue_total/timeseries by channel.
# For mock, we only filter the breakdown table/chart.

# -----------------------------
# KPI row (Shopify)
# -----------------------------
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Revenue", money(data.shopify.revenue_total))
k2.metric("Sessions", f"{data.shopify.sessions:,}")
k3.metric("Conversion rate", pct(data.shopify.conversion_rate))
k4.metric("New customers", f"{data.shopify.new_customers:,}")
k5.metric("Returning customers", f"{data.shopify.returning_customers:,}")

st.divider()

# -----------------------------
# Sections layout
# -----------------------------
left, right = st.columns((1.35, 1))

with left:
    st.subheader("A) Shopify data")

    c1, c2 = st.columns((1.2, 1))

    with c1:
        st.markdown("**Revenue (trend)**")
        ts = data.shopify.revenue_timeseries.copy()
        ts["date"] = pd.to_datetime(ts["date"])
        st.line_chart(ts.set_index("date")[["revenue"]])

    with c2:
        st.markdown("**Revenue by sales channel**")
        if filtered_rev_by_channel.empty:
            st.info("No channels selected.")
        else:
            st.bar_chart(filtered_rev_by_channel.set_index("channel")[["revenue"]])

    st.markdown("**Top products**")
    st.dataframe(
        data.shopify.product_sales,
        use_container_width=True,
        hide_index=True,
    )

with right:
    st.subheader("B) Meta ads data")
    m1, m2, m3 = st.columns(3)
    m1.metric("Ad spend", money(data.meta.spend))
    m2.metric("ROAS", f"{data.meta.roas:.2f}")
    m3.metric("CPA", money(data.meta.cpa))

    st.subheader("C) Google ads data")
    g1, g2, g3 = st.columns(3)
    g1.metric("Ad spend", money(data.google.spend))
    g2.metric("ROAS", f"{data.google.roas:.2f}")
    g3.metric("CPA", money(data.google.cpa))

    st.subheader("Blended summary")
    b1, b2, b3 = st.columns(3)
    b1.metric("Blended spend", money(data.blended_spend))
    b2.metric("Blended ROAS", f"{data.blended_roas:.2f}")
    b3.metric("Blended CPA", money(data.blended_cpa))

st.divider()

# -----------------------------
# Export section
# -----------------------------
st.subheader("Export")

export_cols = st.columns(3)

with export_cols[0]:
    # Summary table
    summary = pd.DataFrame([{
        "client_id": client_id,
        "client_name": client_name,
        "start_date": date_start.isoformat(),
        "end_date": date_end.isoformat(),
        "shopify_revenue": data.shopify.revenue_total,
        "sessions": data.shopify.sessions,
        "conversion_rate": data.shopify.conversion_rate,
        "new_customers": data.shopify.new_customers,
        "returning_customers": data.shopify.returning_customers,
        "meta_spend": data.meta.spend,
        "meta_roas": data.meta.roas,
        "meta_cpa": data.meta.cpa,
        "google_spend": data.google.spend,
        "google_roas": data.google.roas,
        "google_cpa": data.google.cpa,
        "blended_spend": data.blended_spend,
        "blended_roas": data.blended_roas,
        "blended_cpa": data.blended_cpa,
    }])

    csv_bytes = summary.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download summary CSV",
        data=csv_bytes,
        file_name=f"{client_id}_summary_{date_start}_{date_end}.csv",
        mime="text/csv",
        use_container_width=True,
    )

with export_cols[1]:
    channels_csv = data.shopify.revenue_by_channel.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download channel breakdown CSV",
        data=channels_csv,
        file_name=f"{client_id}_shopify_channels_{date_start}_{date_end}.csv",
        mime="text/csv",
        use_container_width=True,
    )

with export_cols[2]:
    products_csv = data.shopify.product_sales.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download product sales CSV",
        data=products_csv,
        file_name=f"{client_id}_shopify_products_{date_start}_{date_end}.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.caption("Mock mode is on by default. Switch it off once real API connectors are wired in.")
