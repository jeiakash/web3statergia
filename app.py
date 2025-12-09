# app.py
# -----------------------------------------------------------
# Web3 Analytics Dashboard – Fully Interactive (Plotly, Dark)
# Auto-loads JSON data, no notebooks, no pickles.
# -----------------------------------------------------------

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------
# Streamlit page config
# -----------------------------------------------------------
st.set_page_config(page_title="Web3 Analytics Dashboard", layout="wide")

# -----------------------------------------------------------
# Helper functions
# -----------------------------------------------------------

def load_json_candidates(candidates):
    """Try multiple filenames, return (parsed_json, filename) for the first that exists."""
    for name in candidates:
        if os.path.exists(name):
            with open(name, "r") as f:
                return json.load(f), name
    raise FileNotFoundError(f"None of these files exist: {candidates}")


def normalize(series: pd.Series) -> pd.Series:
    """Min-max normalize a numeric series to [0, 1]. If constant, returns 0.5."""
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    if s.max() == s.min():
        return pd.Series(0.5, index=s.index)
    return (s - s.min()) / (s.max() - s.min())


def safe_qcut(series: pd.Series, q: int, labels) -> pd.Series:
    """Quantile cut with a fallback if not enough unique values."""
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    if s.nunique() < q:
        return pd.cut(
            s.rank(method="first"),
            bins=q,
            labels=labels,
            include_lowest=True,
        ).astype(int)
    return pd.qcut(
        s.rank(method="first"),
        q=q,
        labels=labels,
        duplicates="drop",
    ).astype(int)


def lifecycle_map(score: float) -> str:
    if score >= 13:
        return "Champion"
    if score >= 10:
        return "Loyal"
    if score >= 7:
        return "Potential"
    if score >= 4:
        return "New"
    return "At risk"


def radar_figure(categories, values, title, height):
    """Minimal clean radar chart using Plotly polar."""
    values = list(map(float, values))
    categories = list(categories)
    # close loop
    values_loop = values + [values[0]]
    categories_loop = categories + [categories[0]]

    fig = go.Figure(
        data=go.Scatterpolar(
            r=values_loop,
            theta=categories_loop,
            fill="toself",
            name=title,
            line=dict(width=2),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title=title,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100]),
        ),
        showlegend=False,
        height=height,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


# -----------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------
st.sidebar.title("Dashboard Controls")

chart_height = st.sidebar.slider("Chart height (px)", 220, 600, 300, 10)

st.sidebar.markdown("---")
st.sidebar.write("JSON files must be in the same folder as `app.py`.")

# -----------------------------------------------------------
# Load JSON data (auto-detect filename variants)
# -----------------------------------------------------------

campaign_raw, campaign_file = load_json_candidates(
    ["campaign_definition_data.json", "campaign_defenition_data.json"]
)
tx_raw, tx_file = load_json_candidates(["transaction_data.json"])
user_raw, user_file = load_json_candidates(
    ["userIdentity_data.json", "useridentity_data.json"]
)
session_raw, session_file = load_json_candidates(
    ["simple_session_data.json", "sample_session_data.json"]
)

st.sidebar.markdown("### Loaded files")
st.sidebar.write(f"- Campaign: `{campaign_file}`")
st.sidebar.write(f"- Transactions: `{tx_file}`")
st.sidebar.write(f"- Users: `{user_file}`")
st.sidebar.write(f"- Session: `{session_file}`")

# -----------------------------------------------------------
# Build DataFrames
# -----------------------------------------------------------

campaign_df = pd.json_normalize(campaign_raw, sep="_")

tx_df = pd.json_normalize(tx_raw, sep="_")
tx_df["transaction_time"] = pd.to_datetime(tx_df["transaction_time"], utc=True)
tx_df["value_wei"] = pd.to_numeric(tx_df["value"], errors="coerce").fillna(0)
tx_df["value_eth"] = tx_df["value_wei"] / 1e18

user_df = pd.json_normalize(user_raw, sep="_")
for col in ["first_visit", "last_visit"]:
    if col in user_df.columns:
        user_df[col] = pd.to_datetime(user_df[col], utc=True)

session_df = pd.json_normalize(session_raw, sep="_")
session_df["start_time"] = pd.to_datetime(session_df["start_time"], utc=True)
session_df["end_time"] = pd.to_datetime(session_df["end_time"], utc=True)
session_df["duration"] = pd.to_numeric(session_df["duration"], errors="coerce").fillna(
    0
)

visited_pages_df = pd.json_normalize(session_raw["visited_pages"], sep="_")
visited_pages_df["timestamp"] = pd.to_datetime(
    visited_pages_df["timestamp"], utc=True
)

clicks_df = pd.json_normalize(session_raw["interactions"]["clicks"], sep="_")
scroll_df = pd.json_normalize(session_raw["interactions"]["scrollEvents"], sep="_")
form_df = pd.json_normalize(session_raw["interactions"]["formInteractions"], sep="_")

chron_df = pd.json_normalize(session_raw["interactions"]["chronological"], sep="_")
chron_df["timestamp"] = pd.to_datetime(chron_df["timestamp"], utc=True)

# -----------------------------------------------------------
# Feature engineering: RFM, ICP, Upsell, Brand
# -----------------------------------------------------------

now = pd.Timestamp.now(tz="UTC")
user_df["recency_days"] = (now - user_df["last_visit"]).dt.days

tx_agg = (
    tx_df.groupby("to_address")
    .agg(
        tx_count=("value_eth", "count"),
        tx_value_sum=("value_eth", "sum"),
        last_tx_time=("transaction_time", "max"),
    )
    .reset_index()
)

user_feat = user_df.merge(
    tx_agg,
    left_on="primary_wallet_address",
    right_on="to_address",
    how="left",
)

user_feat["tx_count"] = user_feat["tx_count"].fillna(0)
user_feat["tx_value_sum"] = user_feat["tx_value_sum"].fillna(0)

user_feat["recency_days"] = user_feat["recency_days"].fillna(
    user_feat["recency_days"].max()
)

user_feat["frequency"] = user_feat["tx_count"]
user_feat["monetary"] = user_feat["tx_value_sum"]

if user_feat["recency_days"].nunique() > 1:
    user_feat["R_score"] = pd.qcut(
        user_feat["recency_days"],
        q=5,
        labels=[5, 4, 3, 2, 1],
        duplicates="drop",
    ).astype(int)
else:
    user_feat["R_score"] = 3

user_feat["F_score"] = safe_qcut(user_feat["frequency"], 5, [1, 2, 3, 4, 5])
user_feat["M_score"] = safe_qcut(user_feat["monetary"], 5, [1, 2, 3, 4, 5])

user_feat["RFM_sum"] = (
    user_feat["R_score"] + user_feat["F_score"] + user_feat["M_score"]
)
user_feat["lifecycle_stage"] = user_feat["RFM_sum"].apply(lifecycle_map)

for col in ["wallet_networth", "total_page_views", "total_time_spent", "tx_value_sum"]:
    if col not in user_feat.columns:
        user_feat[col] = 0

icp_components = pd.DataFrame(
    {
        "networth_n": normalize(user_feat["wallet_networth"]),
        "views_n": normalize(user_feat["total_page_views"]),
        "time_n": normalize(user_feat["total_time_spent"]),
        "txval_n": normalize(user_feat["tx_value_sum"]),
    }
)
user_feat["ICP_score"] = icp_components.mean(axis=1) * 100

user_feat["upsell_readiness_score"] = (
    0.6 * normalize(user_feat["RFM_sum"]) + 0.4 * normalize(user_feat["ICP_score"])
) * 100

brand_df = campaign_df.copy()
if not brand_df.empty:
    clicks_n = normalize(brand_df["clicks"])
    brand_df["premium_score"] = (60 + clicks_n * 40).clip(0, 100)
    brand_df["accessible_score"] = (70 - clicks_n * 20).clip(0, 100)
    brand_df["innovative_score"] = (50 + clicks_n * 50).clip(0, 100)
    brand_df["safe_score"] = (65 + np.random.randn(len(brand_df)) * 5).clip(0, 100)
    brand_df["engage_score"] = (50 + clicks_n * 40).clip(0, 100)
else:
    brand_df = pd.DataFrame(
        columns=[
            "name",
            "clicks",
            "premium_score",
            "accessible_score",
            "innovative_score",
            "safe_score",
            "engage_score",
        ]
    )

# -----------------------------------------------------------
# Tabs
# -----------------------------------------------------------

tabs = st.tabs(
    [
        "User Overview",
        "RFM & Lifecycle",
        "ICP Modeling",
        "Upsell Readiness",
        "Brand Positioning",
        "Transactions",
        "Correlation & Advanced",
        "Session & Interactions",
    ]
)

# -----------------------------------------------------------
# Tab 0 – User Overview
# -----------------------------------------------------------
with tabs[0]:
    st.title("👤 User Overview")

    st.write("User identity data:")
    st.dataframe(user_df, use_container_width=True)

    c1, c2, c3 = st.columns(3)

    if "wallet_networth" in user_df.columns:
        with c1:
            fig = px.histogram(
                user_df,
                x="wallet_networth",
                nbins=10,
                marginal="box",
                template="plotly_dark",
                title="Wallet Networth",
            )
            fig.update_layout(height=chart_height)
            st.plotly_chart(fig, use_container_width=True)

    if "total_time_spent" in user_df.columns:
        with c2:
            fig = px.histogram(
                user_df,
                x="total_time_spent",
                nbins=10,
                marginal="box",
                template="plotly_dark",
                title="Total Time Spent",
            )
            fig.update_layout(height=chart_height)
            st.plotly_chart(fig, use_container_width=True)

    if "total_page_views" in user_df.columns:
        with c3:
            fig = px.histogram(
                user_df,
                x="total_page_views",
                nbins=10,
                marginal="box",
                template="plotly_dark",
                title="Total Page Views",
            )
            fig.update_layout(height=chart_height)
            st.plotly_chart(fig, use_container_width=True)

    c4, c5 = st.columns(2)

    if "device_type" in user_df.columns:
        with c4:
            fig = px.histogram(
                user_df,
                x="device_type",
                template="plotly_dark",
                title="Device Types",
            )
            fig.update_layout(height=chart_height)
            st.plotly_chart(fig, use_container_width=True)

    if "browser_name" in user_df.columns:
        with c5:
            fig = px.histogram(
                user_df,
                x="browser_name",
                template="plotly_dark",
                title="Browsers",
            )
            fig.update_layout(height=chart_height)
            st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------
# Tab 1 – RFM & Lifecycle
# -----------------------------------------------------------
with tabs[1]:
    st.title("🔁 RFM Segmentation & Lifecycle Funnel")

    st.write("RFM user table:")
    st.dataframe(
        user_feat[
            [
                "user_id",
                "wallet_networth",
                "frequency",
                "monetary",
                "R_score",
                "F_score",
                "M_score",
                "RFM_sum",
                "lifecycle_stage",
            ]
        ],
        use_container_width=True,
    )

    c1, c2 = st.columns(2)

    with c1:
        if len(user_feat) > 0:
            rfm_pivot = user_feat.pivot_table(
                index="F_score", columns="M_score", values="R_score", aggfunc="mean"
            )
            fig = px.imshow(
                rfm_pivot,
                text_auto=".1f",
                color_continuous_scale="RdBu",
                origin="lower",
                template="plotly_dark",
                title="RFM Heatmap (F×M, avg R)",
            )
            fig.update_layout(height=chart_height)
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        order = ["New", "Potential", "Loyal", "Champion", "At risk"]
        counts = (
            user_feat["lifecycle_stage"]
            .value_counts()
            .reindex(order)
            .fillna(0)
            .astype(int)
            .reset_index()
        )
        counts.columns = ["lifecycle_stage", "count"]
        fig = px.funnel(
            counts,
            y="lifecycle_stage",
            x="count",
            template="plotly_dark",
            title="Lifecycle Funnel",
        )
        fig.update_layout(height=chart_height)
        st.plotly_chart(fig, use_container_width=True)

    if len(user_feat["lifecycle_stage"].unique()) > 1:
        st.subheader("R / F / M Distribution by Lifecycle Stage")
        melt_rfm = user_feat.melt(
            id_vars="lifecycle_stage",
            value_vars=["R_score", "F_score", "M_score"],
            var_name="metric",
            value_name="score",
        )
        fig = px.violin(
            melt_rfm,
            x="metric",
            y="score",
            color="lifecycle_stage",
            box=True,
            points="all",
            template="plotly_dark",
            title="R/F/M by Lifecycle Stage",
        )
        fig.update_layout(height=chart_height + 100)
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------
# Tab 2 – ICP Modeling
# -----------------------------------------------------------
with tabs[2]:
    st.title("🎯 ICP Modeling")

    st.write("User features with ICP score:")
    st.dataframe(
        user_feat[
            [
                "user_id",
                "wallet_networth",
                "total_page_views",
                "total_time_spent",
                "tx_value_sum",
                "RFM_sum",
                "ICP_score",
            ]
        ],
        use_container_width=True,
    )

    c1, c2 = st.columns(2)

    with c1:
        fig = px.histogram(
            user_feat,
            x="ICP_score",
            nbins=10,
            histnorm="probability density",
            template="plotly_dark",
            title="ICP Score Distribution",
        )
        fig.update_layout(height=chart_height)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.scatter(
            user_feat,
            x="monetary",
            y="ICP_score",
            size="total_page_views",
            hover_name="user_id",
            template="plotly_dark",
            title="ICP vs Monetary (Bubble)",
        )
        fig.update_layout(height=chart_height)
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------
# Tab 3 – Upsell Readiness
# -----------------------------------------------------------
with tabs[3]:
    st.title("📈 Upsell Readiness")

    st.write("User upsell readiness table:")
    st.dataframe(
        user_feat[["user_id", "RFM_sum", "ICP_score", "upsell_readiness_score"]],
        use_container_width=True,
    )

    c1, c2 = st.columns(2)

    with c1:
        fig = px.histogram(
            user_feat,
            x="upsell_readiness_score",
            nbins=10,
            histnorm="probability density",
            template="plotly_dark",
            title="Upsell Readiness Distribution",
        )
        fig.update_layout(height=chart_height)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.scatter(
            user_feat,
            x="RFM_sum",
            y="upsell_readiness_score",
            hover_name="user_id",
            template="plotly_dark",
            title="RFM vs Upsell Readiness",
        )
        fig.update_layout(height=chart_height)
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------
# Tab 4 – Brand Positioning (Radar)
# -----------------------------------------------------------
with tabs[4]:
    st.title("⭐ Brand Positioning")

    st.write("Campaign-based brand scores:")
    st.dataframe(
        brand_df[
            [
                "name",
                "clicks",
                "premium_score",
                "accessible_score",
                "innovative_score",
                "safe_score",
                "engage_score",
            ]
        ],
        use_container_width=True,
    )

    dims = [
        "premium_score",
        "accessible_score",
        "innovative_score",
        "safe_score",
        "engage_score",
    ]
    labels = ["Premium", "Accessible", "Innovative", "Safe", "Engaging"]

    if not brand_df.empty:
        cols = st.columns(min(len(brand_df), 3))
        for i, (_, row) in enumerate(brand_df.iterrows()):
            col = cols[i % len(cols)]
            with col:
                title = str(row.get("name", f"Campaign {i+1}"))
                fig = radar_figure(labels, row[dims].values, title, chart_height)
                st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------
# Tab 5 – Transactions
# -----------------------------------------------------------
with tabs[5]:
    st.title("💰 On-chain Transaction Analysis")

    st.write("Raw transactions:")
    st.dataframe(tx_df, use_container_width=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        tx_sorted = tx_df.sort_values("transaction_time")
        fig = px.line(
            tx_sorted,
            x="transaction_time",
            y="value_eth",
            template="plotly_dark",
            title="Transaction Curve (Value over Time)",
        )
        fig.update_layout(height=chart_height)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.histogram(
            tx_df,
            x="function_name",
            template="plotly_dark",
            title="Function Call Distribution",
        )
        fig.update_layout(height=chart_height)
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        fig = px.density_heatmap(
            tx_df,
            x="block_no",
            y="value_eth",
            nbinsx=10,
            nbinsy=10,
            color_continuous_scale="Viridis",
            template="plotly_dark",
            title="Block vs Value Density (Hexbin-like)",
        )
        fig.update_layout(height=chart_height)
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------
# Tab 6 – Correlation & Advanced
# -----------------------------------------------------------
with tabs[6]:
    st.title("📡 Correlation & Advanced Plots")

    numeric_cols = [
        "wallet_networth",
        "total_page_views",
        "total_time_spent",
        "tx_value_sum",
        "frequency",
        "monetary",
        "RFM_sum",
        "ICP_score",
        "upsell_readiness_score",
    ]
    feature_df = user_feat[numeric_cols].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0)

    c1, c2 = st.columns(2)

    with c1:
        if feature_df.shape[1] > 1:
            corr = feature_df.corr()
            fig = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="RdBu",
                origin="lower",
                template="plotly_dark",
                title="Feature Correlation Heatmap",
            )
            fig.update_layout(height=chart_height + 50)
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        subset_cols = [
            col for col in ["wallet_networth", "total_page_views",
                            "total_time_spent", "ICP_score"]
            if col in feature_df.columns
        ]
        if len(subset_cols) >= 2:
            fig.update_layout(height=chart_height + 100)
            st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------
# Tab 7 – Session & Interactions
# -----------------------------------------------------------
with tabs[7]:
    st.title("🧭 Session & Interaction Analytics")

    c1, c2 = st.columns(2)

    with c1:
        fig = px.bar(
            visited_pages_df,
            x="path",
            y="duration",
            template="plotly_dark",
            title="Time Spent per Page",
        )
        fig.update_layout(height=chart_height)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        if "path" in clicks_df.columns:
            clicks_counts = (
                clicks_df["path"].value_counts().reset_index()
            )
            clicks_counts.columns = ["path", "count"]
            fig = px.bar(
                clicks_counts,
                x="path",
                y="count",
                template="plotly_dark",
                title="Clicks per Page",
            )
            fig.update_layout(height=chart_height)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Scroll Depth by Page")
    fig = px.scatter(
        scroll_df,
        x="path",
        y="scrollPercentage",
        color="scrollDirection",
        template="plotly_dark",
        title="Scroll Depth Distribution",
    )
    fig.update_layout(height=chart_height + 50)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Chronological Event Timeline")
    cols = ["timestamp", "category", "interactionType", "path"]
    cols = [c for c in cols if c in chron_df.columns]

    fig = px.scatter(
        chron_df,
        x="timestamp",
        y="category",
        color="interactionType",
        hover_data=["path"],
        template="plotly_dark",
        title="Event Timeline",
    )
    fig.update_layout(height=chart_height + 50)
    st.plotly_chart(fig, use_container_width=True)

    st.write("Event table:")
    st.dataframe(
        chron_df[cols].sort_values("timestamp"),
        use_container_width=True,
    )
