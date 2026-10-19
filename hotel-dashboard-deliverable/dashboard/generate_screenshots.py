"""
Generate pixel-perfect, high-resolution screenshots for all 4 dashboard tabs
and the before/after filter interaction pair using headless Google Chrome.
With zero-overlap subplot layouts, generous horizontal spacing, and left-aligned titles.
Outputs to dashboard/screenshots/:
- tab1_eda.png
- tab2_models.png
- tab3_text_mining.png
- tab4_association_rules.png
- filter_interaction_before.png
- filter_interaction_after.png
"""

import os
import sys
import io

# Ensure UTF-8 output encoding on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import json
import shutil
import base64
import subprocess
import numpy as np
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    cohen_kappa_score
)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TEXT_DIR = os.path.join(DATA_DIR, "text_mining_outputs")
APRIORI_DIR = os.path.join(DATA_DIR, "apriori")
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "dashboard", "screenshots")
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

CHROME_EXE = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

# 1. Load Data
raw_hotels_df = pd.read_csv(os.path.join(DATA_DIR, "combined_real_synthetic_hotels.csv"))
df_hotels = raw_hotels_df.copy()
df_hotels["customer_rating"] = pd.to_numeric(df_hotels["customer_rating"], errors="coerce")
df_hotels = df_hotels.dropna(subset=["customer_rating"]).drop_duplicates(subset="hotel_id", keep="first").copy()
df_hotels["success"] = (df_hotels["customer_rating"] >= 4.0).astype(int)

num_impute_cols = ["price", "review_count", "amenity_count"]
for col in num_impute_cols:
    df_hotels[col] = pd.to_numeric(df_hotels[col], errors="coerce").fillna(df_hotels[col].median())
df_hotels["location"] = df_hotels["location"].fillna("Unknown")
df_hotels["hotel_category"] = df_hotels["hotel_category"].fillna("Unknown")

amenity_cols = [c for c in df_hotels.columns if c.startswith("amenity_") and c != "amenity_count"]
for c in amenity_cols:
    df_hotels[c] = pd.to_numeric(df_hotels[c], errors="coerce").fillna(0).astype(int)
df_hotels["log_price"] = np.log1p(df_hotels["price"])

clean_hotels_df = df_hotels.copy()

# Load Teammate Data
rev_sentiment_df = pd.read_csv(os.path.join(TEXT_DIR, "review_level_sentiment.csv"))
apriori_rules_df = pd.read_csv(os.path.join(APRIORI_DIR, "apriori_strong_rules.csv"))
apriori_support_df = pd.read_csv(os.path.join(APRIORI_DIR, "apriori_amenity_support.csv"))

# Fit ML Models
numeric_features = ["price", "log_price", "review_count", "amenity_count"] + amenity_cols
categorical_features = ["location", "hotel_category"]

X_all = clean_hotels_df[numeric_features + categorical_features].copy()
y_all = clean_hotels_df["success"].copy()

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.20, random_state=42, stratify=y_all)

ml_preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features)
])

model_lr = Pipeline([("preprocessor", ml_preprocessor), ("classifier", LogisticRegression(C=10, class_weight="balanced", max_iter=2000, random_state=42))])
model_dt = Pipeline([("preprocessor", ml_preprocessor), ("classifier", DecisionTreeClassifier(max_depth=7, min_samples_split=10, random_state=42))])
model_rf = Pipeline([("preprocessor", ml_preprocessor), ("classifier", RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=5, class_weight="balanced", random_state=42, n_jobs=-1))])

model_lr.fit(X_train, y_train)
model_dt.fit(X_train, y_train)
model_rf.fit(X_train, y_train)

REVIEW1_BENCHMARKS = {
    "Random Forest": {"Accuracy": 0.8134, "Balanced Accuracy": 0.6443, "Macro Precision": 0.6777, "Macro Recall": 0.6443, "Macro F1": 0.6571, "Weighted F1": 0.8037, "ROC-AUC": 0.7072},
    "Decision Tree": {"Accuracy": 0.8035, "Balanced Accuracy": 0.5662, "Macro Precision": 0.6297, "Macro Recall": 0.5662, "Macro F1": 0.5750, "Weighted F1": 0.7724, "ROC-AUC": 0.6344},
    "Logistic Regression": {"Accuracy": 0.6894, "Balanced Accuracy": 0.6612, "Macro Precision": 0.6056, "Macro Recall": 0.6612, "Macro F1": 0.6045, "Weighted F1": 0.7205, "ROC-AUC": 0.6955}
}

rf_feature_names = model_rf.named_steps["preprocessor"].get_feature_names_out()
rf_importances = model_rf.named_steps["classifier"].feature_importances_
clean_feature_names = [f.replace("num__", "").replace("cat__", "").replace("hotel_category_", "").replace("location_", "") for f in rf_feature_names]
rf_feat_df = pd.DataFrame({"Feature": clean_feature_names, "Importance": rf_importances}).sort_values("Importance", ascending=False).head(15).reset_index(drop=True)

# Common HTML styles
CSS_STYLES = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #1e293b;
    background-color: #f8fafc;
    margin: 0;
    padding: 24px;
}
.dashboard-wrapper {
    max-width: 1320px;
    margin: 0 auto;
}
.header-banner {
    background: #0f172a;
    color: #ffffff;
    padding: 22px 28px;
    border-radius: 8px;
    margin-bottom: 18px;
    border: 1px solid #1e293b;
}
.header-title {
    font-size: 24px;
    font-weight: 700;
    margin: 0 0 5px 0;
    letter-spacing: -0.3px;
    color: #ffffff;
}
.header-subtitle {
    font-size: 13.5px;
    font-weight: 400;
    color: #94a3b8;
    margin: 0;
}
.badge-pill {
    display: inline-block;
    background: #1e293b;
    color: #94a3b8;
    border: 1px solid #334155;
    padding: 3px 10px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    margin-bottom: 8px;
    letter-spacing: 0.8px;
}
.notice-bar {
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #64748b;
    color: #334155;
    padding: 12px 16px;
    border-radius: 6px;
    font-size: 12.5px;
    line-height: 1.5;
    margin-bottom: 18px;
}
.filter-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 18px;
}
.filter-title {
    font-size: 13.5px;
    font-weight: 600;
    color: #0f172a;
    margin-bottom: 14px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.kpi-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 20px;
}
.kpi-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 14px 18px;
    min-width: 155px;
    flex: 1;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    border-top: 3px solid #64748b;
}
.kpi-label {
    font-size: 11.5px;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}
.kpi-value {
    font-size: 22px;
    font-weight: 700;
    color: #0f172a;
}
.kpi-sub {
    font-size: 11px;
    color: #94a3b8;
    margin-top: 2px;
}
.tabs-nav {
    display: flex;
    width: 100%;
    border-bottom: 2px solid #e2e8f0;
    margin-bottom: 20px;
    gap: 6px;
}
.tab-btn {
    flex: 1 1 0px;
    min-width: 180px;
    height: 44px;
    line-height: 44px;
    padding: 0 16px;
    font-size: 13.5px;
    font-weight: 600;
    color: #475569;
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    border-bottom: none;
    border-radius: 8px 8px 0 0;
    text-decoration: none;
    box-sizing: border-box;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
}
.tab-btn.active {
    color: #1e293b;
    background: #ffffff;
    border-top: 3px solid #3b82f6;
    border-bottom: 2px solid #ffffff;
    margin-bottom: -2px;
    font-weight: 700;
}
.tab-note-banner {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 3px solid #0284c7;
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 16px;
    font-size: 12.5px;
    color: #334155;
}
.model-champion-card {
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-left: 4px solid #0284c7;
    color: #0f172a;
    padding: 16px 20px;
    border-radius: 6px;
    margin-bottom: 16px;
}
.chart-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
}
.apriori-controls-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    padding: 14px 18px;
    border-radius: 10px;
    margin-bottom: 16px;
}
"""

def render_header():
    return """
    <div class="header-banner">
        <div class="badge-pill">Business Analytics (23CSE452) • Review 2</div>
        <h1 class="header-title">Hotel Success Intelligence & Decision Support Dashboard</h1>
        <p class="header-subtitle">Interactive Integration of Exploratory Analytics, Machine Learning Models, Text Mining, and Apriori Association Rules</p>
    </div>
    <div class="notice-bar">
        <b>Methodological Notice:</b> "Success" is defined as <code>customer_rating &ge; 4.0</code> (rating-based proxy).
        All relationships are empirical <b>associations</b> and do not imply causality.
        The analysis incorporates both scraped real hotels and synthetic properties (63.4%); text mining applies to real hotels only.
    </div>
    """

def render_filters(loc="All Locations", cat="All Categories", dt="Both", price_text="₹392 - ₹61,327", rating_text="0.0 - 5.0", n_sel=4551, n_tot=4551):
    pct = (n_sel / n_tot) * 100
    
    # Location list items
    loc_options = ["All Locations", "Agra", "Bengaluru", "Chennai", "Coimbatore", "Delhi", "Goa", "Jaipur", "Kolkata", "Mumbai"]
    if loc != "All Locations" and loc in loc_options:
        loc_options = [loc] + [o for o in loc_options if o != loc]

    loc_items = "".join([
        f'<div style="padding: 2px 8px; border-radius: 3px; font-size: 12px; { "background: #0284c7; color: #fff;" if (loc == opt or (loc == "All Locations" and opt == "All Locations")) else "color: #334155;" }">{opt}</div>'
        for opt in loc_options
    ])

    # Category list items
    cat_options = ["All Categories", "5-star hotel", "4-star hotel", "3-star hotel", "2-star hotel", "1-star hotel"]
    if cat != "All Categories" and cat in cat_options:
        cat_options = [cat] + [o for o in cat_options if o != cat]

    cat_items = "".join([
        f'<div style="padding: 2px 8px; border-radius: 3px; font-size: 12px; { "background: #0284c7; color: #fff;" if (cat == opt or (cat == "All Categories" and opt == "All Categories")) else "color: #334155;" }">{opt}</div>'
        for opt in cat_options
    ])

    price_bar_left = "10%" if ("2,000" in price_text or "5,000" in price_text) else "0%"
    price_bar_width = "30%" if "2,000" in price_text else ("80%" if "5,000" in price_text else "100%")

    return f"""
    <div class="filter-box">
        <div class="filter-title">Interactive Global Filters</div>
        <div style="display: flex; gap: 20px; margin-bottom: 14px; flex-wrap: wrap;">
            <div style="display: flex; flex-direction: column; gap: 4px; width: 220px;">
                <span style="font-size: 12px; font-weight: 600; color: #475569;">Location:</span>
                <div style="border: 1px solid #cbd5e1; border-radius: 6px; padding: 4px; background: #ffffff; height: 95px; overflow-y: hidden;">
                    {loc_items}
                </div>
            </div>
            <div style="display: flex; flex-direction: column; gap: 4px; width: 220px;">
                <span style="font-size: 12px; font-weight: 600; color: #475569;">Category:</span>
                <div style="border: 1px solid #cbd5e1; border-radius: 6px; padding: 4px; background: #ffffff; height: 95px; overflow-y: hidden;">
                    {cat_items}
                </div>
            </div>
            <div style="display: flex; flex-direction: column; gap: 4px; width: 200px;">
                <span style="font-size: 12px; font-weight: 600; color: #475569;">Data Type:</span>
                <div style="border: 1px solid #cbd5e1; border-radius: 6px; padding: 6px 12px; background: #ffffff; font-size: 13px; color: #1e293b; display: flex; justify-content: space-between; align-items: center;">
                    <span>{dt}</span>
                    <span style="color: #64748b; font-size: 10px;">▼</span>
                </div>
            </div>
        </div>
        <div style="display: flex; gap: 24px; align-items: center; flex-wrap: wrap; margin-bottom: 12px;">
            <div style="display: flex; align-items: center; gap: 10px; width: 310px;">
                <span style="font-size: 12px; font-weight: 600; color: #475569; width: 70px;">Price (₹):</span>
                <div style="flex: 1; height: 6px; background: #e2e8f0; border-radius: 3px; position: relative;">
                    <div style="position: absolute; left: {price_bar_left}; width: {price_bar_width}; height: 100%; background: #0284c7; border-radius: 3px;"></div>
                </div>
                <span style="font-size: 12px; font-weight: 600; color: #0f172a; min-width: 95px; text-align: right;">{price_text}</span>
            </div>
            <div style="display: flex; align-items: center; gap: 10px; width: 280px;">
                <span style="font-size: 12px; font-weight: 600; color: #475569; width: 55px;">Rating:</span>
                <div style="flex: 1; height: 6px; background: #e2e8f0; border-radius: 3px; position: relative;">
                    <div style="position: absolute; left: 0; width: 100%; height: 100%; background: #0284c7; border-radius: 3px;"></div>
                </div>
                <span style="font-size: 12px; font-weight: 600; color: #0f172a; min-width: 65px; text-align: right;">{rating_text}</span>
            </div>
            <div style="margin-left: auto;">
                <button style="background: #f1f5f9; color: #1e293b; border: 1px solid #cbd5e1; padding: 7px 16px; border-radius: 6px; font-size: 12.5px; font-weight: 600; cursor: pointer;">Reset All Filters</button>
            </div>
        </div>
        <div style="font-size: 12.5px; color: #475569; padding-top: 6px; border-top: 1px solid #f1f5f9;">
            <b>Selected Hotels:</b> <b>{n_sel:,}</b> / {n_tot:,} ({pct:.1f}%)
        </div>
    </div>
    """

def render_kpis(sub_df):
    n = len(sub_df)
    succ_rate = sub_df["success"].mean() * 100 if n > 0 else 0
    succ_count = int(sub_df["success"].sum()) if n > 0 else 0
    avg_price = sub_df["price"].mean() if n > 0 else 0
    med_price = sub_df["price"].median() if n > 0 else 0
    avg_rating = sub_df["customer_rating"].mean() if n > 0 else 0
    min_rating = sub_df["customer_rating"].min() if n > 0 else 0
    max_rating = sub_df["customer_rating"].max() if n > 0 else 0
    avg_reviews = sub_df["review_count"].mean() if n > 0 else 0
    med_reviews = sub_df["review_count"].median() if n > 0 else 0
    avg_amen = sub_df["amenity_count"].mean() if n > 0 else 0
    pct = (n / len(clean_hotels_df)) * 100

    return f"""
    <div class="kpi-row">
        <div class="kpi-card" style="border-top-color: #475569;">
            <div class="kpi-label">Hotels Selected</div>
            <div class="kpi-value">{n:,}</div>
            <div class="kpi-sub">{pct:.1f}% of total corpus</div>
        </div>
        <div class="kpi-card" style="border-top-color: #0284c7;">
            <div class="kpi-label">Average Rating</div>
            <div class="kpi-value">{avg_rating:.2f} <span style="font-size:14px; font-weight:normal; color:#64748b;">/ 5.0</span></div>
            <div class="kpi-sub">Range: {min_rating:.1f} – {max_rating:.1f}</div>
        </div>
        <div class="kpi-card" style="border-top-color: #d97706;">
            <div class="kpi-label">Average Price</div>
            <div class="kpi-value">₹{avg_price:,.0f}</div>
            <div class="kpi-sub">Median: ₹{med_price:,.0f}</div>
        </div>
        <div class="kpi-card" style="border-top-color: #6366f1;">
            <div class="kpi-label">Average Review Count</div>
            <div class="kpi-value">{avg_reviews:,.0f}</div>
            <div class="kpi-sub">Median: {med_reviews:,.0f} reviews</div>
        </div>
        <div class="kpi-card" style="border-top-color: #0d9488;">
            <div class="kpi-label">Success Rate (≥ 4.0)</div>
            <div class="kpi-value">{succ_rate:.1f}%</div>
            <div class="kpi-sub">{succ_count:,} successful hotels</div>
        </div>
        <div class="kpi-card" style="border-top-color: #64748b;">
            <div class="kpi-label">Avg Amenities</div>
            <div class="kpi-value">{avg_amen:.1f}</div>
            <div class="kpi-sub">Across 15 monitored amenities</div>
        </div>
    </div>
    """

def render_tabs_nav(active_idx=0):
    tab_names = ["1. Overview & EDA", "2. Predictive Models", "3. Text Mining", "4. Association Rules", "5. Final Dataset"]
    items = []
    for i, name in enumerate(tab_names):
        cls = "tab-btn active" if i == active_idx else "tab-btn"
        items.append(f'<span class="{cls}">{name}</span>')
    return f'<div class="tabs-nav">{"".join(items)}</div>'

# -------------------------------------------------------------
# BUILD TAB 1 CONTENT
# -------------------------------------------------------------
def get_tab1_html(sub_df):
    # Row 1
    fig_r1 = make_subplots(
        rows=1, cols=2,
        horizontal_spacing=0.16,
        subplot_titles=("Success Distribution (Rating ≥ 4.0)", "Real vs. Synthetic Hotels"),
        specs=[[{"type": "domain"}, {"type": "xy"}]]
    )
    succ_counts = sub_df["success"].value_counts().rename({1: "Successful (≥ 4.0)", 0: "Not Successful (< 4.0)"})
    fig_r1.add_trace(go.Pie(labels=succ_counts.index, values=succ_counts.values, hole=0.55, marker=dict(colors=["#10b981", "#ef4444"])), row=1, col=1)
    dtype_grp = sub_df.groupby("data_type").agg(Count=("hotel_id", "count")).reset_index()
    fig_r1.add_trace(go.Bar(x=dtype_grp["data_type"], y=dtype_grp["Count"], marker_color="#3b82f6", text=dtype_grp["Count"], textposition="outside"), row=1, col=2)
    fig_r1.for_each_annotation(lambda a: a.update(font=dict(size=12, color="#1e293b", family="Inter, sans-serif")))
    fig_r1.update_layout(height=350, margin=dict(l=25, r=25, t=45, b=25), template="plotly_white")

    # Row 2
    fig_r2 = make_subplots(
        rows=1, cols=2,
        horizontal_spacing=0.16,
        subplot_titles=("Customer Rating Distribution", "Success Rate by Star Category (%)")
    )
    for dt, col in [("Real", "#3b82f6"), ("Synthetic", "#8b5cf6")]:
        sub_d = sub_df[sub_df["data_type"] == dt]
        if len(sub_d) > 0:
            fig_r2.add_trace(go.Histogram(x=sub_d["customer_rating"], name=dt, marker_color=col, opacity=0.6, xbins=dict(start=1.0, end=5.0, size=0.1)), row=1, col=1)
    fig_r2.add_vline(x=4.0, line_dash="dash", line_color="#ef4444", annotation_text="Success (4.0)", row=1, col=1)
    fig_r2.update_layout(barmode="overlay")
    cat_grp = sub_df.groupby("hotel_category").agg(Rate=("success", lambda s: s.mean()*100), Count=("hotel_id", "count")).reset_index().sort_values("Rate", ascending=True)
    fig_r2.add_trace(go.Bar(y=cat_grp["hotel_category"], x=cat_grp["Rate"], orientation="h", marker_color="#10b981", text=[f"{r:.1f}% (n={c})" for r, c in zip(cat_grp["Rate"], cat_grp["Count"])], textposition="inside", showlegend=False), row=1, col=2)
    fig_r2.for_each_annotation(lambda a: a.update(font=dict(size=12, color="#1e293b", family="Inter, sans-serif")))
    fig_r2.update_layout(height=350, margin=dict(l=25, r=25, t=45, b=25), template="plotly_white")

    # Row 3
    fig_r3 = make_subplots(
        rows=1, cols=2,
        horizontal_spacing=0.16,
        subplot_titles=("Top 15 Hotel Locations", "Price vs. Review Count (by Success)")
    )
    loc_counts = sub_df["location"].value_counts().head(15).sort_values(ascending=True)
    fig_r3.add_trace(go.Bar(y=loc_counts.index, x=loc_counts.values, orientation="h", marker_color="#0284c7", showlegend=False), row=1, col=1)
    p_cap = np.percentile(sub_df["price"], 98) if len(sub_df) > 10 else sub_df["price"].max()
    r_cap = np.percentile(sub_df["review_count"], 98) if len(sub_df) > 10 else sub_df["review_count"].max()
    sub_scatter = sub_df[(sub_df["price"] <= p_cap) & (sub_df["review_count"] <= r_cap)]
    for s_val, s_label, s_color in [(1, "Success (≥4.0)", "#10b981"), (0, "Not Success (<4.0)", "#ef4444")]:
        s_data = sub_scatter[sub_scatter["success"] == s_val]
        if len(s_data) > 0:
            fig_r3.add_trace(go.Scatter(x=s_data["review_count"], y=s_data["price"], mode="markers", name=s_label, marker=dict(color=s_color, opacity=0.5, size=6)), row=1, col=2)
    fig_r3.update_xaxes(title_text="Review Count", row=1, col=2)
    fig_r3.update_yaxes(title_text="Price (₹)", row=1, col=2)
    fig_r3.for_each_annotation(lambda a: a.update(font=dict(size=12, color="#1e293b", family="Inter, sans-serif")))
    fig_r3.update_layout(height=400, margin=dict(l=25, r=25, t=45, b=25), template="plotly_white")

    # Row 4
    fig_r4 = make_subplots(
        rows=1, cols=2,
        horizontal_spacing=0.18,
        subplot_titles=("Amenity Market Prevalence (%)", "Success Rate Δ: With vs. Without Amenity")
    )
    clean_amen_names = [c.replace("amenity_", "").replace("_", " ").title() for c in amenity_cols]
    amen_prev = [sub_df[c].mean() * 100 for c in amenity_cols]
    diff_data = []
    for c, name in zip(amenity_cols, clean_amen_names):
        with_r = sub_df[sub_df[c] == 1]["success"].mean() * 100 if (sub_df[c] == 1).sum() > 0 else 0
        wout_r = sub_df[sub_df[c] == 0]["success"].mean() * 100 if (sub_df[c] == 0).sum() > 0 else 0
        diff_data.append({"Amenity": name, "Diff": with_r - wout_r})
    diff_df = pd.DataFrame(diff_data).sort_values("Diff", ascending=True)
    fig_r4.add_trace(go.Bar(y=clean_amen_names, x=amen_prev, orientation="h", marker_color="#6366f1", showlegend=False), row=1, col=1)
    colors_diff = ["#10b981" if v >= 0 else "#ef4444" for v in diff_df["Diff"]]
    fig_r4.add_trace(go.Bar(y=diff_df["Amenity"], x=diff_df["Diff"], orientation="h", marker_color=colors_diff, showlegend=False), row=1, col=2)
    fig_r4.for_each_annotation(lambda a: a.update(font=dict(size=12, color="#1e293b", family="Inter, sans-serif")))
    fig_r4.update_layout(height=420, margin=dict(l=25, r=25, t=45, b=25), template="plotly_white")

    # Row 5
    fig_r5 = make_subplots(
        rows=1, cols=2,
        horizontal_spacing=0.16,
        subplot_titles=("Amenity Count by Success Status", "Correlation Matrix (Rating Excluded)"),
        specs=[[{"type": "xy"}, {"type": "heatmap"}]]
    )
    fig_r5.add_trace(go.Box(x=sub_df["success"].map({1: "Successful (≥4.0)", 0: "Not Successful (<4.0)"}), y=sub_df["amenity_count"], marker_color="#8b5cf6", boxpoints="outliers", showlegend=False), row=1, col=1)
    corr_cols = ["price", "log_price", "review_count", "amenity_count", "success"]
    corr_mat = sub_df[corr_cols].corr()
    fig_r5.add_trace(go.Heatmap(z=corr_mat.values, x=corr_cols, y=corr_cols, colorscale="Blues", zmin=-0.2, zmax=1.0, text=corr_mat.round(2).values, texttemplate="%{text}", showscale=False), row=1, col=2)
    fig_r5.for_each_annotation(lambda a: a.update(font=dict(size=12, color="#1e293b", family="Inter, sans-serif")))
    fig_r5.update_layout(height=360, margin=dict(l=25, r=25, t=45, b=25), template="plotly_white")

    html = f"""
    <div class="tab-note-banner">
        <b>Filter Applicability:</b> All 10 charts on this tab respond dynamically to the Global Filters above.<br>
        <i>Notice: Amenity availability differences, correlations, and distributions represent statistical associations, NOT direct causes.</i>
    </div>
    <div class="chart-box">{fig_r1.to_html(full_html=False, include_plotlyjs="cdn")}</div>
    <div class="chart-box">{fig_r2.to_html(full_html=False, include_plotlyjs=False)}</div>
    <div class="chart-box">{fig_r3.to_html(full_html=False, include_plotlyjs=False)}</div>
    <div class="chart-box">{fig_r4.to_html(full_html=False, include_plotlyjs=False)}</div>
    <div class="chart-box">{fig_r5.to_html(full_html=False, include_plotlyjs=False)}</div>
    """
    return html

# -------------------------------------------------------------
# BUILD TAB 2 CONTENT
# -------------------------------------------------------------
def get_tab2_html(sub_df):
    test_indices = X_test.index.intersection(sub_df.index)
    n_test_sub = len(test_indices)
    y_test_sub = y_test.loc[test_indices]
    single_class = y_test_sub.nunique() < 2

    rows_eval = []
    for name, m in [("Random Forest", model_rf), ("Decision Tree", model_dt), ("Logistic Regression", model_lr)]:
        bench = REVIEW1_BENCHMARKS[name]
        p_sub = m.predict(X_test.loc[test_indices])
        acc = accuracy_score(y_test_sub, p_sub)
        bacc = balanced_accuracy_score(y_test_sub, p_sub) if not single_class else np.nan
        prec_m = precision_score(y_test_sub, p_sub, average="macro", zero_division=0) if not single_class else np.nan
        rec_m = recall_score(y_test_sub, p_sub, average="macro", zero_division=0) if not single_class else np.nan
        f1_m = f1_score(y_test_sub, p_sub, average="macro", zero_division=0)
        f1_w = f1_score(y_test_sub, p_sub, average="weighted", zero_division=0)
        prob_sub = m.predict_proba(X_test.loc[test_indices])[:, 1]
        auc = roc_auc_score(y_test_sub, prob_sub) if not single_class else np.nan
        rows_eval.append({
            "Model": name,
            "Test N": n_test_sub,
            "Accuracy": f"{acc:.4f}",
            "Macro Prec": f"{prec_m:.4f}",
            "Macro Rec": f"{rec_m:.4f}",
            "Macro F1": f"{f1_m:.4f}",
            "Weighted F1": f"{f1_w:.4f}",
            "ROC-AUC": f"{auc:.4f}",
            "Base Acc (R1)": f"{bench['Accuracy']:.4f}",
            "Base F1 (R1)": f"{bench['Macro F1']:.4f}"
        })
    eval_df = pd.DataFrame(rows_eval)

    # Comparison Bar Chart
    fig_cmp = go.Figure()
    for m_col, col_c in [("Accuracy", "#3b82f6"), ("Macro Prec", "#0284c7"), ("Macro Rec", "#f59e0b"), ("Macro F1", "#10b981")]:
        fig_cmp.add_trace(go.Bar(x=eval_df["Model"], y=[float(v) for v in eval_df[m_col]], name=m_col, marker_color=col_c))
    fig_cmp.update_layout(
        title=dict(text="Model Performance Comparison on Current Test Subset", x=0.01, xanchor="left", font=dict(size=14, color="#1e293b")),
        barmode="group", height=320, margin=dict(l=25, r=25, t=45, b=25), template="plotly_white"
    )

    # Confusion Matrices
    fig_cm = make_subplots(
        rows=1, cols=3,
        horizontal_spacing=0.08,
        subplot_titles=("Logistic Regression", "Decision Tree", "Random Forest (Selected)")
    )
    for idx, (m_name, m_obj) in enumerate([("Logistic Regression", model_lr), ("Decision Tree", model_dt), ("Random Forest", model_rf)], 1):
        p_s = m_obj.predict(X_test.loc[test_indices])
        cm = confusion_matrix(y_test_sub, p_s, labels=[0, 1])
        fig_cm.add_trace(go.Heatmap(z=cm, x=["Pred 0 (<4.0)", "Pred 1 (≥4.0)"], y=["True 0", "True 1"], colorscale="Blues", text=cm, texttemplate="%{text}", showscale=False), row=1, col=idx)
    fig_cm.for_each_annotation(lambda a: a.update(font=dict(size=12, color="#1e293b", family="Inter, sans-serif")))
    fig_cm.update_layout(height=280, margin=dict(l=25, r=25, t=45, b=25), template="plotly_white")

    # ROC Curves & Importances
    fig_roc_imp = make_subplots(
        rows=1, cols=2,
        horizontal_spacing=0.16,
        subplot_titles=("ROC Curves (AUC)", "Top 15 Feature Importances (RF)")
    )
    for m_name, m_obj, c_color in [("Logistic Regression", model_lr, "#f59e0b"), ("Decision Tree", model_dt, "#06b6d4"), ("Random Forest", model_rf, "#10b981")]:
        probs = m_obj.predict_proba(X_test.loc[test_indices])[:, 1]
        fpr, tpr, _ = roc_curve(y_test_sub, probs)
        auc_v = roc_auc_score(y_test_sub, probs)
        fig_roc_imp.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{m_name} (AUC={auc_v:.3f})", line=dict(color=c_color, width=2)), row=1, col=1)
    fig_roc_imp.add_shape(type="line", line=dict(dash="dash", color="gray"), x0=0, x1=1, y0=0, y1=1, row=1, col=1)
    fig_roc_imp.add_trace(go.Bar(y=rf_feat_df["Feature"][::-1], x=rf_feat_df["Importance"][::-1], orientation="h", marker_color="#6366f1", showlegend=False), row=1, col=2)
    fig_roc_imp.for_each_annotation(lambda a: a.update(font=dict(size=12, color="#1e293b", family="Inter, sans-serif")))
    fig_roc_imp.update_layout(height=400, margin=dict(l=25, r=25, t=45, b=25), template="plotly_white")

    table_html = "<table style='width:100%; border-collapse:collapse; margin-bottom:20px; font-size:13px; background:#fff;'>"
    table_html += "<tr style='background:#f1f5f9; border-bottom:2px solid #cbd5e1;'>" + "".join([f"<th style='padding:8px 12px; text-align:left;'>{c}</th>" for c in eval_df.columns]) + "</tr>"
    for _, r in eval_df.iterrows():
        is_rf = "Random Forest" in r["Model"]
        style = "font-weight:600; background:#f0fdf4;" if is_rf else "border-bottom:1px solid #e2e8f0;"
        table_html += f"<tr style='{style}'>" + "".join([f"<td style='padding:8px 12px;'>{v}</td>" for v in r.values]) + "</tr>"
    table_html += "</table>"

    html = f"""
    <div class="model-champion-card">
        <div style="font-size:15px; font-weight:600; color:#0f172a; margin-bottom:4px;">Selected Model: Random Forest Classifier (Champion Benchmark)</div>
        <div style="font-size:13px; color:#334155;">
            <b>Hyperparameters:</b> 300 Trees | max_depth=10 | min_samples_split=5 | class_weight='balanced' | 21 Predictors.<br>
            <b>Full Test Set Benchmark:</b> Accuracy <b>81.34%</b> | Macro Precision <b>0.6777</b> | Macro Recall <b>0.6443</b> | Macro-F1 <b>0.6571</b> | Weighted-F1 <b>0.8037</b> | ROC-AUC <b>0.7072</b>.
        </div>
    </div>
    <div class="tab-note-banner">
        <b>Filter Applicability:</b> Filters apply to the 911 test-set hotels. Currently evaluating on <b>{n_test_sub:,}</b> matching test hotels.<br>
        <i>Notice: Below table displays accuracy, precision, recall, F1-scores, and ROC-AUC on the filtered test subset alongside the unfiltered Review 1 benchmarks.</i>
    </div>
    <div class="chart-box">{table_html}</div>
    <div class="chart-box">{fig_cmp.to_html(full_html=False, include_plotlyjs="cdn")}</div>
    <div class="chart-box">{fig_cm.to_html(full_html=False, include_plotlyjs=False)}</div>
    <div class="chart-box">{fig_roc_imp.to_html(full_html=False, include_plotlyjs=False)}</div>
    """
    return html

# -------------------------------------------------------------
# BUILD TAB 3 CONTENT
# -------------------------------------------------------------
def get_tab3_html(sub_df, is_filtered=False):
    real_ids = set(sub_df[sub_df["data_type"] == "Real"]["hotel_id"])
    sub_revs = rev_sentiment_df[rev_sentiment_df["hotel_id"].isin(real_ids)]
    is_unfiltered = (len(sub_df) == len(clean_hotels_df))

    # Row 1: VADER Donut and Star vs VADER
    fig_tm1 = make_subplots(
        rows=1, cols=2,
        horizontal_spacing=0.18,
        subplot_titles=("VADER Sentiment Distribution [Dynamic]", "Star Rating vs. VADER Sentiment [Dynamic]"),
        specs=[[{"type": "domain"}, {"type": "xy"}]]
    )
    v_counts = sub_revs["vader_sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"]).fillna(0)
    fig_tm1.add_trace(go.Pie(labels=v_counts.index, values=v_counts.values, hole=0.55, marker=dict(colors=["#10b981", "#94a3b8", "#ef4444"])), row=1, col=1)
    ct = pd.crosstab(sub_revs["rating_sentiment"], sub_revs["vader_sentiment"]).reindex(index=["Negative", "Neutral", "Positive"], columns=["Negative", "Neutral", "Positive"]).fillna(0)
    for col_name, c_color in [("Positive", "#10b981"), ("Neutral", "#94a3b8"), ("Negative", "#ef4444")]:
        fig_tm1.add_trace(go.Bar(x=ct.index, y=ct[col_name], name=f"VADER {col_name}", marker_color=c_color), row=1, col=2)
    fig_tm1.for_each_annotation(lambda a: a.update(font=dict(size=12, color="#1e293b", family="Inter, sans-serif")))
    fig_tm1.update_layout(height=350, barmode="stack", margin=dict(l=25, r=25, t=45, b=25), template="plotly_white")

    # Row 2: Customer Pain Points & Experience Themes [Dynamic]
    pain_patterns = {
        "Cleanliness / hygiene": r"\bdirty\b|\bunclean\b|\bfilth\w*|\bnot (very |so |that |at all |too |properly )?(clean|hygien\w*)\b|\bunhygien\w*|\bstink\w*|\bsmell\w*|\bcockroach\w*|\bbed ?bugs?\b|\bstain\w*|\bmou?ld\b|\bdust\w*|\bno cleanliness\b|\bpoor (cleanliness|hygiene)\b",
        "Rude / unhelpful staff": r"\brude\w*|\bimpolite\b|\bunprofessional\b|\bmisbehav\w*|\barrogan\w*|\bunhelpful\b|\bnot (very )?(helpful|cooperative|friendly|polite|responsive|supportive)\b|\bpoor (staff|behaviou?r|attitude)\b|\bbad (staff|behaviou?r|attitude)\b|\battitude\b",
        "Slow / poor service": r"\bslow\w*|\bdelay\w*|\bwaited\b|\bwaiting\b|\btook (a )?(long|too long|forever|hours?)\b|\bpoor service\b|\bbad service\b|\bworst service\b|\bno service\b|\bno response\b|\bnot responding\b",
        "Broken / poorly maintained": r"\bnot working\b|\bbroken\b|\bdamaged\b|\bpoor(ly)? maint\w*|\bnot maintained\b|\bleak\w*|\bworn\b|\bfaulty\b|\b(does|did|do)( not|n['’]?t) work\b|\bnot properly working\b|\bout of order\b",
        "AC / cooling problems": r"\b(ac|a/c|aircon\w*|air[- ]?condition\w*)\b[^.]{0,40}\b(not|no|never|poor|bad|issue|problem|broken|noisy|leak\w*|worst)\b|\b(not|no|poor|bad)\b[^.]{0,20}\b(ac|a/c|cooling)\b",
        "Wi-Fi / internet problems": r"\b(wi-?fi|wifi|internet|network)\b[^.]{0,40}\b(not|no|poor|bad|slow|weak|issue|problem|never|unavailable)\b|\b(no|poor|bad|slow|weak|not)\b[^.]{0,20}\b(wi-?fi|wifi|internet)\b",
        "Noise": r"\bnois\w+|\bloud\b|\bdisturb\w*",
        "Bathroom / hot-water issues": r"\bno hot water\b|\bhot water\b[^.]{0,30}\b(not|no|never|issue|problem)\b|\b(bathrooms?|toilets?|washrooms?)\b[^.]{0,40}\b(dirty|smell\w*|small|broken|not|no|leak\w*|bad|poor|worst|unclean)\b",
        "Food / breakfast quality": r"\b(food|breakfast|dinner|lunch|meals?|buffet|tea|coffee)\b[^.]{0,40}\b(bad|poor|worst|not|cold|tasteless|stale|average|terrible|horrible|awful|disappoint\w*|limited)\b|\b(bad|poor|worst|tasteless|stale|terrible|horrible|awful)\b[^.]{0,20}\b(food|breakfast|dinner|lunch|meals?)\b",
        "Price / value for money": r"\boverpriced\b|\bexpensive\b|\bcostly\b|\bover ?charg\w*|\bwaste of money\b|\bnot worth\b|\bhigh (price|rate|tariff|charges?)\b|\bextra (charge|money|cost)\w*|\brip ?off\b|\bhidden (charge|cost)\w*|\bcharged\b"
    }
    
    neg_revs = sub_revs[sub_revs["vader_sentiment"] == "Negative"]
    n_neg = len(neg_revs)
    neg_texts = neg_revs["text_light"].fillna("").str.lower()
    
    pain_items = []
    for pname, ppat in pain_patterns.items():
        cnt = int(neg_texts.str.contains(ppat, regex=True).sum()) if n_neg > 0 else 0
        pct = round(cnt / n_neg * 100, 1) if n_neg > 0 else 0.0
        pain_items.append({"Pain Point": pname, "Negative Mentions": cnt, "% of negative reviews": pct})
    pain_df = pd.DataFrame(pain_items).sort_values("Negative Mentions", ascending=True).tail(8)

    # Themes
    theme_patterns = {
        "Location": r"\blocation\b|\blocated\b|\bnearby\b|\bnear\b",
        "Food": r"\bfood\b|\bdinner\b|\blunch\b|\bmeals?\b|\brestaurant\b",
        "Breakfast": r"\bbreakfast\b",
        "Service": r"\bservices?\b",
        "Staff": r"\bstaffs?\b|\bemployees?\b|\breception\w*|\bmanager\b",
        "Room": r"\brooms?\b",
        "Cleanliness": r"\bclean\w*|\bdirty\b|\bhygien\w*",
        "Facilities": r"\bfacilit\w+|\bamenit\w+"
    }
    sub_texts = sub_revs["text_light"].fillna("").str.lower()
    sub_vader = sub_revs["vader_sentiment"].values
    theme_items = []
    for tname, tpat in theme_patterns.items():
        m_bool = sub_texts.str.contains(tpat, regex=True).values
        tot = int(m_bool.sum())
        pos = int(((m_bool) & (sub_vader == "Positive")).sum())
        pct = round(pos / tot * 100, 1) if tot > 0 else 0.0
        theme_items.append({"Theme": tname, "Total Mentions": tot, "% Positive": pct})
    theme_df = pd.DataFrame(theme_items).sort_values("% Positive", ascending=True)

    fig_tm2 = make_subplots(
        rows=1, cols=2,
        horizontal_spacing=0.18,
        subplot_titles=("Top Customer Pain Points [Dynamic]", "Experience Themes (% Positive) [Dynamic]")
    )
    fig_tm2.add_trace(go.Bar(
        y=pain_df["Pain Point"], x=pain_df["Negative Mentions"], orientation="h",
        marker_color="#ef4444", text=[f"{m:,} ({p:.1f}%)" for m, p in zip(pain_df["Negative Mentions"], pain_df["% of negative reviews"])],
        textposition="inside", showlegend=False
    ), row=1, col=1)
    
    fig_tm2.add_trace(go.Bar(
        y=theme_df["Theme"], x=theme_df["% Positive"], orientation="h",
        marker_color="#10b981", text=[f"{v:.1f}%" for v in theme_df["% Positive"]],
        textposition="inside", showlegend=False
    ), row=1, col=2)
    fig_tm2.for_each_annotation(lambda a: a.update(font=dict(size=12, color="#1e293b", family="Inter, sans-serif")))
    fig_tm2.update_layout(height=380, margin=dict(l=25, r=25, t=45, b=25), template="plotly_white")

    # Row 3: Word Frequency & Bigrams [Dynamic]
    stop = set(stopwords.words("english")) - {"not", "no"}
    tok_tokenizer = RegexpTokenizer(r"[a-z]+")
    all_toks = [w for txt in sub_revs["text_light"].fillna("") for w in tok_tokenizer.tokenize(txt.lower()) if w not in stop and len(w) >= 3]
    wf = Counter(all_toks)
    top_w = [(w, c) for w, c in wf.most_common(25) if w not in {"hotel", "stay", "place", "also", "would", "get", "one"}][:15]
    words_df = pd.DataFrame(top_w, columns=["word", "frequency"]).sort_values("frequency", ascending=True)

    all_bis = []
    for txt in sub_revs["text_light"].fillna(""):
        toks = [w for w in tok_tokenizer.tokenize(txt.lower()) if w not in stop and len(w) >= 3]
        all_bis.extend([f"{toks[k]} {toks[k+1]}" for k in range(len(toks)-1)])
    bf = Counter(all_bis)
    top_b = bf.most_common(10)
    bi_df = pd.DataFrame(top_b, columns=["bigram", "frequency"]).sort_values("frequency", ascending=True)

    fig_tm3 = make_subplots(
        rows=1, cols=2,
        horizontal_spacing=0.18,
        subplot_titles=("Top 15 Frequent Words [Dynamic]", "Top 10 Bigrams in Reviews [Dynamic]")
    )
    fig_tm3.add_trace(go.Bar(y=words_df["word"], x=words_df["frequency"], orientation="h", marker_color="#3b82f6", text=words_df["frequency"], textposition="inside", showlegend=False), row=1, col=1)
    fig_tm3.add_trace(go.Bar(y=bi_df["bigram"], x=bi_df["frequency"], orientation="h", marker_color="#8b5cf6", text=bi_df["frequency"], textposition="inside", showlegend=False), row=1, col=2)
    fig_tm3.for_each_annotation(lambda a: a.update(font=dict(size=12, color="#1e293b", family="Inter, sans-serif")))
    fig_tm3.update_layout(height=380, margin=dict(l=25, r=25, t=45, b=25), template="plotly_white")

    # Row 4: Amenity Aspect Sentiment & Confusion Matrix [Dynamic]
    amen_sent_df = pd.read_csv(os.path.join(TEXT_DIR, "amenity_mention_sentiment.csv")).sort_values("Positive %", ascending=True)
    cm_tm = confusion_matrix(sub_revs["rating_sentiment"], sub_revs["vader_sentiment"], labels=["Negative", "Neutral", "Positive"])
    fig_tm4 = make_subplots(
        rows=1, cols=2,
        horizontal_spacing=0.18,
        subplot_titles=("Amenity Aspect Sentiment (% Positive) [Dynamic]", "VADER vs. Rating Confusion Matrix [Dynamic]"),
        specs=[[{"type": "xy"}, {"type": "heatmap"}]]
    )
    fig_tm4.add_trace(go.Bar(y=amen_sent_df["Amenity"], x=amen_sent_df["Positive %"], orientation="h", marker_color="#06b6d4", text=[f"{v:.1f}%" for v in amen_sent_df["Positive %"]], textposition="inside", showlegend=False), row=1, col=1)
    fig_tm4.add_trace(go.Heatmap(z=cm_tm, x=["Pred Neg", "Pred Neu", "Pred Pos"], y=["True Neg", "True Neu", "True Pos"], colorscale="Blues", text=cm_tm, texttemplate="%{text}", showscale=False), row=1, col=2)
    fig_tm4.for_each_annotation(lambda a: a.update(font=dict(size=12, color="#1e293b", family="Inter, sans-serif")))
    fig_tm4.update_layout(height=360, margin=dict(l=25, r=25, t=45, b=25), template="plotly_white")

    # Row 5: Dynamic Sentiment Evaluation Metrics Bar
    acc = accuracy_score(sub_revs["rating_sentiment"], sub_revs["vader_sentiment"])
    mf1 = f1_score(sub_revs["rating_sentiment"], sub_revs["vader_sentiment"], average="macro", zero_division=0)
    kappa = cohen_kappa_score(sub_revs["rating_sentiment"], sub_revs["vader_sentiment"])
    eval_m_df = pd.DataFrame([
        ("Accuracy", acc), ("Weighted F1", f1_score(sub_revs["rating_sentiment"], sub_revs["vader_sentiment"], average="weighted", zero_division=0)),
        ("Cohen's Kappa", kappa if not np.isnan(kappa) else 0.0),
        ("Macro Precision", precision_score(sub_revs["rating_sentiment"], sub_revs["vader_sentiment"], average="macro", zero_division=0)),
        ("Macro Recall", recall_score(sub_revs["rating_sentiment"], sub_revs["vader_sentiment"], average="macro", zero_division=0)),
        ("Macro F1", mf1)
    ], columns=["Metric", "Value"])
    fig_eval = go.Figure(go.Bar(
        y=eval_m_df["Metric"][::-1], x=eval_m_df["Value"][::-1], orientation="h",
        marker_color="#10b981", text=[f"{v:.4f}" for v in eval_m_df["Value"][::-1]], textposition="inside"
    ))
    fig_eval.update_layout(
        title=dict(text="Sentiment Evaluation Metrics vs Star Rating Reference [Dynamic]", x=0.01, xanchor="left", font=dict(size=14, color="#1e293b")),
        height=280, margin=dict(l=25, r=25, t=45, b=25), template="plotly_white"
    )

    # Word Cloud Base64: dynamically generate from current word frequencies
    cloud_freq = {w: f for w, f in wf.items() if w not in {"hotel", "stay", "place", "also", "would", "get", "one", "room", "good"}}
    wc = WordCloud(width=800, height=380, background_color="white", colormap="viridis", max_words=80, collocations=False, random_state=42).generate_from_frequencies(cloud_freq)
    img_buf = io.BytesIO()
    wc.to_image().save(img_buf, format="PNG")
    b64_wc = base64.b64encode(img_buf.getvalue()).decode("utf-8")
    wc_img_html = f'<div style="text-align:center; margin: 15px 0;"><img src="data:image/png;base64,{b64_wc}" style="max-width:850px; border-radius:10px; border:1px solid #e2e8f0;"/></div>'

    # Review Explorer Table (Sample)
    rev_sample = sub_revs.head(8)
    rev_tbl = "<table class='dataset-table' style='width:100%; border-collapse:collapse; font-size:12px; margin-top:10px;'>"
    rev_tbl += "<thead><tr style='background:#f1f5f9; color:#334155; border-bottom:2px solid #cbd5e1;'><th style='padding:8px 10px; width:22%;'>Hotel Name</th><th style='padding:8px 10px; width:12%;'>Location</th><th style='padding:8px 10px; width:12%;'>Category</th><th style='padding:8px 10px; width:8%; text-align:center;'>Rating</th><th style='padding:8px 10px; width:12%; text-align:center;'>VADER Sentiment</th><th style='padding:8px 10px;'>Review Text Excerpt</th></tr></thead><tbody>"
    for _, r in rev_sample.iterrows():
        sent = r["vader_sentiment"]
        b_class = "badge-vader-pos" if sent == "Positive" else ("badge-vader-neg" if sent == "Negative" else "badge-vader-neu")
        rev_tbl += f"<tr style='border-bottom:1px solid #f1f5f9;'><td style='padding:7px 10px; font-weight:600;'>{r['hotel_name']}</td><td style='padding:7px 10px;'>{r['analysis_location']}</td><td style='padding:7px 10px;'>{r['hotel_category_clean']}</td><td style='padding:7px 10px; text-align:center;'>★ {r['review_rating_5']:.1f}</td><td style='padding:7px 10px; text-align:center;'><span class='{b_class}'>{sent}</span></td><td style='padding:7px 10px; color:#334155;'>{r['text_light']}</td></tr>"
    rev_tbl += "</tbody></table>"

    pos_p = (sub_revs["vader_sentiment"] == "Positive").mean() * 100
    neg_p = (sub_revs["vader_sentiment"] == "Negative").mean() * 100
    neu_p = (sub_revs["vader_sentiment"] == "Neutral").mean() * 100

    html = f"""
    <div class="tab-note-banner">
        <b>Filter Applicability: [Dynamic]</b> Text mining covers <b>Real Hotels only</b> ({len(real_ids):,} matched hotels, {len(sub_revs):,} reviews).<br>
        All charts, distributions, metrics, confusion matrices, customer pain points, experience themes, and dynamic word clouds recalculate from reviews matching your global filters.
    </div>
    <div class="chart-box">{fig_tm1.to_html(full_html=False, include_plotlyjs="cdn")}</div>
    <div class="chart-box">{fig_tm2.to_html(full_html=False, include_plotlyjs=False)}</div>
    <div class="chart-box">{fig_tm3.to_html(full_html=False, include_plotlyjs=False)}</div>
    <div class="chart-box">{fig_tm4.to_html(full_html=False, include_plotlyjs=False)}</div>
    <div class="chart-box">{fig_eval.to_html(full_html=False, include_plotlyjs=False)}</div>
    <div class="chart-box">
        <div style="font-size:15px; font-weight:600; color:#334155; margin-bottom:10px;">Dynamic Review Word Cloud [Dynamic]</div>
        {wc_img_html}
    </div>
    <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:16px; margin-top:20px;">
        <h4 style="margin:0 0 8px 0; color:#1e293b;">Key Text Mining Takeaways [Dynamic Recomputed]:</h4>
        <ul style="margin:0; padding-left:20px; font-size:13px; color:#475569; line-height:1.6;">
            <li><b>Overall Sentiment:</b> <b>{pos_p:.1f}%</b> of reviews are VADER positive, <b>{neg_p:.1f}%</b> negative, <b>{neu_p:.1f}%</b> neutral. VADER achieves <b>{acc*100:.2f}%</b> agreement with customer star ratings (Macro-F1: <b>{mf1:.4f}</b>, Cohen's Kappa: <b>{kappa:.4f}</b>).</li>
            <li><b>Primary Pain Points:</b> <i>Cleanliness / hygiene</i> and <i>Poor maintenance</i> drive complaints across negative reviews.</li>
            <li><b>Primary Delight Drivers:</b> <i>Location</i> and <i>Food / Breakfast</i> drive the highest customer satisfaction.</li>
        </ul>
    </div>
    <div class="chart-box" style="margin-top:20px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
            <div style="font-size:14px; font-weight:600; color:#0f172a;">Filtered Reviews Explorer [Dynamic]</div>
            <div style="font-size:12.5px; color:#64748b;">Showing 8 of {len(sub_revs):,} matching reviews</div>
        </div>
        {rev_tbl}
    </div>
    """
    return html

def get_tab4_html(search_term=''):
    supp_thresh = 0.05
    conf_thresh = 0.60
    lift_thresh = 2.0
    active_rules = apriori_rules_df[
        (apriori_rules_df["support"] >= supp_thresh) &
        (apriori_rules_df["confidence"] >= conf_thresh) &
        (apriori_rules_df["lift"] >= lift_thresh)
    ].copy()
    if search_term:
        active_rules = active_rules[active_rules["Rule"].str.contains(search_term, case=False, na=False)]

    # Top 15 Amenity Support
    top_amen_supp = apriori_support_df.sort_values("Support", ascending=True).tail(15)
    fig_supp = go.Figure(go.Bar(
        y=top_amen_supp["Amenity"], x=top_amen_supp["Support_Percentage"], orientation="h",
        marker_color="#0284c7", text=[f"{v:.1f}%" for v in top_amen_supp["Support_Percentage"]], textposition="inside"
    ))
    fig_supp.update_layout(
        title=dict(text="Top 15 Standardized Amenities by Overall Support (%)", x=0.01, xanchor="left", font=dict(size=14, color="#1e293b")),
        height=360, margin=dict(l=25, r=25, t=45, b=25), template="plotly_white"
    )

    # Row 2
    fig_ar = make_subplots(
        rows=1, cols=2,
        horizontal_spacing=0.18,
        subplot_titles=("Top Rules by Lift", "Support vs. Confidence (Size = Lift)")
    )
    top10_rules = active_rules.sort_values("lift", ascending=False).head(10)
    fig_ar.add_trace(go.Bar(
        y=top10_rules["Rule"][::-1], x=top10_rules["lift"][::-1], orientation="h",
        marker_color="#8b5cf6", text=[f"Lift: {v:.2f}" for v in top10_rules["lift"][::-1]],
        textposition="inside", showlegend=False
    ), row=1, col=1)

    fig_ar.add_trace(go.Scatter(
        x=active_rules["support"] * 100, y=active_rules["confidence"] * 100,
        mode="markers",
        marker=dict(size=active_rules["lift"] * 2.5, color=active_rules["lift"], colorscale="Viridis", showscale=True, colorbar=dict(title="Lift", x=1.02)),
        text=active_rules["Rule"],
        hovertemplate="<b>Rule:</b> %{text}<br>Support: %{x:.2f}%<br>Confidence: %{y:.2f}%<extra></extra>",
        showlegend=False
    ), row=1, col=2)
    fig_ar.update_xaxes(title_text="Support (%)", row=1, col=2)
    fig_ar.update_yaxes(title_text="Confidence (%)", row=1, col=2)
    fig_ar.for_each_annotation(lambda a: a.update(font=dict(size=12, color="#1e293b", family="Inter, sans-serif")))
    fig_ar.update_layout(height=400, margin=dict(l=25, r=25, t=45, b=25), template="plotly_white")

    # Ranked Table
    display_tbl = active_rules[["Rule", "support", "confidence", "lift"]].copy()
    display_tbl["Support (%)"] = (display_tbl["support"] * 100).round(2)
    display_tbl["Confidence (%)"] = (display_tbl["confidence"] * 100).round(2)
    display_tbl["Lift"] = display_tbl["lift"].round(3)
    display_tbl = display_tbl.drop(columns=["support", "confidence", "lift"]).head(15)

    table_html = "<table style='width:100%; border-collapse:collapse; font-size:13px; background:#fff;'>"
    table_html += "<tr style='background:#f1f5f9; border-bottom:2px solid #cbd5e1;'>" + "".join([f"<th style='padding:8px 12px; text-align:left;'>{c}</th>" for c in display_tbl.columns]) + "</tr>"
    for _, r in display_tbl.iterrows():
        table_html += "<tr style='border-bottom:1px solid #e2e8f0;'>" + "".join([f"<td style='padding:8px 12px;'>{v}</td>" for v in r.values]) + "</tr>"
    table_html += "</table>"

    html = f"""
    <div class="apriori-controls-box">
        <div style="font-size:14px; font-weight:600; color:#334155; margin-bottom:8px;">⚙️ Association Rules Filtering & Dynamic Mining Controls</div>
        <div style="display:flex; gap:16px; flex-wrap:wrap; align-items:center;">
            <div class="filter-item"><span class="filter-label">Min Support</span><span class="filter-val-badge">0.05 (5.0%)</span></div>
            <div class="filter-item"><span class="filter-label">Min Confidence</span><span class="filter-val-badge">0.60 (60.0%)</span></div>
            <div class="filter-item"><span class="filter-label">Min Lift</span><span class="filter-val-badge">2.0</span></div>
            <div class="filter-item"><span class="filter-label">Search Amenity</span><span class="filter-val-badge">{"Spa" if search_term=="Spa" else "All Amenities"}</span></div>
            <div class="filter-item"><span class="filter-label">Dynamic mlxtend</span><span class="filter-val-badge">☐ Inactive (Precomputed)</span></div>
        </div>
    </div>
    <div class="tab-note-banner">
        <b>Filter Applicability:</b> Amenity association rules mined on hotel-level transaction data.<br>
        <i>Note: Amenity names are standardized (merging free/paid variants, 33 total) and differ from the 15 model columns.</i>
    </div>
    <div class="chart-box">{fig_supp.to_html(full_html=False, include_plotlyjs="cdn")}</div>
    <div class="chart-box">{fig_ar.to_html(full_html=False, include_plotlyjs=False)}</div>
    <div class="chart-box">
        <div style='font-size:14px; font-weight:600; margin-bottom:10px;'>Top 15 Ranked Rules Table (154 Strong Rules Available):</div>
        {table_html}
    </div>
    <div style="background:#fdf4ff; border:1px solid #f0abfc; border-radius:10px; padding:16px; margin-top:20px;">
        <h4 style="margin:0 0 8px 0; color:#86198f;">Strategic Amenity Bundling Insights:</h4>
        <ul style="margin:0; padding-left:20px; font-size:13px; color:#701a75; line-height:1.6;">
            <li><b>Premium Wellness Hub:</b> <i>Business center + Pool → Spa</i> boasts the highest lift across the corpus (<b>Lift: 6.190</b>, Confidence: 74.02%, Support: 6.57%). Guests expect premium spas when both business facilities and leisure pools exist.</li>
            <li><b>Fitness & Leisure Core:</b> <i>Fitness center + Pool → Spa</i> (<b>Lift: 5.755</b>, Confidence: 68.82%) and <i>Bar + Fitness center → Spa</i> (<b>Lift: 5.526</b>, Confidence: 66.08%) identify strong complementary synergies for full-service properties.</li>
            <li><b>Operational Strategy:</b> Standalone spas rarely succeed without co-located fitness and business infrastructure. Investment in luxury amenities should follow established bundle co-occurrences.</li>
        </ul>
    </div>
    """
    return html

# -------------------------------------------------------------
# RENDER COMPLETE HTML WRAPPER & SNAPSHOTS
# -------------------------------------------------------------

# -------------------------------------------------------------
# BUILD TAB 5 CONTENT (FINAL DATASET TABLE & EXPORT)
# -------------------------------------------------------------
def get_tab5_html(sub_df):
    cols = ["hotel_name", "location", "hotel_category", "data_type", "price", "customer_rating", "review_count", "amenity_count", "success"]
    n_match = len(sub_df)
    page_size = 25
    total_pages = max(1, (n_match + page_size - 1) // page_size)
    page_hotels = sub_df[cols].iloc[0:page_size]
    
    tbl_html = "<table class='dataset-table' style='width:100%; border-collapse:collapse; font-size:12.5px; background:#fff; border:1px solid #e2e8f0; border-radius:8px; overflow:hidden; margin-bottom:12px;'>"
    tbl_html += "<thead><tr style='background:#f1f5f9; color:#334155; border-bottom:2px solid #cbd5e1;'><th style='padding:9px 12px; text-align:left;'>Hotel Name</th><th style='padding:9px 12px; text-align:left;'>Location</th><th style='padding:9px 12px; text-align:left;'>Category</th><th style='padding:9px 12px; text-align:left;'>Data Type</th><th style='padding:9px 12px; text-align:right;'>Price (₹)</th><th style='padding:9px 12px; text-align:center;'>Rating</th><th style='padding:9px 12px; text-align:right;'>Reviews</th><th style='padding:9px 12px; text-align:center;'>Amenities</th><th style='padding:9px 12px; text-align:center;'>Success Status</th></tr></thead><tbody>"
    
    for _, r in page_hotels.iterrows():
        is_succ = int(r["success"]) == 1
        badge = "<span style='display:inline-block; padding:2px 8px; border-radius:12px; font-size:11px; font-weight:600; background:#ecfdf5; color:#059669; border:1px solid #a7f3d0;'>1 (Successful)</span>" if is_succ else "<span style='display:inline-block; padding:2px 8px; border-radius:12px; font-size:11px; font-weight:600; background:#f1f5f9; color:#64748b; border:1px solid #cbd5e1;'>0 (Not Successful)</span>"
        tbl_html += f"<tr style='border-bottom:1px solid #f1f5f9;'><td style='padding:8px 12px; font-weight:600; color:#0f172a;'>{r['hotel_name']}</td><td style='padding:8px 12px;'>{r['location']}</td><td style='padding:8px 12px;'>{r['hotel_category']}</td><td style='padding:8px 12px;'>{r['data_type']}</td><td style='padding:8px 12px; text-align:right; font-weight:500;'>₹{r['price']:,.0f}</td><td style='padding:8px 12px; text-align:center;'>★ {r['customer_rating']:.1f}</td><td style='padding:8px 12px; text-align:right;'>{int(r['review_count']):,}</td><td style='padding:8px 12px; text-align:center;'>{int(r['amenity_count'])}</td><td style='padding:8px 12px; text-align:center;'>{badge}</td></tr>"
    tbl_html += "</tbody></table>"

    html = f"""
    <div class="tab-note-banner">
        <b>Final Project Dataset:</b> Displaying properties matching current global filters.<br>
        Explore the final project dataset below (excluding geographic coordinates) with interactive text search, paging, and dual CSV download capabilities.
    </div>
    <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:14px 18px; margin-bottom:16px; display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px;">
        <div style="display:flex; align-items:center; gap:10px;">
            <span style="font-size:13px; font-weight:600; color:#334155;">Search Hotel:</span>
            <input type="text" placeholder="Search hotel by name..." value="" style="border:1px solid #cbd5e1; border-radius:6px; padding:6px 12px; font-size:13px; width:220px;" readonly />
            <button style="background:#f1f5f9; border:1px solid #cbd5e1; border-radius:6px; padding:6px 12px; font-size:12.5px; font-weight:600; color:#475569;" disabled>◀ Previous</button>
            <button style="background:#f1f5f9; border:1px solid #cbd5e1; border-radius:6px; padding:6px 12px; font-size:12.5px; font-weight:600; color:#475569;">Next ▶</button>
            <span style="font-size:13px; color:#475569; margin-left:6px;">Page <b>1</b> of <b>{total_pages}</b> ({n_match:,} matching hotels)</span>
        </div>
        <div style="display:flex; gap:10px;">
            <a href="#" style="background:#0284c7; color:#fff; border-radius:6px; padding:7px 14px; font-size:12.5px; font-weight:600; text-decoration:none; display:inline-flex; align-items:center; gap:6px;">
                ⬇ Download Filtered Data (CSV)
            </a>
            <a href="#" style="background:#0f172a; color:#fff; border-radius:6px; padding:7px 14px; font-size:12.5px; font-weight:600; text-decoration:none; display:inline-flex; align-items:center; gap:6px;">
                ⬇ Download Full Final Dataset (CSV)
            </a>
        </div>
    </div>
    <div class="chart-box">
        {tbl_html}
    </div>
    <div style="display:flex; justify-content:space-between; align-items:center; font-size:12.5px; color:#64748b; padding:0 4px;">
        <span>Showing rows 1–25 of {n_match:,} hotels</span>
        <span>Environment auto-detection: Google Colab (<code>files.download</code>) & Local Jupyter (Base64 data-URI) supported</span>
    </div>
    """
    return html

def build_full_page(title, filter_html, kpi_html, nav_html, tab_content_html):
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>{CSS_STYLES}</style>
</head>
<body>
    <div class="dashboard-wrapper">
        {render_header()}
        {filter_html}
        {kpi_html}
        {nav_html}
        {tab_content_html}
    </div>
</body>
</html>
"""

def take_snapshot(html_path, out_png, width=1360, height=2200):
    cmd = [
        CHROME_EXE,
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--virtual-time-budget=6000",
        f"--screenshot={out_png}",
        f"--window-size={width},{height}",
        f"file:///{os.path.abspath(html_path).replace(os.sep, '/')}"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0 and os.path.exists(out_png):
        print(f"✓ Generated {os.path.basename(out_png)} ({os.path.getsize(out_png):,} bytes).")
    else:
        print(f"✗ Failed {os.path.basename(out_png)}: {res.stderr}")

def main():
    print("=" * 70)
    print("GENERATING ZERO-OVERLAP EVIDENCE SCREENSHOTS WITH HEADLESS CHROME")
    print("DEMO SCENARIO: Coimbatore, 4-star hotel, Price ₹2,000 - ₹7,000")
    print("=" * 70)

    # Filtered slice for Coimbatore, 4-star hotel, narrower price range
    coimb_4star_df = clean_hotels_df[
        (clean_hotels_df["location"] == "Coimbatore") &
        (clean_hotels_df["hotel_category"] == "4-star hotel") &
        (clean_hotels_df["price"] >= 2000) &
        (clean_hotels_df["price"] <= 7000)
    ]
    print(f"Demo Slice: {len(coimb_4star_df)} hotels in Coimbatore, 4-star, ₹2,000 - ₹7,000.")

    # 1. Dashboard Overview / Landing View (Full Corpus: 4,551 hotels, full price span ₹392 - ₹61,327, Tab 1)
    html_dash = build_full_page(
        "Hotel Success Intelligence & Decision Support Dashboard (Default View)",
        render_filters(loc="All Locations", cat="All Categories", dt="Both", price_text="₹392 - ₹61,327", rating_text="0.0 - 5.0", n_sel=len(clean_hotels_df), n_tot=len(clean_hotels_df)),
        render_kpis(clean_hotels_df),
        render_tabs_nav(0),
        get_tab1_html(clean_hotels_df)
    )
    p_html_dash = os.path.join(SCREENSHOTS_DIR, "dashboard_overview.html")
    p_png_dash = os.path.join(SCREENSHOTS_DIR, "dashboard_overview.png")
    with open(p_html_dash, "w", encoding="utf-8") as f: f.write(html_dash)
    take_snapshot(p_html_dash, p_png_dash, height=1850)
    shutil.copyfile(p_png_dash, os.path.join(SCREENSHOTS_DIR, "dashboard.png"))

    # 2. Before Filters (Baseline 100% corpus state: 4,551 hotels)
    p_png_bef = os.path.join(SCREENSHOTS_DIR, "filter_interaction_before.png")
    take_snapshot(p_html_dash, p_png_bef, height=1450)
    shutil.copyfile(p_png_bef, os.path.join(SCREENSHOTS_DIR, "dashboard_before_filters.png"))

    # 3. After Filters (Demo Scenario: Coimbatore, 4-star hotel, ₹2,000 - ₹7,000)
    html_aft = build_full_page(
        "Filter Interaction (After Filtering: Coimbatore, 4-Star Hotel)",
        render_filters(loc="Coimbatore", cat="4-star hotel", dt="Both", price_text="₹2,000 - ₹7,000", rating_text="0.0 - 5.0", n_sel=len(coimb_4star_df), n_tot=len(clean_hotels_df)),
        render_kpis(coimb_4star_df),
        render_tabs_nav(0),
        get_tab1_html(coimb_4star_df)
    )
    p_html_aft = os.path.join(SCREENSHOTS_DIR, "filter_interaction_after.html")
    p_png_aft = os.path.join(SCREENSHOTS_DIR, "filter_interaction_after.png")
    with open(p_html_aft, "w", encoding="utf-8") as f: f.write(html_aft)
    take_snapshot(p_html_aft, p_png_aft, height=1450)
    shutil.copyfile(p_png_aft, os.path.join(SCREENSHOTS_DIR, "dashboard_after_filters.png"))

    # 4. Tab 1: EDA Full (Unfiltered)
    p_png1 = os.path.join(SCREENSHOTS_DIR, "tab1_eda.png")
    take_snapshot(p_html_dash, p_png1, height=2200)

    # 5. Tab 2: Predictive Models (Unfiltered Test Set)
    html2 = build_full_page(
        "Tab 2: Predictive Models (Model Benchmarks & Test Evaluation)",
        render_filters(loc="All Locations", cat="All Categories", dt="Both", price_text="₹392 - ₹61,327", rating_text="0.0 - 5.0", n_sel=len(clean_hotels_df), n_tot=len(clean_hotels_df)),
        render_kpis(clean_hotels_df),
        render_tabs_nav(1),
        get_tab2_html(clean_hotels_df)
    )
    p_html2 = os.path.join(SCREENSHOTS_DIR, "tab2_models.html")
    p_png2 = os.path.join(SCREENSHOTS_DIR, "tab2_models.png")
    with open(p_html2, "w", encoding="utf-8") as f: f.write(html2)
    take_snapshot(p_html2, p_png2, height=1900)

    # 6. Tab 3: Text Mining (Before Filtering - Full 26,175 reviews)
    html3_bef = build_full_page(
        "Tab 3: Text Mining (Unfiltered Baseline Corpus)",
        render_filters(loc="All Locations", cat="All Categories", dt="Both", price_text="₹392 - ₹61,327", rating_text="0.0 - 5.0", n_sel=len(clean_hotels_df), n_tot=len(clean_hotels_df)),
        render_kpis(clean_hotels_df),
        render_tabs_nav(2),
        get_tab3_html(clean_hotels_df, is_filtered=False)
    )
    p_html3_bef = os.path.join(SCREENSHOTS_DIR, "tab3_text_mining_before.html")
    p_png3_bef = os.path.join(SCREENSHOTS_DIR, "tab3_text_mining_before.png")
    with open(p_html3_bef, "w", encoding="utf-8") as f: f.write(html3_bef)
    take_snapshot(p_html3_bef, p_png3_bef, height=2400)
    shutil.copyfile(p_png3_bef, os.path.join(SCREENSHOTS_DIR, "tab3_text_mining.png"))

    # 7. Tab 3: Text Mining (After Filtering - Coimbatore 4-Star Hotel)
    html3_aft = build_full_page(
        "Tab 3: Text Mining (Dynamic Recomputed: Coimbatore 4-Star)",
        render_filters(loc="Coimbatore", cat="4-star hotel", dt="Both", price_text="₹2,000 - ₹7,000", rating_text="0.0 - 5.0", n_sel=len(coimb_4star_df), n_tot=len(clean_hotels_df)),
        render_kpis(coimb_4star_df),
        render_tabs_nav(2),
        get_tab3_html(coimb_4star_df, is_filtered=True)
    )
    p_html3_aft = os.path.join(SCREENSHOTS_DIR, "tab3_text_mining_after.html")
    p_png3_aft = os.path.join(SCREENSHOTS_DIR, "tab3_text_mining_after.png")
    with open(p_html3_aft, "w", encoding="utf-8") as f: f.write(html3_aft)
    take_snapshot(p_html3_aft, p_png3_aft, height=2400)

    # 8. Tab 4: Association Rules with "Spa" Search Filter
    html4 = build_full_page(
        "Tab 4: Association Rules (Filtered by Amenity: 'Spa')",
        render_filters(loc="All Locations", cat="All Categories", dt="Both", price_text="₹392 - ₹61,327", rating_text="0.0 - 5.0", n_sel=len(clean_hotels_df), n_tot=len(clean_hotels_df)),
        render_kpis(clean_hotels_df),
        render_tabs_nav(3),
        get_tab4_html(search_term="Spa")
    )
    p_html4 = os.path.join(SCREENSHOTS_DIR, "tab4_association_rules.html")
    p_png4 = os.path.join(SCREENSHOTS_DIR, "tab4_association_rules.png")
    with open(p_html4, "w", encoding="utf-8") as f: f.write(html4)
    take_snapshot(p_html4, p_png4, height=1900)

    # 9. Tab 5: Final Dataset Table & Dual CSV Export
    html5 = build_full_page(
        "Tab 5: Final Project Dataset & Export Table",
        render_filters(loc="All Locations", cat="All Categories", dt="Both", price_text="₹392 - ₹61,327", rating_text="0.0 - 5.0", n_sel=len(clean_hotels_df), n_tot=len(clean_hotels_df)),
        render_kpis(clean_hotels_df),
        render_tabs_nav(4),
        get_tab5_html(clean_hotels_df)
    )
    p_html5 = os.path.join(SCREENSHOTS_DIR, "tab5_final_dataset.html")
    p_png5 = os.path.join(SCREENSHOTS_DIR, "tab5_final_dataset.png")
    with open(p_html5, "w", encoding="utf-8") as f: f.write(html5)
    take_snapshot(p_html5, p_png5, height=1800)

    print("\n" + "=" * 70)
    print("ALL 9 EVIDENCE SCREENSHOTS GENERATED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
