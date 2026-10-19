"""
Generator script to build dashboard/hotel_dashboard.ipynb
Creates a clean, top-to-bottom executable Jupyter notebook using nbformat.
Includes all required charts, metrics, filters, and documentation.
With enhanced spacing, responsive margins, and zero-overlap subplot layouts.
"""

import nbformat as nbf
import json
import os

nb = nbf.v4.new_notebook()

# ---------------------------------------------------------
# Cell 0: Header & Project Overview (Markdown)
# ---------------------------------------------------------
cell0_md = """# Hotel Success Intelligence & Review 2 Integration Dashboard
## Business Analytics (23CSE452) Project

**Project Objective**: Predict hotel success from hotel characteristics, pricing, location, and amenities, utilizing real hotels scraped via SerpApi combined with synthetic hotels.
* **Target Definition**: "Success" is defined as `customer_rating >= 4.0` (binary classification proxy).
* **Review 2 Deliverable**: One integrated, interactive analytical dashboard synthesizing the entire team's work:
  1. **Exploratory Data Analysis (EDA)**: Dynamic distributions, pricing dynamics, amenity associations, and success drivers across 10 coordinated visualizations.
  2. **Predictive Modeling**: Tuned benchmarks for Logistic Regression, Decision Tree, and Random Forest (Champion Model) evaluated with dynamic test-set filtering.
  3. **Text Mining (Poojitha)**: Dynamic sentiment recomputation, customer pain points, experience themes, amenity aspect sentiment, dynamic word clouds, and searchable review explorer on real hotels.
  4. **Association Rule Mining (Anupama)**: Apriori algorithm on 33 standardized amenities discovering key amenity bundles (e.g., *Business Center + Pool → Spa*, Lift 6.190) with interactive parameter filtering and dynamic re-mining.
  5. **Final Dataset & Export**: Interactive, searchable, paged table displaying the final project dataset with dual CSV download capabilities (supporting Google Colab and local Jupyter).

> **Important Analytical & Methodological Disclaimers**:
> * **Association vs. Causation**: All findings, correlations, and feature importances describe statistical associations and co-occurrences in the data; they do not establish causal effects.
> * **Rating-Based Success Proxy**: Success is a proxy derived strictly from `customer_rating >= 4.0`. `customer_rating` is never used as a model predictor to prevent target leakage.
> * **Data Composition**: The dataset consists of 36.6% real scraped hotels (1,664 usable) and 63.4% synthetic hotels (2,887 usable). Synthetic coordinates sit on four fixed reference points, so latitude/longitude are excluded.
> * **Text Mining Coverage**: Text mining covers real hotels exclusively, as synthetic records do not possess customer review text.
"""

# ---------------------------------------------------------
# Cell 1: Setup & Configuration (Code)
# ---------------------------------------------------------
cell1_code = """# ==============================================================================
# CELL 1: CONFIGURATION, IMPORTS & STYLING
# Works seamlessly in both local Jupyter environments and Google Colab
# ==============================================================================
import os
import sys
import glob
import json
import warnings
warnings.filterwarnings("ignore")

# Detect execution environment and configure paths robustly
IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    from google.colab import output
    output.enable_custom_widget_manager()

curr_dir = os.path.abspath(os.getcwd())
if os.path.exists(os.path.join(curr_dir, "data", "combined_real_synthetic_hotels.csv")):
    BASE_DIR = curr_dir
elif os.path.exists(os.path.join(os.path.dirname(curr_dir), "data", "combined_real_synthetic_hotels.csv")):
    BASE_DIR = os.path.dirname(curr_dir)
elif os.path.exists("/content/hotel-dashboard/data/combined_real_synthetic_hotels.csv"):
    BASE_DIR = "/content/hotel-dashboard"
elif os.path.exists("/content/data/combined_real_synthetic_hotels.csv"):
    BASE_DIR = "/content"
else:
    BASE_DIR = curr_dir

DATA_DIR = os.path.join(BASE_DIR, "data")
TEXT_MINING_DIR = os.path.join(DATA_DIR, "text_mining_outputs")
APRIORI_DIR = os.path.join(DATA_DIR, "apriori")

# Centralized file paths configuration
HOTELS_CSV = os.path.join(DATA_DIR, "combined_real_synthetic_hotels.csv")
REVIEWS_CSV = os.path.join(TEXT_MINING_DIR, "review_level_sentiment.csv")
HOTEL_SENT_CSV = os.path.join(TEXT_MINING_DIR, "hotel_level_text_sentiment.csv")
PRESENTATION_JSON = os.path.join(TEXT_MINING_DIR, "presentation_numbers.json")
STRONG_RULES_CSV = os.path.join(APRIORI_DIR, "apriori_strong_rules.csv")
FREQUENT_ITEMSETS_CSV = os.path.join(APRIORI_DIR, "apriori_frequent_itemsets.csv")
AMENITY_SUPPORT_CSV = os.path.join(APRIORI_DIR, "apriori_amenity_support.csv")
TRANSACTION_MATRIX_CSV = os.path.join(APRIORI_DIR, "apriori_transaction_matrix.csv")

# Core Data Science and Modeling Libraries
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# Set explicit Plotly renderer to prevent duplicate (HTML + MIME) rendering in VS Code / Jupyter
if IN_COLAB:
    pio.renderers.default = "colab"
else:
    pio.renderers.default = "plotly_mimetype"

import ipywidgets as widgets
from IPython.display import display, HTML, Image, clear_output

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
from mlxtend.frequent_patterns import apriori, association_rules
import base64
import io
import time
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud

# Inject Custom Modern CSS Design System
display(HTML(\"\"\"
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .dashboard-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1e293b;
        background-color: #f8fafc;
        padding: 10px;
    }
    .header-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        color: #ffffff;
        padding: 24px 30px;
        border-radius: 14px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.15);
    }
    .header-title {
        font-size: 26px;
        font-weight: 700;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
        color: #ffffff;
    }
    .header-subtitle {
        font-size: 14px;
        font-weight: 400;
        color: #94a3b8;
        margin: 0;
    }
    .badge-pill {
        display: inline-block;
        background: rgba(99, 102, 241, 0.2);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.4);
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 8px;
        letter-spacing: 0.5px;
    }
    .notice-bar {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        color: #1e40af;
        padding: 12px 16px;
        border-radius: 8px;
        font-size: 13px;
        line-height: 1.5;
        margin-bottom: 20px;
    }
    .filter-panel-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 20px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .filter-title {
        font-size: 15px;
        font-weight: 600;
        color: #334155;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 20px;
        min-width: 170px;
        flex: 1;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        border-top: 4px solid #6366f1;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.06);
    }
    .kpi-label {
        font-size: 12px;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 24px;
        font-weight: 700;
        color: #0f172a;
    }
    .kpi-sub {
        font-size: 11px;
        color: #94a3b8;
        margin-top: 2px;
    }
    .tab-note-banner {
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 16px;
        font-size: 12px;
        color: #475569;
    }
    .warning-badge {
        background: #fffbeb;
        border: 1px solid #fde68a;
        color: #b45309;
        padding: 10px 14px;
        border-radius: 8px;
        font-size: 13px;
        margin-bottom: 14px;
        font-weight: 500;
    }
    .model-champion-card {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 1px solid #86efac;
        color: #166534;
        padding: 16px 20px;
        border-radius: 12px;
        margin-bottom: 18px;
    }
    .empty-state-box {
        text-align: center;
        padding: 40px 20px;
        background: #ffffff;
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        color: #64748b;
        margin: 20px 0;
    }
    
    /* Enhanced Tab Bar Styling: Perfectly Centered, Uniform Auto-Sizing, No Overflow */
    .widget-tab,
    .jupyter-widgets.widget-tab,
    .jupyter-widget-tab {
        width: 100% !important;
    }
    .widget-tab .widget-tab-bar,
    .widget-tab .p-TabBar,
    .widget-tab .lm-TabBar,
    .jupyter-widgets .p-TabBar,
    .jupyter-widgets .lm-TabBar {
        width: 100% !important;
        border-bottom: 2px solid #e2e8f0 !important;
    }
    .widget-tab .p-TabBar-content,
    .widget-tab .lm-TabBar-content,
    .jupyter-widgets .p-TabBar-content,
    .jupyter-widgets .lm-TabBar-content {
        display: flex !important;
        flex-direction: row !important;
        width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
        list-style: none !important;
    }
    .widget-tab .p-TabBar-tab,
    .widget-tab .lm-TabBar-tab,
    .jupyter-widgets .p-TabBar-tab,
    .jupyter-widgets .lm-TabBar-tab {
        flex: 1 1 0px !important;
        min-width: 140px !important;
        max-width: none !important;
        width: auto !important;
        height: 44px !important;
        line-height: 44px !important;
        padding: 0 16px !important;
        font-size: 13.5px !important;
        font-weight: 600 !important;
        color: #475569 !important;
        background: #f1f5f9 !important;
        border: 1px solid #cbd5e1 !important;
        border-bottom: none !important;
        border-radius: 8px 8px 0 0 !important;
        margin-right: 4px !important;
        box-sizing: border-box !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        cursor: pointer !important;
        overflow: visible !important;
    }
    .widget-tab .p-TabBar-tab.p-mod-current,
    .widget-tab .lm-TabBar-tab.lm-mod-current,
    .jupyter-widgets .p-TabBar-tab.p-mod-current,
    .jupyter-widgets .lm-TabBar-tab.lm-mod-current {
        background: #ffffff !important;
        color: #1e293b !important;
        border-top: 3px solid #3b82f6 !important;
        border-bottom: 2px solid #ffffff !important;
        margin-bottom: -2px !important;
        font-weight: 700 !important;
    }
    .widget-tab .p-TabBar-tabLabel,
    .widget-tab .lm-TabBar-tabLabel,
    .jupyter-widgets .p-TabBar-tabLabel,
    .jupyter-widgets .lm-TabBar-tabLabel {
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        width: 100% !important;
        margin: 0 auto !important;
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }
    .widget-tab .p-TabBar-tabIcon,
    .widget-tab .lm-TabBar-tabIcon,
    .jupyter-widgets .p-TabBar-tabIcon,
    .jupyter-widgets .lm-TabBar-tabIcon {
        display: none !important;
    }

    /* Final Dataset & Review Explorer Table Styles */
    .dataset-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 12.5px;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        overflow: hidden;
        margin-bottom: 12px;
    }
    .dataset-table th {
        background: #f1f5f9;
        color: #334155;
        font-weight: 600;
        padding: 9px 12px;
        text-align: left;
        border-bottom: 2px solid #cbd5e1;
    }
    .dataset-table td {
        padding: 8px 12px;
        border-bottom: 1px solid #f1f5f9;
        color: #1e293b;
        vertical-align: middle;
    }
    .dataset-table tr:hover {
        background: #f8fafc;
    }
    .status-badge-success {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        background: #ecfdf5;
        color: #059669;
        border: 1px solid #a7f3d0;
    }
    .status-badge-unsuccess {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        background: #f1f5f9;
        color: #64748b;
        border: 1px solid #cbd5e1;
    }
    .badge-vader-pos {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        background: #ecfdf5;
        color: #059669;
        border: 1px solid #a7f3d0;
    }
    .badge-vader-neg {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        background: #fef2f2;
        color: #dc2626;
        border: 1px solid #fecaca;
    }
    .badge-vader-neu {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        background: #f8fafc;
        color: #64748b;
        border: 1px solid #e2e8f0;
    }

</style>
\"\"\"))

print("Configuration ready. Environment:", "Google Colab" if IN_COLAB else "Jupyter Notebook")
"""

# ---------------------------------------------------------
# Cell 2: Data Ingestion & Preprocessing (Code)
# ---------------------------------------------------------
cell2_code = """# ==============================================================================
# CELL 2: DATA INGESTION, TARGET RECOMPUTATION & CLEANING
# Follows reference/BA_total_Review1_Complete_CORRECTED.ipynb exactly
# ==============================================================================

# 1. Load Primary Hotels Dataset
raw_hotels_df = pd.read_csv(HOTELS_CSV)

# Convert customer_rating to numeric and drop rows with missing ratings (target source)
df_hotels = raw_hotels_df.copy()
df_hotels["customer_rating"] = pd.to_numeric(df_hotels["customer_rating"], errors="coerce")
dropped_missing_rating = int(df_hotels["customer_rating"].isna().sum())
df_hotels = df_hotels.dropna(subset=["customer_rating"]).copy()

# Recalculate success target for ALL rows: success = 1 iff customer_rating >= 4.0
df_hotels["success"] = (df_hotels["customer_rating"] >= 4.0).astype(int)

# Deduplicate by hotel_id, keeping first occurrence
clean_hotels_df = df_hotels.drop_duplicates().drop_duplicates(subset="hotel_id", keep="first").copy()

# Median imputation for numerical features
num_impute_cols = ["price", "review_count", "amenity_count"]
for col in num_impute_cols:
    clean_hotels_df[col] = pd.to_numeric(clean_hotels_df[col], errors="coerce")
    clean_hotels_df[col] = clean_hotels_df[col].fillna(clean_hotels_df[col].median())

# Categorical missing values filled with 'Unknown'
for col in ["location", "hotel_category"]:
    clean_hotels_df[col] = clean_hotels_df[col].fillna("Unknown")

# 15 individual amenity indicator columns
amenity_cols = [c for c in clean_hotels_df.columns if c.startswith("amenity_") and c != "amenity_count"]
for c in amenity_cols:
    clean_hotels_df[c] = pd.to_numeric(clean_hotels_df[c], errors="coerce").fillna(0).astype(int)

# Feature engineering: log_price = log(1 + price)
clean_hotels_df["log_price"] = np.log1p(clean_hotels_df["price"])

# Display formatting helpers (for UI only; raw values used for ML models)
clean_hotels_df["display_rating"] = clean_hotels_df["customer_rating"].round(1)
clean_hotels_df["display_review_count"] = clean_hotels_df["review_count"].round(0).astype(int)

# Summary verification
total_hotels = len(clean_hotels_df)
successful_hotels = int(clean_hotels_df["success"].sum())
unsuccessful_hotels = total_hotels - successful_hotels
success_rate_overall = (successful_hotels / total_hotels) * 100

print(f"Dataset Verified: {total_hotels:,} hotels (Dropped {dropped_missing_rating} missing rating).")
print(f"Success Target: {successful_hotels:,} Successful ({success_rate_overall:.1f}%), {unsuccessful_hotels:,} Not Successful.")

# 2. Load Teammate Outputs from Stage A
# Text Mining datasets
rev_sentiment_df = pd.read_csv(REVIEWS_CSV)
hotel_sentiment_df = pd.read_csv(HOTEL_SENT_CSV)
with open(PRESENTATION_JSON, "r", encoding="utf-8") as f:
    text_numbers = json.load(f)

# Apriori datasets
apriori_rules_df = pd.read_csv(STRONG_RULES_CSV)
apriori_itemsets_df = pd.read_csv(FREQUENT_ITEMSETS_CSV)
apriori_support_df = pd.read_csv(AMENITY_SUPPORT_CSV)
apriori_trans_df = pd.read_csv(TRANSACTION_MATRIX_CSV)

print(f"Loaded Text Mining: {len(rev_sentiment_df):,} reviews, {len(hotel_sentiment_df):,} hotel sentiment summaries.")
print(f"Loaded Apriori: {len(apriori_rules_df):,} strong rules, {len(apriori_itemsets_df):,} itemsets, {len(apriori_support_df):,} amenities.")

# ==============================================================================
# PRECOMPUTE TEXT MINING STRUCTURES FOR SUB-SECOND DYNAMIC AGGREGATION
# Reuses EXACT regular expressions, keyword rules, and stopword lists from reference/review2_updated.ipynb
# ==============================================================================
t_tm_start = time.time()

PAIN_PATTERNS = {
    "Cleanliness / hygiene":       r"\bdirty\b|\bunclean\b|\bfilth\w*|\bnot (very |so |that |at all |too |properly )?(clean|hygien\w*)\b|\bunhygien\w*|\bstink\w*|\bsmell\w*|\bcockroach\w*|\bbed ?bugs?\b|\bstain\w*|\bmou?ld\b|\bdust\w*|\bno cleanliness\b|\bpoor (cleanliness|hygiene)\b",
    "Rude / unhelpful staff":      r"\brude\w*|\bimpolite\b|\bunprofessional\b|\bmisbehav\w*|\barrogan\w*|\bunhelpful\b|\bnot (very )?(helpful|cooperative|friendly|polite|responsive|supportive)\b|\bpoor (staff|behaviou?r|attitude)\b|\bbad (staff|behaviou?r|attitude)\b|\battitude\b",
    "Slow / poor service":         r"\bslow\w*|\bdelay\w*|\bwaited\b|\bwaiting\b|\btook (a )?(long|too long|forever|hours?)\b|\bpoor service\b|\bbad service\b|\bworst service\b|\bno service\b|\bno response\b|\bnot responding\b",
    "Broken / poorly maintained":  r"\bnot working\b|\bbroken\b|\bdamaged\b|\bpoor(ly)? maint\w*|\bnot maintained\b|\bleak\w*|\bworn\b|\bfaulty\b|\b(does|did|do)( not|n['’]?t) work\b|\bnot properly working\b|\bout of order\b",
    "AC / cooling problems":       r"\b(ac|a/c|aircon\w*|air[- ]?condition\w*)\b[^.]{0,40}\b(not|no|never|poor|bad|issue|problem|broken|noisy|leak\w*|worst)\b|\b(not|no|poor|bad)\b[^.]{0,20}\b(ac|a/c|cooling)\b",
    "Wi-Fi / internet problems":   r"\b(wi-?fi|wifi|internet|network)\b[^.]{0,40}\b(not|no|poor|bad|slow|weak|issue|problem|never|unavailable)\b|\b(no|poor|bad|slow|weak|not)\b[^.]{0,20}\b(wi-?fi|wifi|internet)\b",
    "Noise":                       r"\bnois\w+|\bloud\b|\bdisturb\w*",
    "Bathroom / hot-water issues":  r"\bno hot water\b|\bhot water\b[^.]{0,30}\b(not|no|never|issue|problem)\b|\b(bathrooms?|toilets?|washrooms?)\b[^.]{0,40}\b(dirty|smell\w*|small|broken|not|no|leak\w*|bad|poor|worst|unclean)\b",
    "Small / cramped rooms":       r"\bsmall\b|\btiny\b|\bcramped\b|\bcongested\b|\bnarrow\b|\bnot spacious\b|\bsuffocat\w*",
    "Food / breakfast quality":    r"\b(food|breakfast|dinner|lunch|meals?|buffet|tea|coffee)\b[^.]{0,40}\b(bad|poor|worst|not|cold|tasteless|stale|average|terrible|horrible|awful|disappoint\w*|limited)\b|\b(bad|poor|worst|tasteless|stale|terrible|horrible|awful)\b[^.]{0,20}\b(food|breakfast|dinner|lunch|meals?)\b",
    "Price / value for money":     r"\boverpriced\b|\bexpensive\b|\bcostly\b|\bover ?charg\w*|\bwaste of money\b|\bnot worth\b|\bhigh (price|rate|tariff|charges?)\b|\bextra (charge|money|cost)\w*|\brip ?off\b|\bhidden (charge|cost)\w*|\bcharged\b",
    "Booking / refund / check-in":  r"\brefund\w*|\bcancel\w*|\bdeposit\b|\breservation\b|\bcheck ?-?in\b[^.]{0,40}\b(late|delay\w*|problem|issue|refus\w*|denied|not|wait\w*)\b",
    "Parking problems":            r"\bparking\b[^.]{0,40}\b(not|no|problem|issue|difficult|limited|hard|bad)\b|\b(no|not|poor|limited)\b[^.]{0,20}\bparking\b",
    "Bed / bedding discomfort":    r"\b(beds?|mattress\w*|pillows?\w*|bed ?sheets?|blankets?|linen)\b[^.]{0,40}\b(dirty|hard|uncomfortable|stain\w*|smell\w*|bad|poor|old|torn|not|worst|bugs?)\b"
}

THEME_PATTERNS = {
    "Room":             r"\brooms?\b",
    "Staff":            r"\bstaffs?\b|\bemployees?\b|\breception\w*|\bmanager\b|\bcaretaker\b|\bhousekeeping\b|\bwaiters?\b",
    "Service":          r"\bservices?\b",
    "Cleanliness":      r"\bclean\w*|\bdirty\b|\bdirt\b|\bhygien\w*|\btidy\b|\bspotless\b|\bfilth\w*|\bunclean\b|\bsmell\w*|\bstink\w*|\bneat\b",
    "Food":             r"\bfood\b|\bdinner\b|\blunch\b|\bmeals?\b|\bbuffet\b|\bmenu\b|\brestaurant\b",
    "Breakfast":        r"\bbreakfast\b",
    "Location":         r"\blocation\b|\blocated\b|\bnearby\b|\bnear\b|\bclose to\b|\bcentral\b",
    "Wi-Fi":            r"\bwi-?fi\b|\bwifi\b|\binternet\b",
    "Parking":          r"\bparking\b",
    "Bathroom":         r"\bbathrooms?\b|\btoilets?\b|\bwashrooms?\b|\bshower\b|\bgeyser\b|\bhot water\b",
    "Bed / bedding":    r"\bbeds?\b|\bmattress\w*|\bpillows?\b|\bbed ?sheets?\b|\bblankets?\b|\blinen\b",
    "Price / value":    r"\bprice\w*|\bcosts?\b|\bexpensive\b|\bcheap\b|\bvalue\b|\bmoney\b|\bworth\b|\boverpriced\b|\baffordable\b|\bbudget\b|\bpocket\b",
    "Facilities":       r"\bfacilit\w+|\bamenit\w+",
    "Air conditioning": r"\bac\b|\ba/c\b|\baircon\w*|\bair[- ]?condition\w*|\bcooling\b",
    "Noise":            r"\bnois\w+|\bloud\b",
    "View":             r"\bviews?\b"
}

AMENITY_PATTERNS = {
    "Breakfast":        r"\bbreakfast\b",
    "Air conditioning": r"\bac\b|\ba/c\b|\baircon\w*|\bair[- ]?condition\w*|\bcooling\b",
    "Restaurant":       r"\brestaurant\b",
    "Room service":     r"\broom service\b",
    "Parking":          r"\bparking\b",
    "Wi-Fi":            r"\bwi-?fi\b|\bwifi\b|\binternet\b",
    "Accessibility":    r"\bwheelchair\b|\baccessib\w*|\belevator\b|\blift\b",
    "Pool":             r"\bpool\b|\bswimming\b",
    "Kid-friendly":     r"\bkids?\b|\bchild\w*",
    "Bar":              r"\bbar\b|\bpub\b",
    "Spa":              r"\bspa\b|\bmassage\b",
    "Pet-friendly":     r"\bpets?\b|\bdogs?\b",
    "Laundry":          r"\blaundry\b",
    "Fitness / gym":    r"\bgym\b|\bfitness\b",
    "Airport shuttle":  r"\bshuttle\b|\bairport (pick|transfer|drop)\w*|\bpick ?up\b"
}

NEGATIONS  = {"not", "no", "nor", "never", "neither"}
STOP       = set(stopwords.words("english")) - NEGATIONS
KEEP_SHORT = NEGATIONS | {"ac", "tv", "ok"}
FILLER     = {"make", "also", "book", "get", "got", "go", "would", "could", "even", "really", "one", "u", "us", "much", "lot", "like", "well", "overall", "take", "come", "say", "see", "give", "back"}
GENERIC    = {"hotel", "stay", "place", "one", "get", "got", "go", "went", "also", "would", "could", "even", "really", "u", "us", "much", "lot", "like", "take", "make", "come", "say", "see", "give", "back", "day", "night"}

tokenizer  = RegexpTokenizer(r"[a-z]+")
lemmatizer = WordNetLemmatizer()

def expand_contractions(t):
    t = re.sub(r"\b(dont|doesnt|didnt|isnt|arent|wasnt|werent|hasnt|havent|hadnt|wouldnt|couldnt|shouldnt|cant|wont)\b", " not", t)
    t = re.sub(r"\bwon['’]?t\b", "will not", t)
    t = re.sub(r"\bcan['’]?t\b", "can not", t)
    t = re.sub(r"n['’]?t\b", " not", t)
    return t

def lemma(tok):
    return lemmatizer.lemmatize(lemmatizer.lemmatize(tok, "v"), "n")

def to_tokens(text):
    t = text.lower()
    t = expand_contractions(t)
    t = re.sub(r"[^a-z\s]", " ", t)
    toks = tokenizer.tokenize(t)
    toks = [w for w in toks if w not in STOP]
    toks = [lemma(w) for w in toks]
    toks = [w for w in toks if w not in STOP]
    toks = [w for w in toks if len(w) >= 3 or w in KEEP_SHORT]
    return toks

# Precompute boolean arrays
rev_text_lc = rev_sentiment_df["text_light"].fillna("").str.lower()
pain_matrix = {k: rev_text_lc.str.contains(pat, regex=True).fillna(False).values for k, pat in PAIN_PATTERNS.items()}
theme_matrix = {k: rev_text_lc.str.contains(pat, regex=True).fillna(False).values for k, pat in THEME_PATTERNS.items()}

# Fast amenity aspect precomputation
any_amen_pat = re.compile("|".join(f"({p})" for p in AMENITY_PATTERNS.values()), flags=re.IGNORECASE)
sia = SentimentIntensityAnalyzer()
sent_rows = []
for idx, txt in rev_sentiment_df["text_light"].fillna("").items():
    if any_amen_pat.search(txt):
        for s in re.split(r"(?<=[.!?])\s+", txt):
            s_str = s.strip()
            if s_str and any_amen_pat.search(s_str):
                sent_rows.append((idx, s_str.lower(), sia.polarity_scores(s_str)["compound"]))
sent_df = pd.DataFrame(sent_rows, columns=["rev", "sent_lc", "compound"])

amenity_matrix = {}
for name, mention_pat in AMENITY_PATTERNS.items():
    m = sent_df["sent_lc"].str.contains(mention_pat, regex=True)
    per_rev = sent_df[m].groupby("rev")["compound"].mean()
    labels = np.zeros(len(rev_sentiment_df), dtype=np.int8)
    for r_idx, comp in per_rev.items():
        if comp >= 0.05: labels[r_idx] = 1
        elif comp <= -0.05: labels[r_idx] = 3
        else: labels[r_idx] = 2
    amenity_matrix[name] = labels

# Token lists
tokens_list = [to_tokens(t) for t in rev_sentiment_df["text_light"].fillna("")]
bigrams_list = [[f"{toks[k]} {toks[k+1]}" for k in range(len(toks)-1)] for toks in tokens_list]

print(f"Precomputed Text Mining Engine in {time.time()-t_tm_start:.2f}s: {len(pain_matrix)} pain points, {len(theme_matrix)} themes, {len(amenity_matrix)} amenities.")
"""

# ---------------------------------------------------------
# Cell 3: Predictive Models Setup & Benchmarking (Code)
# ---------------------------------------------------------
cell3_code = """# ==============================================================================
# CELL 3: MACHINE LEARNING PIPELINE & MODEL BENCHMARKING
# Rebuilds the three tuned models from Review 1 (21 predictors, stratified 80/20)
# ==============================================================================

# Model Predictors (21 total: 19 numeric including 15 amenities, 2 categorical)
# customer_rating is strictly EXCLUDED as it defines the target
numeric_features = ["price", "log_price", "review_count", "amenity_count"] + amenity_cols
categorical_features = ["location", "hotel_category"]

for forbidden in ["success", "customer_rating", "hotel_id", "hotel_name", "amenities", "display_rating", "display_review_count"]:
    assert forbidden not in numeric_features and forbidden not in categorical_features, f"Target leakage check failed: {forbidden}"

X_all = clean_hotels_df[numeric_features + categorical_features].copy()
y_all = clean_hotels_df["success"].copy()

# Stratified 80/20 Split with random_state=42
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all,
    test_size=0.20,
    random_state=42,
    stratify=y_all
)

# Benchmark Preprocessing Pipeline
num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

ml_preprocessor = ColumnTransformer([
    ("num", num_transformer, numeric_features),
    ("cat", cat_transformer, categorical_features)
])

# Build Tuned Model Pipelines (matching Review 1 tuned hyperparameters)
# 1. Logistic Regression: C=10, class_weight='balanced', max_iter=2000
model_lr = Pipeline([
    ("preprocessor", ml_preprocessor),
    ("classifier", LogisticRegression(C=10, class_weight="balanced", max_iter=2000, random_state=42))
])

# 2. Decision Tree: max_depth=7, min_samples_split=10, class_weight=None
model_dt = Pipeline([
    ("preprocessor", ml_preprocessor),
    ("classifier", DecisionTreeClassifier(max_depth=7, min_samples_split=10, random_state=42))
])

# 3. Random Forest (Champion): 300 trees, max_depth=10, min_samples_split=5, class_weight='balanced'
model_rf = Pipeline([
    ("preprocessor", ml_preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_split=5, min_samples_leaf=1,
        class_weight="balanced", random_state=42, n_jobs=-1
    ))
])

# Fit all three models on training set
model_lr.fit(X_train, y_train)
model_dt.fit(X_train, y_train)
model_rf.fit(X_train, y_train)

# Store test predictions and probabilities
test_preds = {
    "Logistic Regression": model_lr.predict(X_test),
    "Decision Tree": model_dt.predict(X_test),
    "Random Forest": model_rf.predict(X_test)
}

test_probs = {
    "Logistic Regression": model_lr.predict_proba(X_test)[:, 1],
    "Decision Tree": model_dt.predict_proba(X_test)[:, 1],
    "Random Forest": model_rf.predict_proba(X_test)[:, 1]
}

# Precomputed Review 1 Benchmark Figures (for baseline reference)
REVIEW1_BENCHMARKS = {
    "Random Forest": {
        "Accuracy": 0.8134, "Balanced Accuracy": 0.6443,
        "Macro Precision": 0.6777, "Macro Recall": 0.6443,
        "Macro F1": 0.6571, "Weighted F1": 0.8037, "ROC-AUC": 0.7072,
        "Confusion Matrix": [[63, 104], [66, 678]]
    },
    "Decision Tree": {
        "Accuracy": 0.8035, "Balanced Accuracy": 0.5662,
        "Macro Precision": 0.6297, "Macro Recall": 0.5662,
        "Macro F1": 0.5750, "Weighted F1": 0.7724, "ROC-AUC": 0.6344,
        "Confusion Matrix": [[32, 135], [44, 700]]
    },
    "Logistic Regression": {
        "Accuracy": 0.6894, "Balanced Accuracy": 0.6612,
        "Macro Precision": 0.6056, "Macro Recall": 0.6612,
        "Macro F1": 0.6045, "Weighted F1": 0.7205, "ROC-AUC": 0.6955,
        "Confusion Matrix": [[103, 64], [219, 525]]
    }
}

# Extract Random Forest Top 15 Feature Importances
rf_feature_names = model_rf.named_steps["preprocessor"].get_feature_names_out()
rf_importances = model_rf.named_steps["classifier"].feature_importances_
clean_feature_names = [f.replace("num__", "").replace("cat__", "").replace("hotel_category_", "").replace("location_", "") for f in rf_feature_names]

rf_feat_df = pd.DataFrame({
    "Feature": clean_feature_names,
    "Importance": rf_importances
}).sort_values("Importance", ascending=False).head(15).reset_index(drop=True)

print("Tuned Model Pipelines Fitted on 3,640 Training Hotels.")
print("Test Set Ready: 911 Hotels (167 Not Successful, 744 Successful).")
"""

# ---------------------------------------------------------
# Cell 4: Dashboard UI & Event Handlers (Code)
# ---------------------------------------------------------
cell4_code = """# ==============================================================================
# CELL 4: INTERACTIVE DASHBOARD UI, VISUALIZATIONS & EVENT HANDLERS
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# Assembles filter panel, KPI row, and 4 dedicated tabs with zero-overlap layout
# ==============================================================================

# Output Widgets
out_kpi = widgets.Output()
out_tab1 = widgets.Output()
out_tab2 = widgets.Output()
out_tab3 = widgets.Output()
out_tab4 = widgets.Output()
out_tab5 = widgets.Output()

# --- FILTER WIDGETS ---
unique_locations = sorted([loc for loc in clean_hotels_df["location"].unique() if loc != "Unknown"])
loc_options = ["All Locations"] + unique_locations
w_location = widgets.SelectMultiple(
    options=loc_options,
    value=["All Locations"],
    description="Location:",
    rows=5,
    layout=widgets.Layout(width="220px")
)

unique_categories = ["All Categories", "5-star hotel", "4-star hotel", "3-star hotel", "2-star hotel", "1-star hotel", "Unknown"]
w_category = widgets.SelectMultiple(
    options=unique_categories,
    value=["All Categories"],
    description="Category:",
    rows=5,
    layout=widgets.Layout(width="220px")
)

min_price = float(clean_hotels_df["price"].min())
max_price = float(clean_hotels_df["price"].max()) # Full span to include 100.0% of corpus by default
w_price = widgets.FloatRangeSlider(
    value=[min_price, max_price],
    min=min_price,
    max=max_price,
    step=250.0,
    description="Price (₹):",
    readout_format=",.0f",
    layout=widgets.Layout(width="300px")
)

w_rating = widgets.FloatRangeSlider(
    value=[0.0, 5.0],
    min=0.0,
    max=5.0,
    step=0.1,
    description="Rating:",
    readout_format=".1f",
    layout=widgets.Layout(width="300px")
)

w_datatype = widgets.Dropdown(
    options=["Both", "Real", "Synthetic"],
    value="Both",
    description="Data Type:",
    layout=widgets.Layout(width="220px")
)

btn_reset = widgets.Button(
    description="Reset Filters",
    button_style="warning",
    icon="refresh",
    layout=widgets.Layout(width="140px", height="38px", margin="10px 0 0 10px")
)

lbl_filter_summary = widgets.HTML(
    value=f"<b>Selected Hotels:</b> {len(clean_hotels_df):,} / {len(clean_hotels_df):,} (100.0%)"
)

# --- ASSOCIATION RULES FILTER WIDGETS (TAB 4) ---
w_apriori_supp = widgets.FloatSlider(value=0.05, min=0.01, max=0.20, step=0.01, description="Min Support:", readout_format=".2f", layout=widgets.Layout(width="240px"))
w_apriori_conf = widgets.FloatSlider(value=0.60, min=0.30, max=0.95, step=0.05, description="Min Conf:", readout_format=".2f", layout=widgets.Layout(width="240px"))
w_apriori_lift = widgets.FloatSlider(value=2.0, min=1.0, max=6.5, step=0.2, description="Min Lift:", readout_format=".1f", layout=widgets.Layout(width="240px"))
w_apriori_search = widgets.Text(value="", placeholder="Filter by amenity (e.g., Spa)", description="Search Amenity:", layout=widgets.Layout(width="260px"))
w_apriori_recompute = widgets.Checkbox(value=False, description="Recompute on Filtered Hotels (mlxtend)", layout=widgets.Layout(width="320px"))

apriori_controls_box = widgets.VBox([
    widgets.HTML("<div style='font-size:14px; font-weight:600; color:#334155; margin:4px 0 8px 0;'>Association Rules Filtering & Controls</div>"),
    widgets.HBox([w_apriori_supp, w_apriori_conf, w_apriori_lift], layout=widgets.Layout(grid_gap="15px", margin="0 0 8px 0")),
    widgets.HBox([w_apriori_search, w_apriori_recompute], layout=widgets.Layout(grid_gap="20px", align_items="center"))
], layout=widgets.Layout(border="1px solid #e2e8f0", padding="14px 18px", border_radius="10px", background_color="#f8fafc", margin="0 0 16px 0"))

# Tab 4 Container integrates controls directly inside Tab 4
tab4_container = widgets.VBox([apriori_controls_box, out_tab4])

# Plotly display helper: disables distracting floating modebar to keep titles pristine
PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}

# --- REVIEW EXPLORER WIDGETS (TAB 3 TASK 4) ---
w_tm_search = widgets.Text(
    value="",
    placeholder="Search review excerpt or hotel...",
    description="Search Text:",
    layout=widgets.Layout(width="280px")
)
tm_page_state = {"page": 0}
btn_tm_prev = widgets.Button(description="◀ Prev", layout=widgets.Layout(width="80px", height="32px"))
btn_tm_next = widgets.Button(description="Next ▶", layout=widgets.Layout(width="80px", height="32px"))
lbl_tm_page = widgets.HTML(layout=widgets.Layout(margin="5px 12px 0 12px"))
out_tm_reviews = widgets.Output()

# --- FINAL DATASET WIDGETS (TAB 5 TASK 2) ---
w_tab5_search = widgets.Text(
    value="",
    placeholder="Search hotel by name...",
    description="Search Hotel:",
    layout=widgets.Layout(width="280px")
)
tab5_page_state = {"page": 0}
btn_tab5_prev = widgets.Button(description="◀ Previous", layout=widgets.Layout(width="95px", height="34px"))
btn_tab5_next = widgets.Button(description="Next ▶", layout=widgets.Layout(width="95px", height="34px"))
lbl_tab5_page = widgets.HTML(layout=widgets.Layout(margin="6px 14px 0 14px"))
btn_download_filt = widgets.Button(description="Download Filtered Data (CSV)", icon="download", layout=widgets.Layout(height="34px"))
btn_download_full = widgets.Button(description="Download Full Final Dataset (CSV)", icon="download", layout=widgets.Layout(height="34px"))
out_download = widgets.Output()

# --- HELPER: APPLY GLOBAL FILTERS ---
def get_filtered_hotels():
    f_df = clean_hotels_df.copy()
    
    # Location
    selected_locs = list(w_location.value)
    if "All Locations" not in selected_locs and len(selected_locs) > 0:
        f_df = f_df[f_df["location"].isin(selected_locs)]
        
    # Category
    selected_cats = list(w_category.value)
    if "All Categories" not in selected_cats and len(selected_cats) > 0:
        f_df = f_df[f_df["hotel_category"].isin(selected_cats)]
        
    # Price
    p_low, p_high = w_price.value
    f_df = f_df[(f_df["price"] >= p_low) & (f_df["price"] <= p_high)]
    
    # Rating
    r_low, r_high = w_rating.value
    f_df = f_df[(f_df["customer_rating"] >= r_low) & (f_df["customer_rating"] <= r_high)]
    
    # Data Type
    dt = w_datatype.value
    if dt != "Both":
        f_df = f_df[f_df["data_type"] == dt]
        
    return f_df

# --- RENDER KPI CARDS ---
def render_kpis(f_df):
    out_kpi.clear_output(wait=True)
    with out_kpi:
        n = len(f_df)
        if n == 0:
            display(HTML(\"\"\"
            <div class="empty-state-box">
                <h3>No Hotels Match Filter Criteria</h3>
                <p>Please widen your price, rating, category, or location filter selection.</p>
            </div>
            \"\"\"))
            return
            
        succ_rate = (f_df["success"].mean() * 100)
        avg_price = f_df["price"].mean()
        avg_rating = f_df["customer_rating"].mean()
        avg_amenities = f_df["amenity_count"].mean()
        pct_of_all = (n / len(clean_hotels_df)) * 100
        
        kpi_html = f\"\"\"
        <div style="display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 20px;">
            <div class="kpi-card" style="border-top-color: #6366f1;">
                <div class="kpi-label">Hotels Selected</div>
                <div class="kpi-value">{n:,}</div>
                <div class="kpi-sub">{pct_of_all:.1f}% of total corpus</div>
            </div>
            <div class="kpi-card" style="border-top-color: #10b981;">
                <div class="kpi-label">Success Rate (≥ 4.0)</div>
                <div class="kpi-value">{succ_rate:.1f}%</div>
                <div class="kpi-sub">{int(f_df['success'].sum()):,} successful hotels</div>
            </div>
            <div class="kpi-card" style="border-top-color: #f59e0b;">
                <div class="kpi-label">Average Price</div>
                <div class="kpi-value">₹{avg_price:,.0f}</div>
                <div class="kpi-sub">Median: ₹{f_df['price'].median():,.0f}</div>
            </div>
            <div class="kpi-card" style="border-top-color: #06b6d4;">
                <div class="kpi-label">Average Rating</div>
                <div class="kpi-value">{avg_rating:.2f} <span style="font-size:14px; font-weight:normal; color:#64748b;">/ 5.0</span></div>
                <div class="kpi-sub">Range: {f_df['customer_rating'].min():.1f} – {f_df['customer_rating'].max():.1f}</div>
            </div>
            <div class="kpi-card" style="border-top-color: #8b5cf6;">
                <div class="kpi-label">Avg Amenities</div>
                <div class="kpi-value">{avg_amenities:.1f}</div>
                <div class="kpi-sub">Across 15 monitored amenities</div>
            </div>
        </div>
        \"\"\"
        display(HTML(kpi_html))

# --- RENDER TAB 1: EDA ---
def render_tab1(f_df):
    out_tab1.clear_output(wait=True)
    with out_tab1:
        if len(f_df) == 0:
            display(HTML("<div class='empty-state-box'>No data available for current filter selection.</div>"))
            return
            
        display(HTML(\"\"\"
        <div class="tab-note-banner">
            <b>Filter Applicability:</b> All 10 charts on this tab respond dynamically to the Global Filters above.<br>
            <i>Notice: Amenity availability differences, correlations, and distributions represent statistical associations, NOT direct causes.</i>
        </div>
        \"\"\"))
        
        # Row 1: Success Distribution & Real vs Synthetic
        fig_r1 = make_subplots(
            rows=1, cols=2,
            horizontal_spacing=0.16,
            subplot_titles=("Success Distribution (Rating ≥ 4.0)", "Real vs. Synthetic Hotels"),
            specs=[[{"type": "domain"}, {"type": "xy"}]]
        )
        
        # Donut for success
        succ_counts = f_df["success"].value_counts().rename({1: "Successful (≥ 4.0)", 0: "Not Successful (< 4.0)"})
        fig_r1.add_trace(go.Pie(
            labels=succ_counts.index, values=succ_counts.values,
            hole=0.55, marker=dict(colors=["#10b981", "#ef4444"])
        ), row=1, col=1)
        
        # Bar for real vs synthetic
        dtype_grp = f_df.groupby("data_type").agg(Count=("hotel_id", "count"), Success_Rate=("success", lambda s: s.mean() * 100)).reset_index()
        fig_r1.add_trace(go.Bar(
            x=dtype_grp["data_type"], y=dtype_grp["Count"],
            name="Hotel Count", marker_color="#3b82f6", text=dtype_grp["Count"], textposition="outside"
        ), row=1, col=2)
        fig_r1.for_each_annotation(lambda a: a.update(font=dict(size=12, color="#1e293b", family="Inter, sans-serif")))
        fig_r1.update_layout(height=350, margin=dict(l=25, r=25, t=45, b=25), template="plotly_white")
        display(fig_r1)
        
        # Row 2: Customer Rating Distribution & Category Success
        fig_r2 = make_subplots(
            rows=1, cols=2,
            horizontal_spacing=0.16,
            subplot_titles=("Customer Rating Distribution", "Success Rate by Star Category (%)")
        )
        
        for dt, col in [("Real", "#3b82f6"), ("Synthetic", "#8b5cf6")]:
            sub_d = f_df[f_df["data_type"] == dt]
            if len(sub_d) > 0:
                fig_r2.add_trace(go.Histogram(
                    x=sub_d["customer_rating"], name=dt, marker_color=col, opacity=0.6,
                    xbins=dict(start=1.0, end=5.0, size=0.1)
                ), row=1, col=1)
        fig_r2.add_vline(x=4.0, line_dash="dash", line_color="#ef4444", annotation_text="Success (4.0)", row=1, col=1)
        fig_r2.update_layout(barmode="overlay")
        
        cat_grp = f_df.groupby("hotel_category").agg(Rate=("success", lambda s: s.mean()*100), Count=("hotel_id", "count")).reset_index()
        cat_grp = cat_grp.sort_values("Rate", ascending=True)
        fig_r2.add_trace(go.Bar(
            y=cat_grp["hotel_category"], x=cat_grp["Rate"], orientation="h",
            marker_color="#10b981", text=[f"{r:.1f}% (n={c})" for r, c in zip(cat_grp["Rate"], cat_grp["Count"])],
            textposition="inside", showlegend=False
        ), row=1, col=2)
        fig_r2.for_each_annotation(lambda a: a.update(font=dict(size=12, color="#1e293b", family="Inter, sans-serif")))
        fig_r2.update_layout(height=350, margin=dict(l=25, r=25, t=45, b=25), template="plotly_white")
        display(fig_r2)
        
        # Row 3: Hotels per Location (Top 15) & Price vs Review Count
        fig_r3 = make_subplots(
            rows=1, cols=2,
            horizontal_spacing=0.16,
            subplot_titles=("Top 15 Hotel Locations", "Price vs. Review Count (by Success)")
        )
        loc_counts = f_df["location"].value_counts().head(15).sort_values(ascending=True)
        fig_r3.add_trace(go.Bar(
            y=loc_counts.index, x=loc_counts.values, orientation="h",
            marker_color="#0284c7", showlegend=False
        ), row=1, col=1)
        
        # Price vs Review Count scatter
        p_cap = np.percentile(f_df["price"], 98) if len(f_df) > 10 else f_df["price"].max()
        r_cap = np.percentile(f_df["review_count"], 98) if len(f_df) > 10 else f_df["review_count"].max()
        sub_scatter = f_df[(f_df["price"] <= p_cap) & (f_df["review_count"] <= r_cap)]
        for s_val, s_label, s_color in [(1, "Success (≥4.0)", "#10b981"), (0, "Not Success (<4.0)", "#ef4444")]:
            s_data = sub_scatter[sub_scatter["success"] == s_val]
            if len(s_data) > 0:
                fig_r3.add_trace(go.Scatter(
                    x=s_data["review_count"], y=s_data["price"], mode="markers",
                    name=s_label, marker=dict(color=s_color, opacity=0.5, size=6)
                ), row=1, col=2)
        fig_r3.update_xaxes(title_text="Review Count", row=1, col=2)
        fig_r3.update_yaxes(title_text="Price (₹)", row=1, col=2)
        fig_r3.for_each_annotation(lambda a: a.update(font=dict(size=12, color="#1e293b", family="Inter, sans-serif")))
        fig_r3.update_layout(height=400, margin=dict(l=25, r=25, t=45, b=25), template="plotly_white")
        display(fig_r3)
        
        # Row 4: Amenity Availability & Success Difference
        fig_r4 = make_subplots(
            rows=1, cols=2,
            horizontal_spacing=0.18,
            subplot_titles=("Amenity Market Prevalence (%)", "Success Rate Δ: With vs. Without Amenity")
        )
        clean_amen_names = [c.replace("amenity_", "").replace("_", " ").title() for c in amenity_cols]
        amen_prev = [f_df[c].mean() * 100 for c in amenity_cols]
        
        diff_data = []
        for c, name in zip(amenity_cols, clean_amen_names):
            with_r = f_df[f_df[c] == 1]["success"].mean() * 100 if (f_df[c] == 1).sum() > 0 else 0
            wout_r = f_df[f_df[c] == 0]["success"].mean() * 100 if (f_df[c] == 0).sum() > 0 else 0
            diff_data.append({"Amenity": name, "Diff": with_r - wout_r})
        diff_df = pd.DataFrame(diff_data).sort_values("Diff", ascending=True)
        
        fig_r4.add_trace(go.Bar(
            y=clean_amen_names, x=amen_prev, orientation="h",
            marker_color="#6366f1", showlegend=False
        ), row=1, col=1)
        
        colors_diff = ["#10b981" if v >= 0 else "#ef4444" for v in diff_df["Diff"]]
        fig_r4.add_trace(go.Bar(
            y=diff_df["Amenity"], x=diff_df["Diff"], orientation="h",
            marker_color=colors_diff, showlegend=False
        ), row=1, col=2)
        fig_r4.for_each_annotation(lambda a: a.update(font=dict(size=12, color="#1e293b", family="Inter, sans-serif")))
        fig_r4.update_layout(height=420, margin=dict(l=25, r=25, t=45, b=25), template="plotly_white")
        display(fig_r4)
        
        # Row 5: Amenity Count by Success & Correlation Matrix
        fig_r5 = make_subplots(
            rows=1, cols=2,
            horizontal_spacing=0.16,
            subplot_titles=("Amenity Count by Success Status", "Correlation Matrix (Rating Excluded)"),
            specs=[[{"type": "xy"}, {"type": "heatmap"}]]
        )
        
        fig_r5.add_trace(go.Box(
            x=f_df["success"].map({1: "Successful (≥4.0)", 0: "Not Successful (<4.0)"}),
            y=f_df["amenity_count"], marker_color="#8b5cf6", boxpoints="outliers", showlegend=False
        ), row=1, col=1)
        
        corr_cols = ["price", "log_price", "review_count", "amenity_count", "success"]
        corr_mat = f_df[corr_cols].corr()
        fig_r5.add_trace(go.Heatmap(
            z=corr_mat.values, x=corr_cols, y=corr_cols,
            colorscale="Blues", zmin=-0.2, zmax=1.0, text=corr_mat.round(2).values,
            texttemplate="%{text}", showscale=False
        ), row=1, col=2)
        fig_r5.for_each_annotation(lambda a: a.update(font=dict(size=12, color="#1e293b", family="Inter, sans-serif")))
        fig_r5.update_layout(height=360, margin=dict(l=25, r=25, t=45, b=25), template="plotly_white")
        display(fig_r5)

# --- RENDER TAB 2: MODEL RESULTS ---
def render_tab2(f_df):
    out_tab2.clear_output(wait=True)
    with out_tab2:
        # Apply filters to test set
        test_indices = X_test.index.intersection(f_df.index)
        n_test_sub = len(test_indices)
        
        display(HTML(f\"\"\"
        <div class="model-champion-card">
            <div style="font-size:16px; font-weight:700; margin-bottom:4px;">Selected Champion Model: Tuned Random Forest Classifier</div>
            <div style="font-size:13px;">
                <b>Hyperparameters:</b> 300 Trees | max_depth=10 | min_samples_split=5 | class_weight='balanced' | 21 Predictors.<br>
                <b>Full Test Set Benchmark:</b> Accuracy <b>81.34%</b> | Macro Precision <b>0.6777</b> | Macro Recall <b>0.6443</b> | Macro-F1 <b>0.6571</b> | Weighted-F1 <b>0.8037</b> | ROC-AUC <b>0.7072</b>.
            </div>
        </div>
        <div class="tab-note-banner">
            <b>Filter Applicability:</b> Filters apply to the 911 test-set hotels. Currently evaluating on <b>{n_test_sub:,}</b> matching test hotels.<br>
            <i>Notice: Below table displays accuracy, precision, recall, F1-scores, and ROC-AUC on the filtered test subset alongside the unfiltered Review 1 benchmarks.</i>
        </div>
        \"\"\"))
        
        if n_test_sub == 0:
            display(HTML("<div class='empty-state-box'>No test-set hotels match current filter selection.</div>"))
            return
            
        if n_test_sub < 50:
            display(HTML(f\"\"\"
            <div class="warning-badge">
                <b>Sample Size Notice:</b> The current filter selection contains only <b>{n_test_sub}</b> test hotels (< 50 threshold).
                Performance metrics computed on small subsets exhibit high variance and may not generalize.
            </div>
            \"\"\"))
            
        y_test_sub = y_test.loc[test_indices]
        single_class = y_test_sub.nunique() < 2
        
        if single_class:
            display(HTML(f\"\"\"
            <div class="warning-badge">
                <b>Single Class Notice:</b> Filtered test set contains only class <b>{y_test_sub.iloc[0]}</b>.
                Precision, recall, macro metrics, and ROC-AUC cannot be computed.
            </div>
            \"\"\"))
            
        # Recompute test metrics on filtered test hotels
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
            try:
                prob_sub = m.predict_proba(X_test.loc[test_indices])[:, 1]
                auc = roc_auc_score(y_test_sub, prob_sub) if not single_class else np.nan
            except:
                auc = np.nan
                
            rows_eval.append({
                "Model": name,
                "Test N": n_test_sub,
                "Accuracy": f"{acc:.4f}",
                "Macro Prec": f"{prec_m:.4f}" if not np.isnan(prec_m) else "N/A",
                "Macro Rec": f"{rec_m:.4f}" if not np.isnan(rec_m) else "N/A",
                "Macro F1": f"{f1_m:.4f}" if not np.isnan(f1_m) else "N/A",
                "Weighted F1": f"{f1_w:.4f}" if not np.isnan(f1_w) else "N/A",
                "ROC-AUC": f"{auc:.4f}" if not np.isnan(auc) else "N/A",
                "Base Acc (R1)": f"{bench['Accuracy']:.4f}",
                "Base F1 (R1)": f"{bench['Macro F1']:.4f}"
            })
            
        eval_df = pd.DataFrame(rows_eval)
        
        # HTML Table
        table_html = "<table style='width:100%; border-collapse:collapse; margin-bottom:20px; font-size:13px; background:#fff;'>"
        table_html += "<tr style='background:#f1f5f9; border-bottom:2px solid #cbd5e1;'>" + "".join([f"<th style='padding:8px 12px; text-align:left;'>{c}</th>" for c in eval_df.columns]) + "</tr>"
        for _, r in eval_df.iterrows():
            is_rf = "Random Forest" in r["Model"]
            style = "font-weight:600; background:#f0fdf4;" if is_rf else "border-bottom:1px solid #e2e8f0;"
            table_html += f"<tr style='{style}'>" + "".join([f"<td style='padding:8px 12px;'>{v}</td>" for v in r.values]) + "</tr>"
        table_html += "</table>"
        display(HTML(table_html))
        
        # Comparison Bar Chart
        fig_cmp = go.Figure()
        metric_cols = [("Accuracy", "#3b82f6"), ("Macro Prec", "#0284c7"), ("Macro Rec", "#f59e0b"), ("Macro F1", "#10b981")]
        for m_col, col_c in metric_cols:
            fig_cmp.add_trace(go.Bar(
                x=eval_df["Model"],
                y=[float(v) if v != "N/A" else 0 for v in eval_df[m_col]],
                name=m_col, marker_color=col_c
            ))
        fig_cmp.update_layout(
            title=dict(text="Model Performance Comparison on Current Test Subset", x=0.01, xanchor="left", font=dict(size=14, color="#1e293b")),
            barmode="group", height=320, margin=dict(l=25, r=25, t=45, b=25), template="plotly_white"
        )
        display(fig_cmp)
        
        if not single_class:
            # Side-by-side Confusion Matrices
            fig_cm = make_subplots(
                rows=1, cols=3,
                horizontal_spacing=0.08,
                subplot_titles=("Logistic Regression", "Decision Tree", "Random Forest (Selected)")
            )
            for idx, (m_name, m_obj) in enumerate([("Logistic Regression", model_lr), ("Decision Tree", model_dt), ("Random Forest", model_rf)], 1):
                p_s = m_obj.predict(X_test.loc[test_indices])
                cm = confusion_matrix(y_test_sub, p_s, labels=[0, 1])
                fig_cm.add_trace(go.Heatmap(
                    z=cm, x=["Pred 0 (<4.0)", "Pred 1 (≥4.0)"], y=["True 0", "True 1"],
                    colorscale="Blues", text=cm, texttemplate="%{text}", showscale=False
                ), row=1, col=idx)
            fig_cm.for_each_annotation(lambda a: a.update(font=dict(size=12, color="#1e293b", family="Inter, sans-serif")))
            fig_cm.update_layout(height=280, margin=dict(l=25, r=25, t=45, b=25), template="plotly_white")
            display(fig_cm)
            
            # ROC Curves & Top 15 Feature Importances
            fig_roc_imp = make_subplots(
                rows=1, cols=2,
                horizontal_spacing=0.16,
                subplot_titles=("ROC Curves (AUC)", "Top 15 Feature Importances (RF)")
            )
            for m_name, m_obj, c_color in [("Logistic Regression", model_lr, "#f59e0b"), ("Decision Tree", model_dt, "#06b6d4"), ("Random Forest", model_rf, "#10b981")]:
                probs = m_obj.predict_proba(X_test.loc[test_indices])[:, 1]
                fpr, tpr, _ = roc_curve(y_test_sub, probs)
                auc_v = roc_auc_score(y_test_sub, probs)
                fig_roc_imp.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines", name=f"{m_name} (AUC={auc_v:.3f})", line=dict(color=c_color, width=2)
                ), row=1, col=1)
            fig_roc_imp.add_shape(type="line", line=dict(dash="dash", color="gray"), x0=0, x1=1, y0=0, y1=1, row=1, col=1)
            
            fig_roc_imp.add_trace(go.Bar(
                y=rf_feat_df["Feature"][::-1], x=rf_feat_df["Importance"][::-1], orientation="h",
                marker_color="#6366f1", showlegend=False
            ), row=1, col=2)
            fig_roc_imp.for_each_annotation(lambda a: a.update(font=dict(size=12, color="#1e293b", family="Inter, sans-serif")))
            fig_roc_imp.update_layout(height=400, margin=dict(l=25, r=25, t=45, b=25), template="plotly_white")
            display(fig_roc_imp)

# --- RENDER TAB 3: TEXT MINING [DYNAMIC RECALCULATION] ---
def render_tm_reviews_table(sub_revs):
    out_tm_reviews.clear_output(wait=True)
    with out_tm_reviews:
        query = w_tm_search.value.strip().lower()
        disp_revs = sub_revs
        if query:
            m_text = disp_revs["text_light"].fillna("").str.lower().str.contains(query, regex=False)
            m_hotel = disp_revs["hotel_name"].fillna("").str.lower().str.contains(query, regex=False)
            disp_revs = disp_revs[m_text | m_hotel]
            
        n_match = len(disp_revs)
        page_size = 15
        total_pages = max(1, (n_match + page_size - 1) // page_size)
        curr_page = max(0, min(tm_page_state["page"], total_pages - 1))
        tm_page_state["page"] = curr_page
        
        lbl_tm_page.value = f"<span style='font-size:12.5px; color:#475569;'>Page <b>{curr_page+1}</b> of <b>{total_pages}</b> ({n_match:,} reviews)</span>"
        btn_tm_prev.disabled = (curr_page == 0)
        btn_tm_next.disabled = (curr_page >= total_pages - 1)
        
        if n_match == 0:
            display(HTML("<div style='padding:15px; color:#64748b; font-size:13px; text-align:center;'>No reviews match search term.</div>"))
            return
            
        start_idx = curr_page * page_size
        end_idx = min(start_idx + page_size, n_match)
        page_df = disp_revs.iloc[start_idx:end_idx]
        
        thtml = "<table class='dataset-table' style='font-size:12px; margin-top:10px;'>"
        thtml += "<thead><tr><th style='width:22%;'>Hotel Name</th><th style='width:12%;'>Location</th><th style='width:12%;'>Category</th><th style='width:8%; text-align:center;'>Rating</th><th style='width:12%; text-align:center;'>VADER Sentiment</th><th>Review Text Excerpt</th></tr></thead><tbody>"
        
        for _, r in page_df.iterrows():
            sent = r["vader_sentiment"]
            b_class = "badge-vader-pos" if sent == "Positive" else ("badge-vader-neg" if sent == "Negative" else "badge-vader-neu")
            thtml += f"<tr><td style='font-weight:600;'>{r['hotel_name']}</td><td>{r['analysis_location']}</td><td>{r['hotel_category_clean']}</td><td style='text-align:center;'>★ {r['review_rating_5']:.1f}</td><td style='text-align:center;'><span class='{b_class}'>{sent}</span></td><td style='color:#334155; line-height:1.4;'>{r['text_light']}</td></tr>"
            
        thtml += "</tbody></table>"
        display(HTML(thtml))

def render_tab3(f_df):
    out_tab3.clear_output(wait=True)
    with out_tab3:
        # 1. Determine review subset
        is_unfiltered = (len(f_df) == len(clean_hotels_df))
        if is_unfiltered:
            sub_revs = rev_sentiment_df
            sub_idx = np.arange(len(rev_sentiment_df))
            real_ids = set(clean_hotels_df[clean_hotels_df["data_type"] == "Real"]["hotel_id"])
        else:
            real_ids = set(f_df[f_df["data_type"] == "Real"]["hotel_id"])
            sub_revs = rev_sentiment_df[rev_sentiment_df["hotel_id"].isin(real_ids)]
            sub_idx = sub_revs.index.values
            
        n_rev = len(sub_revs)
        n_real = len(real_ids)
        
        display(HTML(f\"\"\"
        <div class="tab-note-banner">
            <b>Filter Applicability: [Dynamic]</b> Text mining covers <b>Real Hotels only</b> ({n_real:,} matched hotels, {n_rev:,} reviews).<br>
            All charts, metrics, confusion matrices, pain points, themes, aspect sentiments, word frequencies, word clouds, and key takeaways dynamically recompute from matching reviews.
        </div>
        \"\"\"))
        
        if n_rev == 0:
            display(HTML(\"\"\"
            <div class="empty-state-box">
                <h3 style="color:#0f172a; margin-bottom:6px;">No Real Hotel Reviews Match Filter Selection</h3>
                <p style="color:#64748b; font-size:13px; margin:0;">Please adjust your location, category, rating, or price filters to include real properties with customer reviews.</p>
            </div>
            \"\"\"))
            return
            
        if n_rev < 100:
            display(HTML(f\"\"\"
            <div class="warning-badge">
                <b>Sample Size Notice:</b> Only <b>{n_rev}</b> reviews match current filters (< 100 threshold).
                Metrics and sentiment percentages may exhibit higher variance on small sample sizes.
            </div>
            \"\"\"))
            
        # Row 1: Dynamic VADER Sentiment Donut & Star Rating vs VADER
        fig_tm1 = make_subplots(
            rows=1, cols=2,
            horizontal_spacing=0.18,
            subplot_titles=("VADER Sentiment Distribution [Dynamic]", "Star Rating vs. VADER Sentiment [Dynamic]"),
            specs=[[{"type": "domain"}, {"type": "xy"}]]
        )
        v_counts = sub_revs["vader_sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"]).fillna(0)
        fig_tm1.add_trace(go.Pie(
            labels=v_counts.index, values=v_counts.values, hole=0.55,
            marker=dict(colors=["#10b981", "#94a3b8", "#ef4444"])
        ), row=1, col=1)
        
        ct = pd.crosstab(sub_revs["rating_sentiment"], sub_revs["vader_sentiment"]).reindex(
            index=["Negative", "Neutral", "Positive"], columns=["Negative", "Neutral", "Positive"]
        ).fillna(0)
        for col_name, c_color in [("Positive", "#10b981"), ("Neutral", "#94a3b8"), ("Negative", "#ef4444")]:
            fig_tm1.add_trace(go.Bar(
                x=ct.index, y=ct[col_name], name=f"VADER {col_name}", marker_color=c_color
            ), row=1, col=2)
        fig_tm1.for_each_annotation(lambda a: a.update(font=dict(size=12, color="#1e293b", family="Inter, sans-serif")))
        fig_tm1.update_layout(height=350, barmode="stack", margin=dict(l=25, r=25, t=45, b=25), template="plotly_white")
        display(fig_tm1)
        
        # Row 2: Customer Pain Points & Experience Themes [Dynamic]
        neg_sub_mask = (sub_revs["vader_sentiment"] == "Negative").values
        neg_sub_idx = sub_idx[neg_sub_mask]
        n_neg = len(neg_sub_idx)
        
        pain_res = []
        for k, bool_arr in pain_matrix.items():
            cnt = int(bool_arr[neg_sub_idx].sum()) if n_neg > 0 else 0
            pct = round(cnt / n_neg * 100, 1) if n_neg > 0 else 0.0
            pain_res.append({"Pain Point": k, "Negative Mentions": cnt, "% of negative": pct})
        pain_df = pd.DataFrame(pain_res).sort_values("Negative Mentions", ascending=True).reset_index(drop=True)
        
        vader_arr = sub_revs["vader_sentiment"].values
        theme_res = []
        for k, bool_arr in theme_matrix.items():
            sub_bool = bool_arr[sub_idx]
            tot = int(sub_bool.sum())
            pos = int(((sub_bool) & (vader_arr == "Positive")).sum())
            neg = int(((sub_bool) & (vader_arr == "Negative")).sum())
            neu = int(((sub_bool) & (vader_arr == "Neutral")).sum())
            theme_res.append({
                "Theme": k, "Total Mentions": tot, "Positive Mentions": pos, "Negative Mentions": neg,
                "% Positive": round(pos / tot * 100, 1) if tot > 0 else 0.0,
                "% Negative": round(neg / tot * 100, 1) if tot > 0 else 0.0
            })
        if len(theme_res) == 0:
            theme_df = pd.DataFrame(columns=["Theme", "Total Mentions", "Positive Mentions", "Negative Mentions", "% Positive", "% Negative"])
        else:
            theme_df = pd.DataFrame(theme_res).sort_values("Total Mentions", ascending=False).reset_index(drop=True)
        
        fig_tm2 = make_subplots(
            rows=1, cols=2,
            horizontal_spacing=0.18,
            subplot_titles=("Top Customer Pain Points [Dynamic]", "Experience Themes (% Positive) [Dynamic]")
        )
        top_pain_display = pain_df.tail(8)
        fig_tm2.add_trace(go.Bar(
            y=top_pain_display["Pain Point"], x=top_pain_display["Negative Mentions"], orientation="h",
            marker_color="#ef4444", text=[f"{m:,} ({p:.1f}%)" for m, p in zip(top_pain_display["Negative Mentions"], top_pain_display["% of negative"])],
            textposition="inside", showlegend=False
        ), row=1, col=1)
        
        if is_unfiltered:
            pos_themes = theme_df[theme_df["Total Mentions"] >= 500].sort_values("% Positive", ascending=True)
        else:
            pos_themes = theme_df[theme_df["Total Mentions"] >= 1].sort_values("% Positive", ascending=True)
            if len(pos_themes) > 10:
                pos_themes = pos_themes.tail(10)
                
        if len(pos_themes) > 0:
            fig_tm2.add_trace(go.Bar(
                y=pos_themes["Theme"], x=pos_themes["% Positive"], orientation="h",
                marker_color="#10b981", text=[f"{v:.1f}%" for v in pos_themes["% Positive"]], textposition="inside", showlegend=False
            ), row=1, col=2)
        else:
            fig_tm2.add_annotation(text="No theme mentions in selection", xref="x2", yref="y2", showarrow=False, font=dict(size=12, color="#64748b"), row=1, col=2)
        fig_tm2.for_each_annotation(lambda a: a.update(font=dict(size=12, color="#1e293b", family="Inter, sans-serif")))
        fig_tm2.update_layout(height=380, margin=dict(l=25, r=25, t=45, b=25), template="plotly_white")
        display(fig_tm2)
        
        # Row 3: Word Frequency & Bigrams [Dynamic]
        all_toks = [w for i in sub_idx for w in tokens_list[i]]
        w_freq = Counter(all_toks)
        top_words = [(w, c) for w, c in w_freq.most_common(40) if w not in FILLER][:15]
        top_words_df = pd.DataFrame(top_words, columns=["word", "frequency"]).sort_values("frequency", ascending=True).reset_index(drop=True)
        
        all_bis = [b for i in sub_idx for b in bigrams_list[i]]
        bi_freq = Counter(all_bis)
        top_bis = bi_freq.most_common(10)
        top_bis_df = pd.DataFrame(top_bis, columns=["bigram", "frequency"]).sort_values("frequency", ascending=True).reset_index(drop=True)
        
        fig_tm3 = make_subplots(
            rows=1, cols=2,
            horizontal_spacing=0.18,
            subplot_titles=("Top 15 Frequent Words [Dynamic]", "Top 10 Bigrams in Reviews [Dynamic]")
        )
        fig_tm3.add_trace(go.Bar(
            y=top_words_df["word"], x=top_words_df["frequency"], orientation="h",
            marker_color="#3b82f6", text=top_words_df["frequency"], textposition="inside", showlegend=False
        ), row=1, col=1)
        fig_tm3.add_trace(go.Bar(
            y=top_bis_df["bigram"], x=top_bis_df["frequency"], orientation="h",
            marker_color="#8b5cf6", text=top_bis_df["frequency"], textposition="inside", showlegend=False
        ), row=1, col=2)
        fig_tm3.for_each_annotation(lambda a: a.update(font=dict(size=12, color="#1e293b", family="Inter, sans-serif")))
        fig_tm3.update_layout(height=380, margin=dict(l=25, r=25, t=45, b=25), template="plotly_white")
        display(fig_tm3)
        
        # Row 4: Amenity Aspect Sentiment & Confusion Matrix [Dynamic]
        amen_res = []
        for k, lab_arr in amenity_matrix.items():
            sub_lab = lab_arr[sub_idx]
            tot = int((sub_lab > 0).sum())
            if tot >= (10 if is_unfiltered else 1):
                pos = int((sub_lab == 1).sum())
                neu = int((sub_lab == 2).sum())
                neg = int((sub_lab == 3).sum())
                amen_res.append({
                    "Amenity": k, "Mentions": tot,
                    "Positive %": round(pos / tot * 100, 1),
                    "Neutral %": round(neu / tot * 100, 1),
                    "Negative %": round(neg / tot * 100, 1)
                })
        if len(amen_res) == 0:
            amen_sent_df = pd.DataFrame(columns=["Amenity", "Mentions", "Positive %", "Neutral %", "Negative %"])
        else:
            amen_sent_df = pd.DataFrame(amen_res).sort_values("Positive %", ascending=True).reset_index(drop=True)
        
        cm_tm = confusion_matrix(sub_revs["rating_sentiment"], sub_revs["vader_sentiment"], labels=["Negative", "Neutral", "Positive"])
        
        fig_tm4 = make_subplots(
            rows=1, cols=2,
            horizontal_spacing=0.18,
            subplot_titles=("Amenity Aspect Sentiment (% Positive) [Dynamic]", "VADER vs. Rating Confusion Matrix [Dynamic]"),
            specs=[[{"type": "xy"}, {"type": "heatmap"}]]
        )
        if len(amen_sent_df) > 0:
            fig_tm4.add_trace(go.Bar(
                y=amen_sent_df["Amenity"], x=amen_sent_df["Positive %"], orientation="h",
                marker_color="#06b6d4", text=[f"{v:.1f}%" for v in amen_sent_df["Positive %"]], textposition="inside", showlegend=False
            ), row=1, col=1)
        else:
            fig_tm4.add_annotation(text="No amenity mentions in selection", xref="x1", yref="y1", showarrow=False, font=dict(size=12, color="#64748b"), row=1, col=1)
        fig_tm4.add_trace(go.Heatmap(
            z=cm_tm, x=["Pred Neg", "Pred Neu", "Pred Pos"], y=["True Neg", "True Neu", "True Pos"],
            colorscale="Blues", text=cm_tm, texttemplate="%{text}", showscale=False
        ), row=1, col=2)
        fig_tm4.for_each_annotation(lambda a: a.update(font=dict(size=12, color="#1e293b", family="Inter, sans-serif")))
        fig_tm4.update_layout(height=360, margin=dict(l=25, r=25, t=45, b=25), template="plotly_white")
        display(fig_tm4)
        
        # Row 5: Dynamic Sentiment Evaluation Metrics Bar
        acc = accuracy_score(sub_revs["rating_sentiment"], sub_revs["vader_sentiment"])
        try:
            prec_m = precision_score(sub_revs["rating_sentiment"], sub_revs["vader_sentiment"], average="macro", zero_division=0)
            rec_m = recall_score(sub_revs["rating_sentiment"], sub_revs["vader_sentiment"], average="macro", zero_division=0)
            mf1 = f1_score(sub_revs["rating_sentiment"], sub_revs["vader_sentiment"], average="macro", zero_division=0)
            wf1 = f1_score(sub_revs["rating_sentiment"], sub_revs["vader_sentiment"], average="weighted", zero_division=0)
            kappa = cohen_kappa_score(sub_revs["rating_sentiment"], sub_revs["vader_sentiment"])
            if np.isnan(kappa): kappa = 0.0
        except Exception:
            prec_m, rec_m, mf1, wf1, kappa = 0.0, 0.0, 0.0, 0.0, 0.0
            
        eval_items = [
            ("Accuracy", acc), ("Weighted F1", wf1), ("Cohen's Kappa", kappa),
            ("Macro Precision", prec_m), ("Macro Recall", rec_m), ("Macro F1", mf1)
        ]
        eval_m_df = pd.DataFrame(eval_items, columns=["Metric", "Value"])
        
        fig_eval = go.Figure(go.Bar(
            y=eval_m_df["Metric"][::-1], x=eval_m_df["Value"][::-1], orientation="h",
            marker_color="#10b981", text=[f"{v:.4f}" for v in eval_m_df["Value"][::-1]], textposition="inside"
        ))
        fig_eval.update_layout(
            title=dict(text="Sentiment Evaluation Metrics vs Star Rating Reference [Dynamic]", x=0.01, xanchor="left", font=dict(size=14, color="#1e293b")),
            height=280, margin=dict(l=25, r=25, t=45, b=25), template="plotly_white"
        )
        display(fig_eval)
        
        # Row 6: Dynamic Word Cloud
        cloud_freq = {w: f for w, f in w_freq.items() if w not in GENERIC and w not in NEGATIONS and w not in FILLER}
        if cloud_freq:
            display(HTML("<div style='font-size:14px; font-weight:600; margin: 18px 0 8px 0; color:#0f172a;'>Dynamic Review Word Cloud [Dynamic]</div>"))
            wc = WordCloud(width=800, height=380, background_color="white", colormap="viridis", max_words=80, collocations=False, random_state=42).generate_from_frequencies(cloud_freq)
            img_buf = io.BytesIO()
            wc.to_image().save(img_buf, format="PNG")
            b64_str = base64.b64encode(img_buf.getvalue()).decode("utf-8")
            display(HTML(f"<img src='data:image/png;base64,{b64_str}' style='max-width:100%; width:750px; height:auto; border-radius:8px; border:1px solid #cbd5e1;' />"))
            
        # Row 7: Dynamic Key Text Mining Takeaways (100% computed from sub_revs)
        pos_p = (sub_revs["vader_sentiment"] == "Positive").mean() * 100
        neg_p = (sub_revs["vader_sentiment"] == "Negative").mean() * 100
        neu_p = (sub_revs["vader_sentiment"] == "Neutral").mean() * 100
        
        top_pain_sorted = pain_df.sort_values("Negative Mentions", ascending=False).reset_index(drop=True)
        p1_str = f"<i>{top_pain_sorted.iloc[0]['Pain Point']}</i> ({top_pain_sorted.iloc[0]['% of negative']:.1f}% of negative reviews)" if len(top_pain_sorted) > 0 and n_neg > 0 else "N/A"
        p2_str = f"<i>{top_pain_sorted.iloc[1]['Pain Point']}</i> ({top_pain_sorted.iloc[1]['% of negative']:.1f}%)" if len(top_pain_sorted) > 1 and n_neg > 0 else "N/A"
        p3_str = f"<i>{top_pain_sorted.iloc[2]['Pain Point']}</i> ({top_pain_sorted.iloc[2]['% of negative']:.1f}%)" if len(top_pain_sorted) > 2 and n_neg > 0 else "N/A"
        
        theme_sorted = theme_df[theme_df["Total Mentions"] >= (500 if is_unfiltered else 1)].sort_values("% Positive", ascending=False).reset_index(drop=True)
        t1_str = f"<i>{theme_sorted.iloc[0]['Theme']}</i> ({theme_sorted.iloc[0]['% Positive']:.1f}% positive)" if len(theme_sorted) > 0 else "N/A"
        t2_str = f"<i>{theme_sorted.iloc[1]['Theme']}</i> ({theme_sorted.iloc[1]['% Positive']:.1f}% positive)" if len(theme_sorted) > 1 else "N/A"
        t3_str = f"<i>{theme_sorted.iloc[2]['Theme']}</i> ({theme_sorted.iloc[2]['% Positive']:.1f}% positive)" if len(theme_sorted) > 2 else "N/A"
        
        display(HTML(f\"\"\"
        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-left:4px solid #0284c7; border-radius:8px; padding:16px 20px; margin-top:20px;">
            <h4 style="margin:0 0 8px 0; color:#0f172a; font-size:14px;">Key Text Mining Takeaways [Dynamic Recomputed]:</h4>
            <ul style="margin:0; padding-left:20px; font-size:13px; color:#334155; line-height:1.6;">
                <li><b>Overall Sentiment:</b> <b>{pos_p:.1f}%</b> of reviews are VADER positive, <b>{neg_p:.1f}%</b> negative, <b>{neu_p:.1f}%</b> neutral. VADER achieves <b>{acc*100:.2f}%</b> agreement with customer star ratings (Macro-F1: <b>{mf1:.4f}</b>, Cohen's Kappa: <b>{kappa:.4f}</b>).</li>
                <li><b>Primary Pain Points:</b> {p1_str} is the top complaint driver, followed by {p2_str} and {p3_str}.</li>
                <li><b>Primary Delight Drivers:</b> {t1_str}, {t2_str}, and {t3_str} drive the highest customer satisfaction.</li>
            </ul>
        </div>
        \"\"\"))
        
        # Row 8: Task 4 Review Explorer Table [Dynamic]
        display(HTML("<div style='font-size:14px; font-weight:600; margin:24px 0 8px 0; color:#0f172a;'>Filtered Reviews Explorer [Dynamic]</div>"))
        controls_tm_row = widgets.HBox([w_tm_search, btn_tm_prev, btn_tm_next, lbl_tm_page], layout=widgets.Layout(align_items="center", grid_gap="10px"))
        display(controls_tm_row)
        display(out_tm_reviews)
        render_tm_reviews_table(sub_revs)

# --- RENDER TAB 4: ASSOCIATION RULES ---
def render_tab4(f_df):
    out_tab4.clear_output(wait=True)
    with out_tab4:
        display(HTML(\"\"\"
        <div class="tab-note-banner">
            <b>Filter Applicability:</b> Amenity association rules mined on hotel-level transaction data.<br>
            <i>Note: Amenity names are standardized (merging free/paid variants, 33 total) and differ from the 15 model columns.</i>
        </div>
        \"\"\"))
        
        # Sliders filtering
        supp_thresh = w_apriori_supp.value
        conf_thresh = w_apriori_conf.value
        lift_thresh = w_apriori_lift.value
        search_query = w_apriori_search.value.strip().lower()
        recompute = w_apriori_recompute.value
        
        if recompute:
            # Dynamic Apriori on filtered hotels
            real_ids = set(f_df["hotel_id"])
            matched_tx = apriori_trans_df[apriori_trans_df["hotel_id"].isin(real_ids)]
            n_tx = len(matched_tx)
            
            if n_tx < 50:
                display(HTML(f\"\"\"
                <div class="warning-badge">
                    <b>Sample Size Notice:</b> Only <b>{n_tx}</b> hotels available in filtered transaction matrix (< 50 threshold).
                    Showing precomputed rules to avoid unrepresentative patterns.
                </div>
                \"\"\"))
                active_rules = apriori_rules_df.copy()
            else:
                display(HTML(f\"\"\"
                <div style="background:#eff6ff; border:1px solid #bfdbfe; color:#1d4ed8; padding:8px 12px; border-radius:6px; font-size:12px; margin-bottom:12px;">
                    ⚡ <b>Dynamically Recomputing Apriori on {n_tx:,} Filtered Hotels</b> with mlxtend (Min Support: {supp_thresh:.2f}, Min Conf: {conf_thresh:.2f})...
                </div>
                \"\"\"))
                tx_matrix_bool = matched_tx.drop(columns=["hotel_id"]).astype(bool)
                f_itemsets = apriori(tx_matrix_bool, min_support=supp_thresh, use_colnames=True, max_len=3)
                if len(f_itemsets) > 0:
                    raw_rules = association_rules(f_itemsets, metric="confidence", min_threshold=conf_thresh)
                    if len(raw_rules) > 0:
                        raw_rules["antecedents"] = raw_rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
                        raw_rules["consequents"] = raw_rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
                        raw_rules["Rule"] = raw_rules["antecedents"] + " → " + raw_rules["consequents"]
                        active_rules = raw_rules[raw_rules["lift"] >= lift_thresh].sort_values("lift", ascending=False)
                    else:
                        active_rules = pd.DataFrame()
                else:
                    active_rules = pd.DataFrame()
        else:
            # Filter Precomputed Rules
            active_rules = apriori_rules_df[
                (apriori_rules_df["support"] >= supp_thresh) &
                (apriori_rules_df["confidence"] >= conf_thresh) &
                (apriori_rules_df["lift"] >= lift_thresh)
            ].copy()
            
        if search_query:
            active_rules = active_rules[active_rules["Rule"].str.lower().str.contains(search_query)]
            
        display(HTML(f\"\"\"
        <div style="font-size:14px; font-weight:600; color:#334155; margin-bottom:10px;">
            Active Association Rules Matching Criteria: <b>{len(active_rules):,}</b> (Thresholds: Support ≥ {supp_thresh*100:.1f}%, Conf ≥ {conf_thresh*100:.1f}%, Lift ≥ {lift_thresh:.1f})
        </div>
        \"\"\"))
        
        # Row 1: Top 15 Standardized Amenity Support
        top_amen_supp = apriori_support_df.sort_values("Support", ascending=True).tail(15)
        fig_supp = go.Figure(go.Bar(
            y=top_amen_supp["Amenity"], x=top_amen_supp["Support_Percentage"], orientation="h",
            marker_color="#0284c7", text=[f"{v:.1f}%" for v in top_amen_supp["Support_Percentage"]], textposition="inside"
        ))
        fig_supp.update_layout(
            title=dict(text="Top 15 Standardized Amenities by Overall Support (%)", x=0.01, xanchor="left", font=dict(size=14, color="#1e293b")),
            height=360, margin=dict(l=25, r=25, t=45, b=25), template="plotly_white"
        )
        display(fig_supp)
        
        if len(active_rules) == 0:
            display(HTML("<div class='empty-state-box'>No association rules match current thresholds. Try reducing minimum support, confidence or lift.</div>"))
            return
            
        # Row 2: Top Rules by Lift & Support-Confidence-Lift Scatter
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
        display(fig_ar)
        
        # Display Ranked Table
        display_tbl = active_rules[["Rule", "support", "confidence", "lift"]].copy()
        display_tbl["Support (%)"] = (display_tbl["support"] * 100).round(2)
        display_tbl["Confidence (%)"] = (display_tbl["confidence"] * 100).round(2)
        display_tbl["Lift"] = display_tbl["lift"].round(3)
        display_tbl = display_tbl.drop(columns=["support", "confidence", "lift"]).head(15)
        
        table_html = "<div style='font-size:14px; font-weight:600; margin:16px 0 8px 0;'>Top 15 Ranked Rules Table:</div>"
        table_html += "<table style='width:100%; border-collapse:collapse; font-size:13px; background:#fff;'>"
        table_html += "<tr style='background:#f1f5f9; border-bottom:2px solid #cbd5e1;'>" + "".join([f"<th style='padding:8px 12px; text-align:left;'>{c}</th>" for c in display_tbl.columns]) + "</tr>"
        for _, r in display_tbl.iterrows():
            table_html += "<tr style='border-bottom:1px solid #e2e8f0;'>" + "".join([f"<td style='padding:8px 12px;'>{v}</td>" for v in r.values]) + "</tr>"
        table_html += "</table>"
        display(HTML(table_html))
        
        # Strategic Amenity Bundling Takeaways
        display(HTML(\"\"\"
        <div style="background:#fdf4ff; border:1px solid #f0abfc; border-radius:10px; padding:16px; margin-top:20px;">
            <h4 style="margin:0 0 8px 0; color:#86198f;">Strategic Amenity Bundling Insights:</h4>
            <ul style="margin:0; padding-left:20px; font-size:13px; color:#701a75; line-height:1.6;">
                <li><b>Premium Wellness Hub:</b> <i>Business center + Pool → Spa</i> boasts the highest lift across the corpus (<b>Lift: 6.190</b>, Confidence: 74.02%, Support: 6.57%). Guests expect premium spas when both business facilities and leisure pools exist.</li>
                <li><b>Fitness & Leisure Core:</b> <i>Fitness center + Pool → Spa</i> (<b>Lift: 5.755</b>, Confidence: 68.82%) and <i>Bar + Fitness center → Spa</i> (<b>Lift: 5.526</b>, Confidence: 66.08%) identify strong complementary synergies for full-service properties.</li>
                <li><b>Operational Strategy:</b> Standalone spas rarely succeed without co-located fitness and business infrastructure. Investment in luxury amenities should follow established bundle co-occurrences.</li>
            </ul>
        </div>
        \"\"\"))


# --- RENDER TAB 5: FINAL DATASET TABLE & DUAL CSV DOWNLOAD (TASK 2) ---
def render_tab5(f_df):
    out_tab5.clear_output(wait=True)
    with out_tab5:
        display(HTML(f\"\"\"
        <div class="tab-note-banner">
            <b>Final Project Dataset:</b> Displaying properties matching current global filters.<br>
            Explore the final project dataset below (excluding geographic coordinates) with interactive text search, paging, and dual CSV download capabilities.
        </div>
        \"\"\"))
        
        # Display dataset table
        query = w_tab5_search.value.strip().lower()
        disp_df = f_df
        if query:
            disp_df = disp_df[disp_df["hotel_name"].fillna("").str.lower().str.contains(query, regex=False)]
            
        n_match = len(disp_df)
        page_size = 25
        total_pages = max(1, (n_match + page_size - 1) // page_size)
        curr_page = max(0, min(tab5_page_state["page"], total_pages - 1))
        tab5_page_state["page"] = curr_page
        
        lbl_tab5_page.value = f"<span style='font-size:13px; color:#475569;'>Page <b>{curr_page+1}</b> of <b>{total_pages}</b> ({n_match:,} matching hotels)</span>"
        btn_tab5_prev.disabled = (curr_page == 0)
        btn_tab5_next.disabled = (curr_page >= total_pages - 1)
        
        # Controls Row
        dl_row = widgets.HBox([
            w_tab5_search,
            btn_tab5_prev, btn_tab5_next, lbl_tab5_page,
            widgets.HBox([btn_download_filt, btn_download_full], layout=widgets.Layout(grid_gap="10px", margin="0 0 0 auto"))
        ], layout=widgets.Layout(align_items="center", grid_gap="10px", margin="0 0 14px 0"))
        
        display(dl_row)
        display(out_download)
        
        if n_match == 0:
            display(HTML(\"\"\"
            <div class="empty-state-box">
                <h3 style="color:#0f172a; margin-bottom:6px;">No Hotels Match Filter / Search Criteria</h3>
                <p style="color:#64748b; font-size:13px; margin:0;">Please adjust your global filters or clear the hotel name search query.</p>
            </div>
            \"\"\"))
            return
            
        start_idx = curr_page * page_size
        end_idx = min(start_idx + page_size, n_match)
        page_hotels = disp_df.iloc[start_idx:end_idx]
        
        # Readable Table (Explicitly EXCLUDES latitude and longitude)
        tbl_html = "<table class='dataset-table'>"
        tbl_html += "<thead><tr><th>Hotel Name</th><th>Location</th><th>Category</th><th>Data Type</th><th style='text-align:right;'>Price (₹)</th><th style='text-align:center;'>Rating</th><th style='text-align:right;'>Reviews</th><th style='text-align:center;'>Amenities</th><th style='text-align:center;'>Success Status</th></tr></thead><tbody>"
        
        for _, r in page_hotels.iterrows():
            is_succ = int(r["success"]) == 1
            badge = "<span class='status-badge-success'>1 (Successful)</span>" if is_succ else "<span class='status-badge-unsuccess'>0 (Not Successful)</span>"
            tbl_html += f"<tr><td style='font-weight:600;'>{r['hotel_name']}</td><td>{r['location']}</td><td>{r['hotel_category']}</td><td>{r['data_type']}</td><td style='text-align:right;'>₹{r['price']:,.0f}</td><td style='text-align:center;'>★ {r['customer_rating']:.1f}</td><td style='text-align:right;'>{int(r['review_count']):,}</td><td style='text-align:center;'>{int(r['amenity_count'])}</td><td style='text-align:center;'>{badge}</td></tr>"
            
        tbl_html += "</tbody></table>"
        display(HTML(tbl_html))

# --- CSV DOWNLOAD HANDLERS (WORKS IN COLAB & LOCAL JUPYTER) ---
def handle_download_filtered(b):
    out_download.clear_output(wait=True)
    with out_download:
        f_df = get_filtered_hotels()
        cols = ["hotel_name", "location", "hotel_category", "data_type", "price", "customer_rating", "review_count", "amenity_count", "success"]
        export_df = f_df[cols].copy()
        filename = "filtered_hotels.csv"
        
        if IN_COLAB:
            from google.colab import files
            export_df.to_csv(filename, index=False)
            files.download(filename)
            display(HTML(f"<div style='margin-top:6px; color:#059669; font-size:12.5px; font-weight:600;'>✓ Initiated Google Colab download for <b>{filename}</b> ({len(export_df):,} rows).</div>"))
        else:
            csv_str = export_df.to_csv(index=False)
            b64 = base64.b64encode(csv_str.encode("utf-8")).decode("utf-8")
            data_uri = f"data:text/csv;base64,{b64}"
            display(HTML(f\"\"\"
            <div style="margin-top:8px;">
                <a id="auto_dl_filt" href="{data_uri}" download="{filename}" style="background:#0284c7; color:#ffffff; padding:6px 14px; border-radius:6px; font-size:12.5px; font-weight:600; text-decoration:none; display:inline-block;">
                    ⬇ Download {filename} ({len(export_df):,} rows)
                </a>
                <script>
                    var el = document.getElementById('auto_dl_filt');
                    if (el) {{ el.click(); }}
                </script>
            </div>
            \"\"\"))

def handle_download_full(b):
    out_download.clear_output(wait=True)
    with out_download:
        cols = ["hotel_name", "location", "hotel_category", "data_type", "price", "customer_rating", "review_count", "amenity_count", "success"]
        export_df = clean_hotels_df[cols].copy()
        filename = "full_final_dataset.csv"
        
        if IN_COLAB:
            from google.colab import files
            export_df.to_csv(filename, index=False)
            files.download(filename)
            display(HTML(f"<div style='margin-top:6px; color:#059669; font-size:12.5px; font-weight:600;'>✓ Initiated Google Colab download for <b>{filename}</b> ({len(export_df):,} rows).</div>"))
        else:
            csv_str = export_df.to_csv(index=False)
            b64 = base64.b64encode(csv_str.encode("utf-8")).decode("utf-8")
            data_uri = f"data:text/csv;base64,{b64}"
            display(HTML(f\"\"\"
            <div style="margin-top:8px;">
                <a id="auto_dl_full" href="{data_uri}" download="{filename}" style="background:#0f172a; color:#ffffff; padding:6px 14px; border-radius:6px; font-size:12.5px; font-weight:600; text-decoration:none; display:inline-block;">
                    ⬇ Download {filename} ({len(export_df):,} rows)
                </a>
                <script>
                    var el = document.getElementById('auto_dl_full');
                    if (el) {{ el.click(); }}
                </script>
            </div>
            \"\"\"))

btn_download_filt.on_click(handle_download_filtered)
btn_download_full.on_click(handle_download_full)

# --- MASTER UPDATE CALLBACK & EVENT DEBOUNCE ---
is_updating = False

def update_dashboard(*args):
    global is_updating
    if is_updating:
        return
    is_updating = True
    try:
        f_df = get_filtered_hotels()
        lbl_filter_summary.value = f"<b>Selected Hotels:</b> {len(f_df):,} / {len(clean_hotels_df):,} ({len(f_df)/len(clean_hotels_df)*100:.1f}%)"
        render_kpis(f_df)
        render_tab1(f_df)
        render_tab2(f_df)
        render_tab3(f_df)
        render_tab4(f_df)
        render_tab5(f_df)
    finally:
        is_updating = False

def reset_filters_action(b):
    global is_updating
    is_updating = True
    try:
        w_location.value = ["All Locations"]
        w_category.value = ["All Categories"]
        w_price.value = [min_price, max_price]
        w_rating.value = [0.0, 5.0]
        w_datatype.value = "Both"
        w_apriori_cohort.value = "All Mined Rules (Corpus-Wide)"
        w_apriori_supp.value = 0.05
        w_apriori_conf.value = 0.60
        w_apriori_lift.value = 2.0
        w_apriori_search.value = ""
        w_apriori_recompute.value = False
        w_tab5_search.value = ""
        tab5_page_state["page"] = 0
        w_tm_search.value = ""
        tm_page_state["page"] = 0
    finally:
        is_updating = False
    update_dashboard()

btn_reset.on_click(reset_filters_action)

# --- CONSTRUCT DASHBOARD TABS ---
tabs = widgets.Tab(children=[out_tab1, out_tab2, out_tab3, tab4_container, out_tab5])
tabs.set_title(0, "1. Overview & EDA")
tabs.set_title(1, "2. Predictive Models")
tabs.set_title(2, "3. Text Mining")
tabs.set_title(3, "4. Association Rules")
tabs.set_title(4, "5. Final Dataset")

# Top Filter Panel Layout
filter_row1 = widgets.HBox([w_location, w_category, w_datatype], layout=widgets.Layout(grid_gap="20px", margin="0 0 10px 0"))
filter_row2 = widgets.HBox([w_price, w_rating, btn_reset], layout=widgets.Layout(grid_gap="20px", align_items="center"))

filter_box = widgets.VBox([
    widgets.HTML("<div class='filter-title'>Interactive Global Filters</div>"),
    filter_row1,
    filter_row2,
    lbl_filter_summary
], layout=widgets.Layout(
    border="1px solid #e2e8f0", padding="16px 20px", border_radius="12px", background_color="#ffffff", margin="0 0 20px 0"
))

header_box = widgets.HTML(\"\"\"
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
\"\"\")

dashboard_view = widgets.VBox([
    header_box,
    filter_box,
    out_kpi,
    tabs
], layout=widgets.Layout(max_width="1280px", margin="0 auto"))

# Unbind any existing observers to guarantee zero duplicate callbacks on re-run
for _w in [w_location, w_category, w_price, w_rating, w_datatype,
           w_apriori_supp, w_apriori_conf, w_apriori_lift, w_apriori_search, w_apriori_recompute,
           w_tab5_search, w_tm_search]:
    try:
        _w.unobserve_all()
    except Exception:
        pass

# Display dashboard container so widgets have active DOM binding
display(dashboard_view)

# Wire observers cleanly
w_location.observe(update_dashboard, names="value")
w_category.observe(update_dashboard, names="value")
w_price.observe(update_dashboard, names="value")
w_rating.observe(update_dashboard, names="value")
w_datatype.observe(update_dashboard, names="value")
w_apriori_supp.observe(update_dashboard, names="value")
w_apriori_conf.observe(update_dashboard, names="value")
w_apriori_lift.observe(update_dashboard, names="value")
w_apriori_search.observe(update_dashboard, names="value")
w_apriori_recompute.observe(update_dashboard, names="value")

# Perform initial render to populate Output widgets cleanly
update_dashboard()

# Observers for Tab 5 pagination and search
def on_tab5_search_change(change):
    tab5_page_state["page"] = 0
    render_tab5(get_filtered_hotels())
w_tab5_search.observe(on_tab5_search_change, names="value")

def on_tab5_prev_click(b):
    if tab5_page_state["page"] > 0:
        tab5_page_state["page"] -= 1
        render_tab5(get_filtered_hotels())
btn_tab5_prev.on_click(on_tab5_prev_click)

def on_tab5_next_click(b):
    tab5_page_state["page"] += 1
    render_tab5(get_filtered_hotels())
btn_tab5_next.on_click(on_tab5_next_click)

# Observers for Tab 3 Review Explorer
def on_tm_search_change(change):
    tm_page_state["page"] = 0
    f_df = get_filtered_hotels()
    real_ids = set(f_df[f_df["data_type"] == "Real"]["hotel_id"]) if len(f_df) < len(clean_hotels_df) else set(clean_hotels_df[clean_hotels_df["data_type"] == "Real"]["hotel_id"])
    sub_revs = rev_sentiment_df if len(f_df) == len(clean_hotels_df) else rev_sentiment_df[rev_sentiment_df["hotel_id"].isin(real_ids)]
    render_tm_reviews_table(sub_revs)
w_tm_search.observe(on_tm_search_change, names="value")

def on_tm_prev_click(b):
    if tm_page_state["page"] > 0:
        tm_page_state["page"] -= 1
        f_df = get_filtered_hotels()
        real_ids = set(f_df[f_df["data_type"] == "Real"]["hotel_id"]) if len(f_df) < len(clean_hotels_df) else set(clean_hotels_df[clean_hotels_df["data_type"] == "Real"]["hotel_id"])
        sub_revs = rev_sentiment_df if len(f_df) == len(clean_hotels_df) else rev_sentiment_df[rev_sentiment_df["hotel_id"].isin(real_ids)]
        render_tm_reviews_table(sub_revs)
btn_tm_prev.on_click(on_tm_prev_click)

def on_tm_next_click(b):
    tm_page_state["page"] += 1
    f_df = get_filtered_hotels()
    real_ids = set(f_df[f_df["data_type"] == "Real"]["hotel_id"]) if len(f_df) < len(clean_hotels_df) else set(clean_hotels_df[clean_hotels_df["data_type"] == "Real"]["hotel_id"])
    sub_revs = rev_sentiment_df if len(f_df) == len(clean_hotels_df) else rev_sentiment_df[rev_sentiment_df["hotel_id"].isin(real_ids)]
    render_tm_reviews_table(sub_revs)
btn_tm_next.on_click(on_tm_next_click)

"""

# Add cells to notebook
nb.cells.append(nbf.v4.new_markdown_cell(cell0_md))
nb.cells.append(nbf.v4.new_code_cell(cell1_code))
nb.cells.append(nbf.v4.new_code_cell(cell2_code))
nb.cells.append(nbf.v4.new_code_cell(cell3_code))
nb.cells.append(nbf.v4.new_code_cell(cell4_code))

out_path = "dashboard/hotel_dashboard.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Generated {out_path} successfully with {len(nb.cells)} cells.")
