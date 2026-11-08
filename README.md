# Integrated Hotel Analytics for Hotel Success, Amenities & Customer Experience

[![Python 3.10 | 3.11 | 3.12](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebooks-orange.svg)](https://jupyter.org/)
[![Plotly & ipywidgets](https://img.shields.io/badge/Dashboard-Plotly%20%26%20ipywidgets-6366f1.svg)](https://plotly.com/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)
[![NLTK VADER](https://img.shields.io/badge/NLP-NLTK%20VADER-388e3c.svg)](https://www.nltk.org/)
[![Apriori mlxtend](https://img.shields.io/badge/Association%20Rules-mlxtend-8e24aa.svg)](https://rasbt.github.io/mlxtend/)
[![Academic Project](https://img.shields.io/badge/Course-Business%20Analytics%20(23CSE452)-red.svg)]()

> **Business Analytics (`23CSE452`) Capstone Project — Team 5**  
> An end-to-end analytical intelligence system combining **Exploratory Data Analysis (EDA)**, **Conditional Tabular GAN (CTGAN) synthetic data generation**, **predictive machine learning benchmarks**, **VADER text mining on 26,175 customer reviews**, **Apriori association rule mining across 33 standardized amenities**, and an **interactive 4-tab decision-support dashboard**.

---

## Table of Contents
1. [Executive Summary & Analytical Highlights](#1-executive-summary--analytical-highlights)
2. [Project Architecture & Repository Directory Tree](#2-project-architecture--repository-directory-tree)
3. [Analytical Workstreams & Notebook Mapping](#3-analytical-workstreams--notebook-mapping)
4. [Runtime, OS & Version Requirements](#4-runtime-os--version-requirements)
5. [Installation & Environment Setup](#5-installation--environment-setup)
6. [Commands to Reproduce All Notebooks, Models & Application](#6-commands-to-reproduce-all-notebooks-models--application)
7. [Dataset Documentation: Provenance, License & Access](#7-dataset-documentation-provenance-license--access)
8. [Benchmarking Results & Key Metrics](#8-benchmarking-results--key-metrics)
9. [Academic Team & Attribution](#9-academic-team--attribution)

---

## 1. Executive Summary & Analytical Highlights

This study investigates the primary determinants of hotel success across 12 major Indian metropolitan and leisure hospitality markets (Delhi, Mumbai, Bengaluru, Chennai, Kolkata, Hyderabad, Pune, Jaipur, Goa, Kochi, Coimbatore, and Visakhapatnam). The project synthesizes quantitative property features, synthetic data augmentation, customer text sentiment, and market basket amenity interactions:

* **The Core Success Paradox**: Across 4,551 usable hotels, **81.7% (3,718 properties)** achieve the success proxy of observed customer rating $\ge 4.0$. However, linear pricing correlation with customer rating is negligible ($r = 0.086$), and total amenity count exhibits a slight negative correlation with rating ($r = -0.134$). Hotel success is driven not by raw price or amenity volume, but by service execution, customer review volume, and synergistic amenity combinations.
* **Predictive Champion Architecture**: A tuned **Random Forest Classifier** (300 estimators, `max_depth=10`, `min_samples_split=5`, balanced class weighting across 21 predictors) achieved **81.34% Accuracy**, **0.6571 Macro-F1**, and an **ROC-AUC of 0.7072** on the stratified holdout test set ($N = 911$), outperforming Decision Trees (80.35%) and Logistic Regression (68.94%).
* **The Operational Hygiene Collapse**: Natural language processing on 26,175 customer reviews reveals that while sentiment is predominantly positive (76.3% VADER positive), customer dissatisfaction is concentrated in fundamental operational failures: **cleanliness/hygiene defects account for 30.7% of all negative reviews (1,497 reviews)**, followed by **rude/unhelpful staff (20.4%, 992 reviews)**. Reviews explicitly discussing **Bedding (55.7% negative)**, **Bathrooms (52.0% negative)**, and **Air Conditioning (50.8% negative)** suffer severe negative sentiment compared to the 18.6% corpus baseline.
* **The High-Lift Wellness Triad**: Association rule mining discovered **154 strong rules** ($\ge 5\%$ support, $\ge 60\%$ confidence, $\ge 2.0$ lift). Standalone spas are rare (11.96% market support); however, when paired with a **Pool and Business Center**, the likelihood of spa presence jumps by **6.19×** (`Business center, Pool → Spa`, Lift: 6.190, Confidence: 74.02%, Support: 6.57%).

---

## 2. Project Architecture & Repository Directory Tree

The repository preserves the project deliverables, organized into root notebooks, raw and unified datasets, text mining outputs, and the integrated dashboard deliverable package:

```text
.
├── .gitignore                                                  # Git exclusion configuration for virtualenvs & cache
├── README.md                                                   # Master comprehensive project documentation (this file)
├── requirements.txt                                            # Root Python dependencies specification
├── environment.yml                                             # Conda/Mamba virtual environment definition
│
├── BA_WebScapring.ipynb                                        # Stage 1: SerpApi Google Hotels scraper across 12 cities
├── hotel_synthetic_data_ctgan_unique_hotel_names (1).ipynb     # Stage 2: CTGAN synthetic tabular data generator
├── BA_total_Review1_Complete_CORRECTED (2).ipynb               # Stage 3: Review 1 EDA & tuned predictive ML benchmarks
├── Text_Mining_Hotel_Reviews_Durgam_Poojitha (1) (1).ipynb     # Stage 4: Review 2 NLP text mining & VADER sentiment
├── review2_updated.ipynb                                       # Stage 5: Review 2 integrated text mining & Apriori rules
│
├── data/                                                       # Root data directory
│   ├── all_india_1685_unique_hotels.csv                        # Raw scraped hotel listings (1,685 unique properties)
│   ├── all_india_hotel_cleaned.csv.zip                         # Raw customer reviews corpus (29,987 reviews, zipped)
│   └── combined_real_synthetic_hotels.csv                      # Unified primary dataset (1,664 real + 2,887 synthetic)
│
├── text-mining-outputs/                                        # Stage 4 Text Mining standalone artifacts (36 files)
│   ├── presentation_numbers.json                               # Central verified metrics and summary counts
│   ├── review_level_sentiment.csv                              # 26,175 reviews with VADER sentiment classifications
│   ├── hotel_level_text_sentiment.csv                          # Hotel-aggregated sentiment metrics (1,659 hotels)
│   ├── pain_points.csv & pain_points.png                       # Ranked complaint drivers and visualization
│   ├── theme_sentiment.csv & theme_sentiment.png               # 16 thematic categories sentiment distributions
│   ├── sentiment_by_category.csv & sentiment_by_city.csv       # Sentiment breakdown across star ratings & cities
│   ├── amenity_mention_sentiment.csv & amenity_sentiment.png   # Sentiment of sentences discussing specific amenities
│   ├── amenity_listing_comparison.csv                          # Rating comparison of hotels listing vs not listing
│   ├── word_frequency_top100.csv & top20_words.png             # Unigram frequencies and distribution chart
│   ├── bigrams_top50.csv & trigrams_top50.csv                  # Multi-word n-gram collocation tables
│   ├── top_bigrams_trigrams.png & top20_positive_words.png     # Visual n-gram and polarized lexical charts
│   ├── top20_negative_words.png & word_cloud_clean.png         # Clean lemmatized customer review word cloud
│   ├── classification_report.csv & confusion_matrix.png        # 3-class VADER vs. rating reference evaluation
│   ├── binary_evaluation_metrics.csv                           # Forced binary & decisive signal evaluation metrics
│   └── sentiment_evaluation_metrics.csv                        # Accuracy, Kappa, Spearman correlation, ROC-AUC
│
└── hotel-dashboard-deliverable/                                # Stage 6 Integrated Review 2 Master Deliverable Package
    ├── README.txt                                              # Review 2 submission manifest and execution guide
    │
    ├── dashboard/                                              # Dashboard application source and documentation
    │   ├── hotel_dashboard.ipynb                               # Master 5-cell interactive widget dashboard
    │   ├── generate_dashboard_notebook.py                      # Automated compiler script to build dashboard notebook
    │   ├── generate_screenshots.py                             # Headless Chrome snapshot capture utility
    │   ├── test_dashboard_filters.py                           # Automated test suite for dynamic filtering
    │   ├── DOCUMENTATION.md                                    # Technical architecture & methodology report
    │   ├── USER_GUIDE.md                                       # Comprehensive user manual and filter guide
    │   ├── verification_report.md                              # Stage A 100% data match verification audit report
    │   ├── verification_report_v2.md                           # Extended audit log for full-range pricing filters
    │   ├── requirements.txt                                    # Dashboard-specific lightweight dependencies
    │   │
    │   └── screenshots/                                        # High-resolution UI evidence captures (1360x2200)
    │       ├── dashboard.png & dashboard_overview.png          # Full interactive dashboard view
    │       ├── dashboard_before_filters.png                    # Baseline state (4,551 hotels, ₹4,077 avg price)
    │       ├── dashboard_after_filters.png                     # Active filter state (Goa 5-star ₹5k-₹25k, N=34)
    │       ├── filter_interaction_before.png & after.png       # Side-by-side filter reactivity evidence
    │       ├── tab1_eda.png & tab1_eda.html                    # Tab 1: 10 coordinated EDA charts
    │       ├── tab2_models.png & tab2_models.html              # Tab 2: Predictive model benchmarks & ROC curves
    │       ├── tab3_text_mining.png & tab3_text_mining.html    # Tab 3: Text mining sentiment & pain points
    │       ├── tab4_association_rules.png & .html              # Tab 4: Apriori amenity bundles & scatter plot
    │       └── tab5_final_dataset.png                          # Tab 5: Dataset view and export controls
    │
    └── data/                                                   # Dashboard local data store (guarantees portability)
        ├── combined_real_synthetic_hotels.csv                  # 4,551 clean hotel properties
        ├── apriori/                                            # Apriori frequent itemsets and strong rules
        │   ├── apriori_amenity_support.csv                     # 33 standardized amenities support ranking
        │   ├── apriori_frequent_itemsets.csv                   # 799 itemsets mined at min support 0.05
        │   ├── apriori_strong_rules.csv                        # 154 association rules (min lift 2.0, conf 0.60)
        │   └── apriori_transaction_matrix.csv                  # 1,430 real hotel amenity transaction vectors
        └── text_mining_outputs/                                # Local replica of the 36 text mining artifacts
```

---

## 3. Analytical Workstreams & Notebook Mapping

```
                                          ANALYTICAL WORKFLOW PIPELINE
                                          
   ┌────────────────────────┐         ┌────────────────────────┐
   │  BA_WebScapring.ipynb  │         │  1,685 Scraped Hotels  │
   │  (SerpApi Google API)  │ ──────> │  29,987 Review Records │
   └────────────────────────┘         └───────────┬────────────┘
                                                  │
                 ┌────────────────────────────────┴────────────────────────────────┐
                 │                                                                 │
                 ▼                                                                 ▼
   ┌───────────────────────────┐                                     ┌───────────────────────────┐
   │ hotel_synthetic_data_     │                                     │ Text_Mining_Hotel_Reviews │
   │ ctgan_unique_hotel_names  │                                     │ _Durgam_Poojitha (NLP)    │
   │ (CTGAN tabular synthesis) │                                     │ (VADER, Themes, Pain Pts) │
   └─────────────┬─────────────┘                                     └─────────────┬─────────────┘
                 │                                                                 │
                 ▼                                                                 ▼
   ┌───────────────────────────┐                                     ┌───────────────────────────┐
   │ combined_real_synthetic_  │                                     │ review2_updated.ipynb     │
   │ hotels.csv (4,551 hotels) │                                     │ (Apriori Association      │
   └─────────────┬─────────────┘                                     │  Rule Mining, 154 rules)  │
                 │                                                   └─────────────┬─────────────┘
                 ▼                                                                 │
   ┌───────────────────────────┐                                                   │
   │ BA_total_Review1_         │                                                   │
   │ Complete_CORRECTED (ML)   │                                                   │
   │ (LR, DT, Random Forest)   │                                                   │
   └─────────────┬─────────────┘                                                   │
                 │                                                                 │
                 └────────────────────────────────┬────────────────────────────────┘
                                                  │
                                                  ▼
                                 ┌─────────────────────────────────┐
                                 │ hotel_dashboard.ipynb           │
                                 │ (Master 4-Tab Interactive UI    │
                                 │  Plotly + ipywidgets Dashboard) │
                                 └─────────────────────────────────┘
```

### Stage 1: Data Acquisition via Web Scraping
* **Notebook**: `BA_WebScapring.ipynb`
* **Focus**: Automated collection of hotel properties, star categories, pricing, review velocity, coordinates, amenities, and user reviews using SerpApi (`google_hotels` engine) across 12 Indian metropolitan and tourist markets.
* **Outputs**: Raw files `data/all_india_1685_unique_hotels.csv` and `data/all_india_hotel_cleaned.csv.zip`.

### Stage 2: Synthetic Data Augmentation via CTGAN
* **Notebook**: `hotel_synthetic_data_ctgan_unique_hotel_names (1).ipynb`
* **Focus**: Addressed sample-size limitations and class representation by training a **Conditional Tabular GAN (CTGAN)** on the 1,685 real hotel records. Generated 2,887 statistically sound synthetic properties with unique names while preserving multivariate covariance.
* **Outputs**: `data/combined_real_synthetic_hotels.csv` (4,572 raw records; 4,551 usable after dropping 21 missing ratings).

### Stage 3: Exploratory Data Analysis & Predictive Modeling (Review 1)
* **Notebook**: `BA_total_Review1_Complete_CORRECTED (2).ipynb`
* **Focus**: Established the rating-based success proxy (`customer_rating >= 4.0`). Evaluated distributions, price dynamics, correlation matrices, and PCA dimensionality reduction. Built and benchmarked three tuned classifiers (Logistic Regression, Decision Tree, Random Forest) across 21 predictors using a stratified 80/20 train/test split.
* **Outputs**: Established the **Tuned Random Forest as the Champion Model** (81.34% test accuracy, 0.6571 Macro-F1).

### Stage 4: Customer Review Text Mining & Sentiment Analysis (Review 2 · Method 1)
* **Notebook**: `Text_Mining_Hotel_Reviews_Durgam_Poojitha (1) (1).ipynb`
* **Author**: Durgam Poojitha (CB.SC.U4CSE23320, Team 5)
* **Focus**: Ingestion of 26,175 clean customer reviews across 1,659 hotels. Implemented full NLP pipeline (tokenization, lemmatization, stopword removal). Generated n-gram frequencies (unigrams, bigrams, trigrams) and clean word clouds. Applied **VADER sentiment classification**, quantifying operational pain points and thematic drivers. Validated sentiment against customer star rating ground truth.
* **Outputs**: 36 export artifacts saved to `text-mining-outputs/` and `hotel-dashboard-deliverable/data/text_mining_outputs/`.

### Stage 5: Association Rule Mining using Apriori (Review 2 · Method 2)
* **Notebook**: `review2_updated.ipynb`
* **Author**: Anupama & Team 5
* **Focus**: Constructed hotel-level transaction vectors across 33 standardized amenities for 1,430 real hotels. Executed the Apriori algorithm with minimum support 0.05, minimum confidence 0.60, and minimum lift 2.0 to uncover complementary amenity bundles.
* **Outputs**: 799 frequent itemsets and 154 strong rules saved to `hotel-dashboard-deliverable/data/apriori/`.

### Stage 6: Integrated Master Dashboard Application
* **Notebook**: `hotel-dashboard-deliverable/dashboard/hotel_dashboard.ipynb`
* **Author**: Sathya Roopan (Dashboard Integrator)
* **Focus**: Complete, production-grade 4-tab interactive dashboard unifying EDA, Predictive ML, Text Mining, and Apriori Association Rules. Powered by `ipywidgets` and `plotly.graph_objects`. Includes dynamic global filtering across Location, Star Category, Price Range, Rating Range, and Data Type, with real-time KPI card calculations and dynamic model re-evaluation on filtered test subsets.

---

## 4. Runtime, OS & Version Requirements

| Requirement Dimension | Specification |
| :--- | :--- |
| **Operating System** | macOS (Apple Silicon M1-M4 / Intel x86_64), Ubuntu Linux 20.04+, Windows 10/11 |
| **Python Version** | **Python 3.10 to 3.12** (Recommended: Python 3.11.x or 3.12.x) |
| **Execution Memory (RAM)** | Minimum 8 GB (16 GB recommended for CTGAN model fitting) |
| **Storage Requirement** | ~500 MB disk space for code, models, datasets, and rendered outputs |
| **Jupyter Environment** | JupyterLab 4.x, Jupyter Notebook 7.x, or VS Code with Python + Jupyter Extensions |
| **Web Browser Engine** | Google Chrome, Mozilla Firefox, Safari, or Microsoft Edge (Chromium-based recommended) |

---

## 5. Installation & Environment Setup

### Option A: Standard Python Virtual Environment (`venv` + `pip`)

```bash
# 1. Clone repository (if not already local)
git clone https://github.com/DurgamPoojitha/Integrated-Hotel-Analytics-for-Hotel-Success-Amenities-Customer-Experience-.git
cd Integrated-Hotel-Analytics-for-Hotel-Success-Amenities-Customer-Experience-

# 2. Create virtual environment
python3 -m venv .venv

# 3. Activate virtual environment
# On macOS / Linux:
source .venv/bin/activate
# On Windows (Command Prompt / PowerShell):
# .venv\Scripts\activate

# 4. Upgrade pip and install all required dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Download required NLTK corpora (VADER lexicon, tokenizers, wordnet)
python3 -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Option B: Conda / Mamba Environment

```bash
# 1. Create environment from environment.yml
conda env create -f environment.yml

# 2. Activate environment
conda activate hotel-analytics

# 3. Download required NLTK corpora
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

---

## 6. Commands to Reproduce All Notebooks, Models & Application

### 6.1 Launching the Interactive Master Dashboard

The primary deliverable is self-contained and pre-executed:

```bash
# Navigate to the dashboard directory
cd hotel-dashboard-deliverable/dashboard

# Launch JupyterLab or classic Jupyter Notebook
jupyter lab hotel_dashboard.ipynb
# or: jupyter notebook hotel_dashboard.ipynb
```
* **Immediate Interaction**: Scroll down to **Cell 4** to use the interactive dashboard with live widgets, responsive KPI cards, and dynamic filter controls.
* **Full Re-execution**: Click **Kernel → Restart Kernel and Run All Cells**. All data ingestion, model fitting, and widget assembly complete in approximately 30–45 seconds.

### 6.2 Running Automated Testing & Headless Verification

```bash
# From repository root or dashboard folder:
cd hotel-dashboard-deliverable/dashboard

# 1. Run the automated filter testing suite (tests all slicers, dynamic ML, and safety guards)
python3 test_dashboard_filters.py

# 2. Recompile the dashboard notebook programmatically from scratch
python3 generate_dashboard_notebook.py

# 3. Regenerate all 1360x2200 high-resolution evidence screenshots (requires Chrome/Chromium)
python3 generate_screenshots.py
```

### 6.3 Executing Individual Research Notebooks

Each notebook can be run top-to-bottom in headless mode using `jupyter nbconvert`:

```bash
# 1. Review 1: EDA & Predictive Machine Learning
jupyter nbconvert --to notebook --execute "BA_total_Review1_Complete_CORRECTED (2).ipynb" --inplace

# 2. Review 2: Customer Review Text Mining & Sentiment Analysis (Poojitha)
jupyter nbconvert --to notebook --execute "Text_Mining_Hotel_Reviews_Durgam_Poojitha (1) (1).ipynb" --inplace

# 3. Review 2: Integrated Text Mining & Apriori Association Rules (Anupama)
jupyter nbconvert --to notebook --execute "review2_updated.ipynb" --inplace

# 4. CTGAN Synthetic Data Generation (optional; computation-intensive)
jupyter nbconvert --to notebook --execute "hotel_synthetic_data_ctgan_unique_hotel_names (1).ipynb" --inplace

# 5. Master Interactive Dashboard Execution
jupyter nbconvert --to notebook --execute "hotel-dashboard-deliverable/dashboard/hotel_dashboard.ipynb" --inplace
```

---

## 7. Dataset Documentation: Provenance, License & Access

### 7.1 Data Provenance & Construction
1. **Real Scraped Hotel Listings (`data/all_india_1685_unique_hotels.csv`)**:
   * Scraped in October 2026 using SerpApi's Google Hotels engine across 12 major Indian cities.
   * Captured property names, coordinates (latitude/longitude), star ratings, room pricing in INR (₹), total review counts, and pipe-delimited amenity strings.
   * After dropping 21 properties with missing customer ratings, **1,664 usable real hotels** remain.
2. **Review-Level Text Corpus (`data/all_india_hotel_cleaned.csv.zip`)**:
   * Contains 29,987 scraped reviews associated with the real hotels.
   * After listwise exclusion of reviews lacking text or valid star ratings, the clean analytical corpus comprises **26,175 reviews across 1,659 hotels** (Mean: 32.4 words per review; 483,068 total tokens; 16,122 unique vocabulary items).
3. **Synthetic Augmented Hotel Corpus (`hotel_synthetic_data_ctgan_unique_hotel_names (1).ipynb`)**:
   * Generated using Conditional Tabular GAN (CTGAN) trained on the 1,685 real hotel records.
   * Produces **2,887 synthetic hotel records** with statistically consistent distributions for price, star category, review counts, and 15 individual binary amenity flags.
   * Coordinates in synthetic records sit on four fixed reference nodes; therefore, raw latitude and longitude are strictly excluded from predictive models to prevent spatial artifacts.
4. **Unified Master Corpus (`data/combined_real_synthetic_hotels.csv`)**:
   * Combines 1,664 real and 2,887 synthetic records for a total of **4,551 usable properties** with zero duplicate records.

### 7.2 Target Variable & Data Dictionary
* **Target Definition**: "Success" is defined as `customer_rating >= 4.0` (Binary Classification Proxy).
* **Anti-Leakage Safeguard**: `customer_rating` is never used as an input feature in any predictive model.

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `hotel_id` | String | Unique alphanumeric identifier for each hotel property |
| `hotel_name` | String | Property brand name |
| `location` | Categorical | Geographic market hub (12 Indian cities: Delhi, Mumbai, Goa, etc.) |
| `hotel_category` | Categorical | Star classification (`1-star hotel` to `5-star hotel`, or `Unknown`) |
| `price` | Float | Room rate per night in Indian Rupees (₹) |
| `log_price` | Float | Feature-engineered natural logarithm: $\ln(1 + \text{price})$ |
| `customer_rating` | Float | Average observed guest star rating (1.0 to 5.0) |
| `review_count` | Float | Total number of published customer reviews for the property |
| `amenity_count` | Integer | Total count of amenities present across 15 monitored amenities |
| `amenity_*` (15 features) | Binary (0/1) | Flags for Wi-Fi, AC, Parking, Pool, Spa, Bar, Room Service, etc. |
| `success` | Binary (0/1) | Target label: 1 if `customer_rating >= 4.0`, else 0 |
| `data_type` | Categorical | Partition label: `Real` (36.6%) or `Synthetic` (63.4%) |

### 7.3 License, Usage Restrictions & Ethical Compliance
* **Academic Fair Use**: All datasets and code artifacts are published solely for academic research, peer review, and educational benchmarking under Course 23CSE452.
* **Third-Party Terms of Service**:
  * Real hotel data were scraped via SerpApi Google Hotels API in adherence to rate limits and robots.txt protocols for public information.
  * Commercial resale or automated redistributive crawling of these datasets is strictly prohibited under Google Maps / SerpApi Terms of Service.
* **Privacy & PII Protection**:
  * All customer reviews have been scrubbed of personal phone numbers, guest home addresses, and private payment details.
  * Reviewer identities are anonymized and not linked to individual personal identities.
* **Security & Secrets Exclusion**:
  * All hardcoded SerpApi API keys have been **strictly redacted** (`REDACTED_SERPAPI_KEY`) from notebook cell outputs and repository files.
  * To rerun live web scraping in `BA_WebScapring.ipynb`, users must supply their own authorized API key via the `getpass.getpass()` prompt or by setting the environment variable:
    ```bash
    export SERPAPI_API_KEY="your_personal_api_key_here"
    ```

---

## 8. Benchmarking Results & Key Metrics

### 8.1 Machine Learning Benchmark (Holdout Test Set $N = 911$)
Trained on 3,640 hotels across 21 predictors (19 numeric, 2 categorical) using stratified 80/20 train/test split:

| Model Architecture | Hyperparameters | Accuracy | Balanced Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Champion)** | 300 Trees, `max_depth=10`, `min_samples_split=5`, `class_weight='balanced'` | **81.34%** | **0.6443** | **0.6777** | **0.6443** | **0.6571** | **0.8037** | **0.7072** |
| **Decision Tree** | `max_depth=7`, `min_samples_split=10`, `class_weight=None` | **80.35%** | **0.5662** | **0.6297** | **0.5662** | **0.5750** | **0.7724** | **0.6344** |
| **Logistic Regression** | $C=10$, `class_weight='balanced'`, `max_iter=2000` | **68.94%** | **0.6612** | **0.6056** | **0.6612** | **0.6045** | **0.7205** | **0.6955** |

* **Confusion Matrix (Random Forest Benchmark)**: `[[TN=63, FP=104], [FN=66, TP=678]]`
* **Top Feature Importances (Random Forest)**:
  1. `price` (0.198) & `log_price` (0.171)
  2. `review_count` (0.145)
  3. `amenity_count` (0.102)
  4. Convenience amenities: `Free Parking` (0.049), `Accessible` (0.044), `Bar` (0.042), `Free Wi-Fi` (0.038)

### 8.2 Text Mining Sentiment & Pain Point Verification ($N = 26,175$)
* **Sentiment Split**: VADER: **76.3% Positive** | **18.6% Negative** | **5.1% Neutral** (vs. Rating Reference: 73.8% Pos, 21.3% Neg, 4.9% Neu).
* **Classification Evaluation (VADER vs. Rating Ground Truth)**:
  * Overall 3-Class Accuracy: **87.37%** (0.8737) vs. 73.78% Majority Class Baseline
  * Weighted F1: **0.8732** | Cohen’s Kappa: **0.6801** | Spearman Correlation: **0.6157** (Hotel-level compound vs rating: **0.64**)
  * Decisive Reviews ($N = 23,746$, 95.37% coverage): **95.52% Accuracy**, Macro F1 **0.9313**, Negative Precision **0.9503**, ROC-AUC **0.9529**
* **Top Operational Pain Points in Negative Reviews ($N = 4,869$)**:
  1. **Cleanliness / hygiene**: **1,497 mentions (30.7% of all negative reviews)**
  2. **Rude / unhelpful staff**: **992 mentions (20.4%)**
  3. **Broken / poorly maintained**: **747 mentions (15.3%)**
  4. **Bathroom / hot-water issues**: **599 mentions (12.3%)**
  5. **Price / value for money**: **576 mentions (11.8%)**

### 8.3 Apriori Association Rules Top Synergies (33 Amenities, $N = 1,430$)
Mined at Min Support = 0.05, Min Confidence = 0.60, Min Lift = 2.0:

| Rank | Rule Antecedents $\rightarrow$ Consequent | Support | Confidence | Lift | Interpretation |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | `Business center, Pool → Spa` | **6.57%** | **74.02%** | **6.190** | Premium wellness synergy: Spas require business + leisure infrastructure |
| **2** | `Fitness center, Pool → Spa` | **8.18%** | **68.82%** | **5.755** | Core fitness/leisure synergy |
| **3** | `Bar, Fitness center → Spa` | **7.90%** | **66.08%** | **5.526** | Premium hospitality bundle |
| **4** | `Business center, Fitness center → Spa` | **6.92%** | **65.13%** | **5.447** | Executive wellness bundle |
| **5** | `Bar, Pool → Spa` | **8.25%** | **64.84%** | **5.422** | Resort leisure bundle |

---

## 9. Academic Team & Attribution

This project was developed for the **Business Analytics (`23CSE452`)** Capstone Review 2 at Amrita Vishwa Vidyapeetham by **Team 5**:

* **Durgam Poojitha** (`CB.SC.U4CSE23320`) — Lead Contributor: Review 2 Method 1 (Text Mining, NLP, VADER Sentiment Modeling, Customer Pain Points & Thematic Extraction)
* **Anupama Nair** — Contributor: Review 2 Method 2 (Market Basket Analysis & Apriori Association Rule Mining on Standardized Amenities)
* **Sathya Roopan M** — Contributor: Review 1 Predictive Modeling Pipelines & Review 2 Master Interactive Dashboard Integration

---
*For technical questions, verification audits, or reproduction assistance, refer to [`hotel-dashboard-deliverable/dashboard/DOCUMENTATION.md`](hotel-dashboard-deliverable/dashboard/DOCUMENTATION.md) and [`hotel-dashboard-deliverable/dashboard/verification_report.md`](hotel-dashboard-deliverable/dashboard/verification_report.md).*
