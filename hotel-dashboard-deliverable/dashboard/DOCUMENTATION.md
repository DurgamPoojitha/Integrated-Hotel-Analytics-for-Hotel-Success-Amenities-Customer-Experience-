# Hotel Success Intelligence & Review 2 Integration Dashboard
## Business Analytics (23CSE452) • Final Review 2 Documentation

---

### 1. Executive Summary & Project Overview

This project investigates the determinants of hotel success across the Indian hospitality market by analyzing hotel characteristics, room pricing, geographic markets, and amenities. The core analysis blends real hotels scraped via SerpApi with synthetic hotel records. 

* **Target Definition**: "Success" is defined as an observed customer rating of **$\ge 4.0$** (`customer_rating >= 4.0`), creating a binary classification target proxy.
* **Review 2 Integration Mandate**: As specified in the project review brief (`reference/review2_task_sheet.png`), Sathya's role is to **integrate** the team's disparate analytical deliverables into **ONE interactive, decision-support dashboard**. Rather than redoing individual analyses, the dashboard unifies:
  1. **Exploratory Data Analysis (EDA)**: Dynamic distributions, pricing dynamics, amenity availability, and success correlations.
  2. **Predictive Modeling**: Tuned classification benchmarks (Logistic Regression, Decision Tree, and the Champion Random Forest model) with dynamic test-set evaluation.
  3. **Text Mining (Poojitha)**: VADER sentiment analysis, customer pain points, satisfaction themes, n-grams, and clean word clouds across 26,175 real hotel reviews.
  4. **Association Rule Mining (Anupama)**: Apriori algorithm applied to 33 standardized amenities across 1,430 real hotel transaction vectors discovering high-lift amenity bundles.

> **Zero Fabricated Values Policy**: Every visual element, KPI card, and metric shown in the dashboard is computed directly from verified project files. No sample, placeholder, or mock numbers are used.

---

### 2. Methodological & Analytical Disclaimers

1. **Association vs. Causation**:
   All statistical correlations, feature importances, and association rules presented in the dashboard represent **empirical co-occurrences and observational associations**. They do **not** imply that adding a specific amenity or altering price will causally increase customer ratings.
2. **Rating-Based Success Proxy**:
   "Success" is an analytical proxy defined as `customer_rating >= 4.0`. To strictly prevent target leakage, `customer_rating` is never used as an input feature in any predictive model.
3. **Data Composition**:
   The primary dataset comprises **4,551 usable hotels** (after dropping 21 records with missing customer ratings):
   * **Real Scraped Hotels**: 1,664 hotels (36.6%) scraped via SerpApi with actual customer reviews and verified amenities.
   * **Synthetic Hotels**: 2,887 hotels (63.4%) generated to augment market scenarios.
   * *Geographic Note*: Synthetic hotels sit on four fixed reference coordinates; therefore, raw latitude and longitude coordinates are strictly excluded from predictive models to prevent spatial artifacts.
4. **Text Mining Scope**:
   Customer review mining applies exclusively to real hotels (26,175 reviews across 1,659 hotels), as synthetic records do not possess customer review text.

---

### 3. File Architecture & Manifest

```
hotel-dashboard/
├── dashboard/
│   ├── hotel_dashboard.ipynb          # Master interactive Jupyter notebook dashboard
│   ├── generate_dashboard_notebook.py  # Automated notebook compiler script
│   ├── requirements.txt               # Complete Python package dependencies
│   ├── run_stage_a.py                 # Teammate code executor & Stage A pipeline
│   ├── verification_report.md         # Exact data match verification audit report
│   ├── test_dashboard_filters.py      # Automated test suite for filter operations & dynamic ML
│   ├── generate_screenshots.py        # Headless Chrome visual evidence snapshot generator
│   ├── DOCUMENTATION.md               # Master technical and user documentation (this document)
│   └── screenshots/                   # High-resolution visual evidence artifacts
│       ├── tab1_eda.png
│       ├── tab2_models.png
│       ├── tab3_text_mining.png
│       ├── tab4_association_rules.png
│       ├── filter_interaction_before.png
│       └── filter_interaction_after.png
├── data/
│   ├── combined_real_synthetic_hotels.csv  # Cleaned base hotels corpus (4,551 usable)
│   ├── text_mining_outputs/           # 36 Stage A text mining outputs (Poojitha)
│   └── apriori/                       # 4 Stage A Apriori outputs (Anupama)
└── reference/
    ├── BA_total_Review1_Complete_CORRECTED.ipynb
    ├── FInal_PPT_Review_1.pptx
    ├── review2_task_sheet.png
    └── review2_updated.ipynb
```

---

### 4. Setup & Execution Guide

#### Local Environment Execution
1. Ensure Python 3.10+ is installed.
2. Install project dependencies:
   ```bash
   pip install -r dashboard/requirements.txt
   ```
3. Launch JupyterLab or classic Jupyter Notebook:
   ```bash
   jupyter lab
   ```
4. Open [`dashboard/hotel_dashboard.ipynb`](file:///d:/Semesters/Semester%207/Business%20Analytics/Project/Review%202/hotel-dashboard/dashboard/hotel_dashboard.ipynb) and select **Kernel → Restart Kernel and Run All Cells**.

#### Google Colab Execution
The dashboard is built to detect Google Colab environments automatically:
1. Upload the project folder to Google Drive or clone the repository into `/content/hotel-dashboard`.
2. Open [`dashboard/hotel_dashboard.ipynb`](file:///d:/Semesters/Semester%207/Business%20Analytics/Project/Review%202/hotel-dashboard/dashboard/hotel_dashboard.ipynb) in Colab.
3. Cell 1 will automatically invoke `google.colab.output.enable_custom_widget_manager()` to render all interactive `ipywidgets` and `Plotly` graphs natively.
4. Select **Runtime → Run All**.

---

### 5. Dashboard Sections & Interactive Capabilities

#### A. Global Filter Panel & Real-Time KPI Row
The top panel provides real-time slicing across the entire corpus:
* **Location Filter**: Multi-select dropdown supporting individual cities (e.g., Goa, Mumbai, Delhi, Bengaluru, Chennai) or "All Locations".
* **Category Filter**: Multi-select supporting star categories (`5-star hotel`, `4-star hotel`, `3-star hotel`, `2-star hotel`, `1-star hotel`, `Unknown`).
* **Data Type Filter**: Dropdown allowing analysis of `Both`, `Real` only, or `Synthetic` only.
* **Price Range Slider (₹)**: Dual-range slider from ₹360 to ₹25,000+ (99th percentile cap).
* **Rating Range Slider**: Dual-range slider from 0.0 to 5.0.
* **Reset Filters Button**: Instantly restores default parameters with state debouncing to prevent flickering.

**Summary KPI Cards (Dynamically Updated)**:
1. **Hotels Selected**: Count of active hotels and percentage of corpus (e.g., 4,551 / 100.0%).
2. **Success Rate (≥ 4.0)**: Percentage and count of properties achieving customer rating $\ge 4.0$.
3. **Average Price**: Mean and median room rates (₹).
4. **Average Rating**: Mean customer rating out of 5.0 and observed range.
5. **Avg Amenities**: Mean count of amenities available per property across 15 monitored amenities.

---

#### B. Tab 1: Exploratory Data Analysis (EDA)
Presents 10 coordinated Plotly visualizations that update dynamically with the filter panel:
* **Chart 1: Observed Success Distribution**: Interactive donut chart breaking down successful (81.7%) vs. unsuccessful properties (18.3%).
* **Chart 2: Data Composition**: Real (1,664) vs. synthetic (2,887) hotel counts.
* **Chart 3: Customer Rating Distribution**: Overlaid histogram for real and synthetic hotels with a dashed vertical threshold line at 4.0.
* **Chart 4: Success Rate by Hotel Category**: Horizontal bar chart showing success rates (5-star: 92.0%, 4-star: 89.0%, down to 1-star: 69.7%) with hotel counts ($n$).
* **Chart 5: Top 15 Locations**: Geographic distribution of properties across major tourist and business hubs.
* **Chart 6: Price vs. Review Count Relationships**: Scatter plot colored by success status highlighting volume-price dynamics.
* **Chart 7: Prevalence of Monitored Amenities**: Market penetration of 15 individual amenities (e.g., Free Wi-Fi 88.8%, Room Service 85.0%, Parking 78.2%).
* **Chart 8: Success Rate Difference (With vs. Without Amenity)**: Diverging bar chart showing the percentage point (pp) success difference associated with each amenity.
* **Chart 9: Amenity Count Distribution by Success**: Box plot comparing total amenities between successful and unsuccessful hotels.
* **Chart 10: Multi-Feature Correlation Matrix Heatmap**: Empirical correlation matrix between price, log price, review count, amenity count, and success (rating excluded to preserve integrity).

---

#### C. Tab 2: Predictive Models (Review 1 Benchmarks & Dynamic Evaluation)
Integrates the tuned predictive models from Review 1, built using 21 predictors (19 numeric including 15 amenities, 2 categorical) on a stratified 80/20 train/test split:
* **Champion Model Highlight Card**:
  Highlights the **Tuned Random Forest Classifier** (300 trees, `max_depth=10`, `min_samples_split=5`, `class_weight='balanced'`) alongside its full test set benchmark: Accuracy **81.34%**, Macro Precision **0.6777**, Macro Recall **0.6443**, Macro-F1 **0.6571**, Weighted-F1 **0.8037**, ROC-AUC **0.7072**.
* **Dynamic Performance Evaluation Table**:
  Evaluates all three models (Random Forest, Decision Tree, Logistic Regression) on the **filtered test set subset**, reporting:
  * Subset Test $N$
  * Accuracy
  * Macro Precision
  * Macro Recall
  * Macro F1
  * Weighted F1
  * ROC-AUC
  * Baseline Review 1 Benchmark Accuracy & F1
* **Sample Size & Single-Class Safeguards**:
  * Displays a prominent warning if the filtered test subset contains fewer than 50 hotels ($n < 50$), noting statistical variance.
  * Disables macro metrics gracefully if the subset contains only a single class.
* **Model Visualizations**:
  * Grouped Performance Bar Chart across Accuracy, Macro Precision, Macro Recall, and Macro F1.
  * Side-by-side Confusion Matrices for Logistic Regression (`[[103, 64], [219, 525]]`), Decision Tree (`[[32, 135], [44, 700]]`), and Random Forest.
  * ROC Curves with AUC legend scores.
  * Top 15 Random Forest Feature Importances horizontal bar chart (`price`, `log_price`, `review_count`, `amenity_count`, `amenity_free_parking`, etc.).

---

#### D. Tab 3: Text Mining Integration (Poojitha's Deliverables)
Synthesizes customer sentiment, thematic drivers, and pain points across 26,175 reviews of real hotels:
* **Dynamic VADER Sentiment Distribution**: Donut chart displaying Positive (76.3%), Negative (18.6%), and Neutral (5.1%) reviews, dynamically filtered to selected hotels.
* **Rating Sentiment vs. VADER Classification**: Stacked bar chart examining classification alignment across customer star ratings.
* **Top Customer Pain Points**:
  Identifies the top operational complaint drivers in negative reviews:
  1. *Cleanliness / hygiene* (30.7% of negative mentions, 1,497 reviews)
  2. *Rude / unhelpful staff* (20.4%, 992 reviews)
  3. *Broken / poorly maintained facilities* (15.3%, 747 reviews)
  4. *Bathroom / hot-water issues* (12.3%, 599 reviews)
  5. *Price / value for money* (11.8%, 576 reviews)
* **Positive Experience Themes**:
  Highlights the highest-rated delight themes:
  * *Location* (84.9% positive)
  * *Food* (82.9% positive)
  * *Breakfast* (79.3% positive)
  * *Service* (79.0% positive)
  * *Staff* (77.9% positive)
* **Word & Bigram Frequencies**: Top 15 frequent unigrams and top 10 bigrams (*good service*, *room clean*, *well maintain*, *highly recommend*).
* **Amenity Mention Sentiment**: Customer satisfaction sentiment specifically when discussing amenities (Spa 84.9% positive, Pet-friendly 72.7%, down to AC complaints at 37.5%).
* **VADER Evaluation Matrix**: 3×3 confusion matrix and evaluation metrics ($87.37\%$ agreement with star ratings, $92.73\%$ on forced binary signals).
* **Embedded Corpus Word Cloud**: High-resolution clean word cloud (`word_cloud_clean.png`).

---

#### E. Tab 4: Apriori Association Rules Integration (Anupama's Deliverables)
Mines transactional co-occurrences across 33 standardized amenities from 1,430 real hotel listings:
* **Interactive Association Rule Controls**:
  * **Min Support Slider**: Adjustable from 0.01 to 0.20 (default 0.05).
  * **Min Confidence Slider**: Adjustable from 0.30 to 0.95 (default 0.60).
  * **Min Lift Slider**: Adjustable from 1.0 to 6.5 (default 2.0).
  * **Amenity Search Filter**: Text input to filter rules by amenity (e.g., "Spa", "Pool").
  * **Dynamic mlxtend Recomputation Toggle**: Allows real-time re-mining on filtered hotels with built-in $n < 50$ sample size safety guards.
* **Top Standardized Amenity Support**: Horizontal bar ranking overall support (Wi-Fi 95.4%, Air conditioning 91.2%, Room service 85.0%, Full-service laundry 80.6%, Breakfast 80.2%, Parking 78.2%).
* **Top Rules by Lift**:
  Visualizes the strongest complementary amenity bundles:
  1. **Business center, Pool → Spa** (Lift: **6.190**, Conf: **74.02%**, Supp: **6.57%**)
  2. **Fitness center, Pool → Spa** (Lift: **5.755**, Conf: **68.82%**, Supp: **8.18%**)
  3. **Bar, Fitness center → Spa** (Lift: **5.526**, Conf: **66.08%**, Supp: **7.90%**)
  4. **Business center, Fitness center → Spa** (Lift: **5.447**, Conf: **65.13%**, Supp: **6.92%**)
* **Interactive Scatter Plot**: Support vs. Confidence with marker size and color mapped to Lift.
* **Top 15 Ranked Rules Table**: Clean tabular view of antecedents, consequents, support, confidence, and lift.
* **Strategic Bundling Insights**: Actionable recommendations for full-service amenity investments.

---

### 6. Visual Evidence Catalog

All visual evidence has been captured using headless Google Chrome at 1360×2200 resolution and saved to `dashboard/screenshots/`:

| Screenshot File | Section / Tab | Verification Evidence Provided |
|:---|:---|:---|
| [`tab1_eda.png`](file:///d:/Semesters/Semester%207/Business%20Analytics/Project/Review%202/hotel-dashboard/dashboard/screenshots/tab1_eda.png) | **Tab 1: EDA** | Verifies full 4,551 hotel corpus, 5 KPI cards, and all 10 coordinated charts. |
| [`tab2_models.png`](file:///d:/Semesters/Semester%207/Business%20Analytics/Project/Review%202/hotel-dashboard/dashboard/screenshots/tab2_models.png) | **Tab 2: Models** | Verifies Champion Random Forest card, comparison table with precision/recall, 3 confusion matrices, ROC curves, and feature importances. |
| [`tab3_text_mining.png`](file:///d:/Semesters/Semester%207/Business%20Analytics/Project/Review%202/hotel-dashboard/dashboard/screenshots/tab3_text_mining.png) | **Tab 3: Text Mining** | Verifies VADER sentiment donut, rating alignment, pain points, themes, n-grams, amenity sentiments, confusion matrix, and word cloud. |
| [`tab4_association_rules.png`](file:///d:/Semesters/Semester%207/Business%20Analytics/Project/Review%202/hotel-dashboard/dashboard/screenshots/tab4_association_rules.png) | **Tab 4: Apriori** | Verifies Apriori slider controls, 33-amenity support ranking, top lift rules (Business center + Pool → Spa), scatter plot, and ranked table. |
| [`filter_interaction_before.png`](file:///d:/Semesters/Semester%207/Business%20Analytics/Project/Review%202/hotel-dashboard/dashboard/screenshots/filter_interaction_before.png) | **Before Filtering** | Baseline state showing 100.0% corpus (4,551 hotels), 81.7% success rate, ₹4,077 avg price, and full distribution charts. |
| [`filter_interaction_after.png`](file:///d:/Semesters/Semester%207/Business%20Analytics/Project/Review%202/hotel-dashboard/dashboard/screenshots/filter_interaction_after.png) | **After Filtering** | Verified active slice: Location="Goa", Category="5-star hotel", Price=₹5k-₹25k ($n=34$, 97.1% success, ₹9,541 avg price, 22.4 avg amenities, dynamically updated charts). |

---

### 7. Strategic Business Insights for Hotel Operators

1. **Amenity Bundling Strategy**:
   * *The Spa Synergistic Triad*: Standalone spas exhibit weak standalone viability. However, when paired with a **Pool** and **Business Center**, the likelihood of spa presence jumps by **6.19×** (Confidence 74.02%). Operators planning premium wellness investments must co-locate fitness and business infrastructure.
2. **Operations & Pain Point Elimination**:
   * While positive reviews frequently praise *Location* (84.9%) and *Food* (82.9%), ratings collapse primarily due to basic operational hygiene failures: *Cleanliness* accounts for **30.7%** of negative feedback, and *Staff Demeanor* accounts for **20.4%**. Eliminating hygiene complaints provides the highest return on customer satisfaction.
3. **Pricing & Feature Predictive Power**:
   * Machine learning feature importance reveals that room pricing (`price`, `log_price`) and customer review velocity (`review_count`) are the strongest empirical correlates of hotel success, followed by total amenity breadth (`amenity_count`) and essential convenience amenities like free parking.
4. **Category Differentiation**:
   * Higher star ratings correlate strongly with higher success rates (5-star: 92.0%, 4-star: 89.0%, 3-star: 79.1%), indicating that brand standard enforcement and comprehensive service amenities reliably differentiate successful hospitality ventures.
