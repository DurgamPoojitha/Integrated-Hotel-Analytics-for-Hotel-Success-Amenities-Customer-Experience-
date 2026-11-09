# Hotel Success Intelligence & Review 2 Integration Dashboard
## Business Analytics (23CSE452) • Final Review 2 Documentation

---

### 1. Executive Summary & Project Overview

This project investigates the determinants of hotel success across the Indian hospitality market by analyzing hotel characteristics, room pricing, geographic markets, and amenities. The core analysis blends real hotels scraped via SerpApi with synthetic hotel records. 

* **Target Definition**: "Success" is defined as an observed customer rating of **$\ge 4.0$** (`customer_rating >= 4.0`), creating a binary classification target proxy.
* **Review 2 Integration Mandate**: As specified in the project review brief (`reference/review2_task_sheet.png`), the objective is to **integrate** the team's analytical deliverables into **ONE interactive, decision-support dashboard** organized across **6 tabs**:
  1. **Tab 1 — Final Project Dataset & Export (Rubric Enhancement)**: Interactive tabular view of the final project dataset placed first for immediate data transparency, with hotel name search, pagination (25 rows/page, crash-proof atomic HTML rendering), and dual CSV download capabilities (filtered data and full final corpus) supporting both Google Colab and local Jupyter environments.
  2. **Tab 2 — Overview & Exploratory Data Analysis (EDA)**: Dynamic distributions, pricing dynamics, amenity availability, and success correlations.
  3. **Tab 3 — Predictive Modeling**: Tuned classification benchmarks (Logistic Regression, Decision Tree, and Champion Random Forest) evaluated via authentic pinned Review 1 (`scikit-learn 1.6.1`) test predictions for dynamic test-set slicing.
  4. **Tab 4 — Text Mining (Poojitha)**: Dynamic recalculation engine for VADER sentiment analysis, customer pain points, satisfaction themes, n-grams, dynamic word clouds, and review explorer across real hotel reviews.
  5. **Tab 5 — Association Rule Mining (Anupama)**: Apriori algorithm applied to 33 standardized amenities across 1,430 real hotel transaction vectors discovering high-lift amenity bundles, with amenity search and interactive filtering.
  6. **Tab 6 — Insights & Recommendations (Static Synthesis)**: Immutable executive synthesis grounded in the full empirical corpus.

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
   * *Geographic Note*: Synthetic hotels sit on four fixed reference coordinates; therefore, raw latitude and longitude coordinates are strictly excluded from predictive models and from the final dataset table to prevent spatial artifacts.
4. **Text Mining Scope**:
   Customer review mining applies exclusively to real hotels (26,175 reviews across 1,659 hotels), as synthetic records do not possess customer review text. A clear note in the dashboard explicitly informs the user.

---

### 3. File Architecture & Manifest

```
hotel-dashboard/
├── dashboard/
│   ├── hotel_dashboard_fixed.ipynb    # Current master interactive Jupyter notebook dashboard (6 tabs)
│   ├── hotel_dashboard.ipynb          # Interactive Jupyter notebook dashboard
│   ├── hotel_dashboard_executed.ipynb # Pre-executed static snapshot of master dashboard
│   ├── generate_dashboard_notebook.py  # Automated notebook compiler script
│   ├── requirements.txt               # Complete Python package dependencies
│   ├── run_stage_a.py                 # Teammate code executor & Stage A pipeline
│   ├── verification_report_v3.md      # Quantitative evidence audit report for Tab 6 (19/19 checks passed)
│   ├── verification_report_v2.md      # Exact data match verification audit report (100% exact baseline match)
│   ├── test_dashboard_filters.py      # Automated test suite (filters, Review 1 Option A, text mining, Tab 1 dataset, Tab 6)
│   ├── generate_screenshots.py        # Headless Chrome visual evidence snapshot generator
│   ├── DOCUMENTATION.md               # Master technical and user documentation (this document)
│   └── screenshots/                   # High-resolution visual evidence artifacts
│       ├── dashboard_overview.png     # Full-corpus default landing view (4,551 hotels, full price span)
│       ├── dashboard.png              # Synced landing view
│       ├── filter_interaction_before.png # Baseline state before filters (4,551 hotels, 100.0%)
│       ├── filter_interaction_after.png  # Filtered slice: Coimbatore, 4-star, ₹2k-₹7k (12 hotels)
│       ├── tab1_final_dataset.png     # Tab 1: Final Dataset table, search, paging, dual CSV download
│       ├── tab2_eda.png               # Tab 2: Overview & EDA charts on full corpus
│       ├── tab3_models.png            # Tab 3: Predictive Models & authentic Review 1 benchmarks
│       ├── tab4_text_mining_before.png # Tab 4: Text Mining baseline (all 26,175 reviews)
│       ├── tab4_text_mining.png       # Synced baseline text mining view
│       ├── tab4_text_mining_after.png # Tab 4: Dynamic recomputation on Coimbatore 4-star reviews
│       ├── tab5_association_rules.png # Tab 5: Apriori Association Rules with "Spa" filter active
│       └── tab6_insights_recommendations.png # Tab 6: Executive business insights & prioritized recommendations
├── data/
│   ├── combined_real_synthetic_hotels.csv  # Cleaned base hotels corpus (4,551 usable)
│   ├── review1_test_predictions.csv   # Pinned authentic Review 1 (scikit-learn 1.6.1) test-set predictions (911 rows)
│   ├── review1_rf_feature_importances.csv # Pinned authentic Review 1 RF feature importances (37 features)
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
3. Cell 1 automatically invokes `google.colab.output.enable_custom_widget_manager()` to render all interactive `ipywidgets` and `Plotly` graphs natively.
4. Select **Runtime → Run All**.
5. When clicking CSV export buttons in Tab 1, the dashboard detects Colab and invokes `google.colab.files.download()` seamlessly.

---

### 5. Dashboard Sections & Interactive Capabilities

#### A. Global Filter Panel & Real-Time KPI Row
The top panel provides real-time slicing across the entire corpus:
* **Location Filter**: Multi-select dropdown supporting individual cities (e.g., Coimbatore, Goa, Mumbai, Delhi, Bengaluru, Chennai) or "All Locations".
* **Category Filter**: Multi-select supporting star categories (`5-star hotel`, `4-star hotel`, `3-star hotel`, `2-star hotel`, `1-star hotel`, `Unknown`).
* **Data Type Filter**: Dropdown allowing analysis of `Both`, `Real` only, or `Synthetic` only.
* **Price Range Slider (₹)**: Dual-range slider spanning the full minimum-to-maximum price (**₹392 to ₹61,327**). With no filters touched, the default selection covers **4,551 hotels (100.0%)**, including median-filled records. The slider remains freely adjustable for narrowing, and clicking "Reset All Filters" restores this full span.
* **Rating Range Slider**: Dual-range slider from 0.0 to 5.0.
* **Reset All Filters Button**: Instantly restores default parameters (All Locations, All Categories, Both Data Types, full price range ₹392–₹61,327, rating 0.0–5.0) with state debouncing to prevent flickering.

**Summary KPI Cards (Dynamically Updated)**:
1. **Hotels Selected**: Count of active hotels and percentage of corpus (e.g., 4,551 / 100.0% by default; 12 / 0.3% under Coimbatore 4-star filter).
2. **Success Rate (≥ 4.0)**: Percentage and count of properties achieving customer rating $\ge 4.0$.
3. **Average Price**: Mean and median room rates (₹).
4. **Average Rating**: Mean customer rating out of 5.0 and observed range.
5. **Avg Amenities**: Mean count of amenities available per property across 15 monitored amenities.

---

#### B. Tab 1: Final Project Dataset & Dual CSV Export (Task 2 Rubric Enhancement)
Placed in Position 1 to provide immediate transparency and interactive access to the final project dataset:
* **Readable Dataset Table (Crash-Proof Atomic HTML Rendering)**:
  Displays properties matching current global filters with clean formatting:
  * Columns: `hotel_name`, `location`, `hotel_category`, `data_type`, `price` (₹), `customer_rating` (★), `review_count`, `amenity_count`, `success` (status badge).
  * Strictly excludes raw latitude and longitude coordinates as mandated by the rubric.
  * Rendered inside a persistent `VBox` container (`tab1_dataset_container`) with atomic `widgets.HTML()` updates (`w_tab1_table`), `html.escape()` cell sanitization, upper-bound guards, and cached search filtering—eliminating nested `Output` widget issues across all 183 pages.
* **Interactive Table Search & Paging**:
  * **Hotel Name Search**: Real-time literal case-insensitive text filter to quickly locate specific properties.
  * **Pagination**: 25 hotels per page with Previous / Next navigation buttons and page counter (e.g., Page 1 of 183 for 4,551 hotels; Page 1 of 1 for 12 hotels).
* **Dual CSV Download Buttons**:
  1. **"Download Filtered Data (CSV)"**: Exports the subset of hotels currently matching the global filters.
  2. **"Download Full Final Dataset (CSV)"**: Exports the complete final corpus of 4,551 cleaned hotels regardless of active filter state.
* **Dual-Environment Auto-Detection**:
  * **Google Colab**: Automatically invokes `google.colab.files.download()` to trigger browser downloads directly in Colab notebooks.
  * **Local Jupyter / Lab**: Generates a standard Base64 data-URI download link that opens in any desktop browser.
* **Empty Selection Safeguard**:
  Displays a clear advisory message when filters yield 0 matching hotels instead of an empty or broken table.

---

#### C. Tab 2: Overview & Exploratory Data Analysis (EDA)
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

#### D. Tab 3: Predictive Models (Review 1 Benchmarks & Dynamic Evaluation)
Integrates the tuned predictive models from Review 1, built using 21 predictors (19 numeric including 15 amenities, 2 categorical) on a stratified 80/20 train/test split (with authentic `scikit-learn 1.6.1` Review 1 test-set predictions loaded from `data/review1_test_predictions.csv` and RF importances from `data/review1_rf_feature_importances.csv`, falling back to live pipelines if absent):
* **Champion Model Highlight Card**:
  Highlights the **Tuned Random Forest Classifier** (300 trees, `max_depth=10`, `min_samples_split=5`, `class_weight='balanced'`) alongside its full test set benchmark: Accuracy **81.34%**, Macro Precision **0.6777**, Macro Recall **0.6443**, Macro-F1 **0.6571**, Weighted-F1 **0.8037**, ROC-AUC **0.6990**.
* **Dynamic Performance Evaluation Table**:
  Evaluates all three models (Random Forest, Decision Tree, Logistic Regression) on the **filtered test set subset**, reporting Subset Test $N$, Accuracy, Macro Precision, Macro Recall, Macro F1, Weighted F1, ROC-AUC, and Baseline Review 1 Benchmark Accuracy & F1. Under full-corpus default state ($N=911$), live evaluation matches Review 1 benchmarks cleanly (`0.8134` accuracy, `0.6990` ROC-AUC).
* **Sample Size & Single-Class Safeguards**:
  * Displays a prominent warning if the filtered test subset contains fewer than 50 hotels ($n < 50$), noting statistical variance.
  * Disables macro metrics gracefully if the subset contains only a single class.
* **Model Visualizations**:
  * Grouped Performance Bar Chart across Accuracy, Macro Precision, Macro Recall, and Macro F1.
  * Side-by-side Confusion Matrices for Logistic Regression, Decision Tree, and Random Forest.
  * ROC Curves with AUC legend scores.
  * Top 15 Random Forest Feature Importances horizontal bar chart (`price`, `log_price`, `review_count`, `amenity_count`, `amenity_free_parking`, etc.).

---

#### E. Tab 4: Text Mining Integration (Dynamic Recomputation Engine)
Upgraded from static precomputed figures to a **high-speed dynamic recomputation engine** that responds directly to global filters by subsetting reviews belonging to active real hotels:
* **Filter Scope & Feedback Banner**:
  Clearly displays the active real hotel count and selected review count (e.g., 26,175 reviews across 1,659 hotels unfiltered; 98 reviews across 6 real hotels under Coimbatore 4-star filter).
* **Dynamic Charts & Metrics**:
  1. **VADER Sentiment Distribution `[Dynamic]`**: Donut chart recalculating Positive, Negative, and Neutral percentages from filtered reviews.
  2. **Star Rating vs. VADER Sentiment `[Dynamic]`**: Stacked bar chart examining classification alignment across customer star ratings for the filtered reviews.
  3. **Sentiment-vs-Rating Evaluation Metrics `[Dynamic]`**: Bar chart recalculating Accuracy, Weighted F1, Cohen's Kappa, Macro Precision, Macro Recall, and Macro F1 against star ratings.
  4. **VADER vs. Rating Confusion Matrix `[Dynamic]`**: 3×3 heatmap recomputed on the filtered reviews.
  5. **Top 15 Frequent Words & Top 10 Bigrams `[Dynamic]`**: Recalculated from tokenized, lemmatized review text using teammate stopword lists.
  6. **Dynamic Word Cloud `[Dynamic]`**: Regenerated on the fly from filtered word frequencies using WordCloud and embedded dynamically via Base64.
  7. **Top Customer Pain Points `[Dynamic]`**: Operational complaint drivers dynamically classified using teammate regex rules, reporting count and percentage of filtered negative reviews.
  8. **Positive & Negative Experience Themes `[Dynamic]`**: Dynamic percentage of positive / negative mentions across 16 operational themes.
  9. **Amenity Aspect Sentiment `[Dynamic]`**: Customer sentiment specifically when discussing amenities, recomputed at sentence level.
  10. **Key Findings Summary `[Dynamic]`**: Dynamically summarizes sentiment split, star-rating agreement, Macro-F1, top pain points, and top delight drivers (Cohen's Kappa is charted in the evaluation metrics bar chart and omitted from the takeaways bullet box).
* **Interactive Review Explorer Table (Task 4)**:
  Searchable, paged review table (using an atomic `widgets.HTML()` container with `html.escape()` sanitization outside `Output` widgets) displaying Hotel Name, City, Category, Star Rating, VADER Sentiment badge, and Review Text Excerpt with text search and pagination (15 reviews per page).
* **Safeguards & Honest Labels**:
  * Displays a yellow advisory banner when fewer than 100 reviews remain.
  * Displays an empty-state message when zero real hotels/reviews match.
  * Disables single-class metrics gracefully without throwing errors.
  * Every dynamic component is honestly labeled `[Dynamic]`. Any static benchmark is explicitly labeled `[Static]`.

---

#### F. Tab 5: Apriori Association Rules Integration (Anupama's Deliverables)
Mines transactional co-occurrences across 33 standardized amenities from 1,430 real hotel listings:
* **Interactive Association Rule Controls**:
  * **Min Support Slider**: Adjustable from 0.01 to 0.20 (default 0.05).
  * **Min Confidence Slider**: Adjustable from 0.30 to 0.95 (default 0.60).
  * **Min Lift Slider**: Adjustable from 1.0 to 6.5 (default 2.0).
  * **Amenity Search Filter**: Text input to filter rules by amenity (e.g., typing "Spa" filters to rules containing Spa in antecedents or consequents).
  * **Dynamic mlxtend Recomputation Toggle**: Allows real-time re-mining on filtered hotels with built-in $n < 50$ sample size safety guards.
* **Top Standardized Amenity Support**: Horizontal bar ranking overall support (Wi-Fi 95.4%, Air conditioning 91.2%, Room service 85.0%, Full-service laundry 80.6%, Breakfast 80.2%, Parking 78.2%).
* **Top Rules by Lift**:
  Visualizes the strongest complementary amenity bundles:
  1. **Business center, Pool → Spa** (Lift: **6.190**, Conf: **74.02%**, Supp: **6.57%**)
  2. **Fitness center, Pool → Spa** (Lift: **5.755**, Conf: **68.82%**, Supp: **8.18%**)
  3. **Bar, Fitness center → Spa** (Lift: **5.526**, Conf: **66.08%**, Supp: **7.90%**)
  4. **Business center, Fitness center → Spa** (Lift: **5.447**, Conf: **65.13%**, Supp: **6.92%**)
* **Interactive Scatter Plot**: Support vs. Confidence with marker size and color mapped to Lift.
* **Top 15 Ranked Rules Table**: Tabular view of antecedents, consequents, support, confidence, and lift.
* **Strategic Bundling Insights**: Actionable recommendations for full-service amenity investments.

---

#### G. Tab 6: Insights & Recommendations (Static Strategic Synthesis)
Provides an executive summary synthesizing findings across all four analytical domains (EDA, Predictive Modeling, Review Text Mining, and Apriori Association Rules).
* **Static Content Guarantee**:
  * Bound to `out_tab6 = widgets.Output()` and rendered ONCE at notebook startup via `render_tab6()`.
  * Completely independent of interactive global filters and filter resets to provide an immutable strategic baseline derived from the full project corpus.
  * Explicit methodological banner: *"Static content: based on the full project corpus; not affected by the global filters. Findings are associations, not causes. Success is a rating-based proxy (rating >= 4.0). About 63% of hotels are synthetic; text mining covers real hotels only."*
* **Section A: Business Insights (Two-Column Responsive Grid)**:
  1. **Price and Amenity Count Alone Say Little About Guest Ratings** (Weak linear correlations: combined price $r = 0.086$, amenity count $r = -0.134$; real price $r = 0.182$, amenity $r = 0.112$). *Source: EDA tab*.
  2. **Operational Failures Are the Biggest Sources of Dissatisfaction** (Cleanliness: 1,497 negative reviews / 30.7%, Staff demeanor: 992 / 20.4%; Bedding 55.7%, Bathrooms 52.0%, AC 50.8% negative). *Source: Text Mining tab*.
  3. **Customer Perception of Value Depends on the Experience Delivered** (29.9% of 3,565 price/value reviews negative vs 18.6% baseline). *Source: Text Mining tab*.
  4. **Hotel Performance Varies Across Categories and Markets** (1-star 69.7% to 5-star 92.0%; Goa 88.6% highest vs Hyderabad 72.0% lowest). *Source: EDA tab*.
  5. **Amenity Strategy Requires Complementarity, Not Just Quantity** (Baseline essentials: Wi-Fi 95.38%, AC 91.19%; Triad bundle: Business center + Pool → Spa, Lift 6.190, Conf 74.02%). *Source: Association Rules tab*.
  6. **Predictive Analytics Can Support Better Decisions** (Tuned RF Accuracy 81.34%, Macro-F1 0.6571; Top features: price, log_price, review_count, amenity_count). *Source: Predictive Models tab*.
* **Section B: Business Recommendations (Prioritized Action Cards)**:
  1. **Prioritize Operational Quality** (`[High Priority]`, Red badge) — Housekeeping checklists, preventive maintenance for bathrooms/AC/bedding. *Measure: Monthly cleanliness and maintenance complaint rates*.
  2. **Strengthen Staff Service and Complaint Resolution** (`[High Priority]`, Red badge) — Front-desk communication training, empowered service recovery. *Measure: Staff negative sentiment and complaint frequency*.
  3. **Align Room Pricing with Perceived Value** (`[High Priority]`, Red badge) — Transparent pricing, condition benchmarking. *Measure: Negative sentiment in price/value reviews*.
  4. **Ensure Reliability of Essential Amenities** (`[High Priority]`, Red badge) — Maintain Wi-Fi, AC, parking (the 3 lowest net-sentiment amenities). *Measure: Amenity complaint logs and failure reports*.
  5. **Invest in Amenities Strategically** (`[Medium Priority]`, Blue badge) — Bundle premium facilities (e.g. Spa) only with supporting infrastructure (Business center, Pool). *Measure: Amenity utilization and incremental revenue*.
  6. **Adopt Data-Driven Performance Monitoring** (`[Medium Priority]`, Blue badge) — Continuous tracking via dashboard and ML exception flagging. *Measure: Rating trends, review volume, and negative sentiment over time*.
* **Overall Strategy Box**:
  * Emphasizes the foundational principle: *"Fix operational weaknesses first, deliver value consistent with pricing, maintain essential amenities, and invest in differentiated offerings only when supported by evidence."*

---

### 6. Visual Evidence Catalog

All visual evidence has been captured using headless Google Chrome at 1360×2200 resolution and saved to [`dashboard/screenshots/`](file:///d:/Semesters/Semester%207/Business%20Analytics/Project/Review%202/hotel-dashboard/dashboard/screenshots/):

| Screenshot File | Section / Tab | Verification Evidence Provided |
|:---|:---|:---|
| [`dashboard_overview.png`](file:///d:/Semesters/Semester%207/Business%20Analytics/Project/Review%202/hotel-dashboard/dashboard/screenshots/dashboard_overview.png) / [`dashboard.png`](file:///d:/Semesters/Semester%207/Business%20Analytics/Project/Review%202/hotel-dashboard/dashboard/screenshots/dashboard.png) | **Master Dashboard Landing View** | Verifies full 4,551 hotel corpus (100.0%), full price span (₹392 – ₹61,327), 81.7% success rate, ₹4,077 avg price, 6-tab navigation bar (starting with `1. Final Dataset`), and Tab 1 table/exports. |
| [`filter_interaction_before.png`](file:///d:/Semesters/Semester%207/Business%20Analytics/Project/Review%202/hotel-dashboard/dashboard/screenshots/filter_interaction_before.png) | **Before Filtering (Baseline)** | Baseline state showing 100.0% corpus (4,551 hotels), 81.7% success rate, ₹4,077 avg price, 10.5 avg amenities, and full distribution charts. |
| [`filter_interaction_after.png`](file:///d:/Semesters/Semester%207/Business%20Analytics/Project/Review%202/hotel-dashboard/dashboard/screenshots/filter_interaction_after.png) | **After Filtering (Demo Scenario)** | Verified active slice: Location="Coimbatore", Category="4-star hotel", Price=₹2,000–₹7,000 ($n=12$ hotels, 91.7% success rate, ₹4,890 avg price, 17.7 avg amenities, dynamically updated charts). *Note: Replaces the previous Goa 5-star scenario to align with presentation slides.* |
| [`tab1_final_dataset.png`](file:///d:/Semesters/Semester%207/Business%20Analytics/Project/Review%202/hotel-dashboard/dashboard/screenshots/tab1_final_dataset.png) | **Tab 1: Final Dataset** | Verifies final project dataset table (columns: name, location, category, data type, price, rating, reviews, amenities, success), search box, pagination (showing Page 1 of 183 / 4,551), and dual CSV download buttons. |
| [`tab2_eda.png`](file:///d:/Semesters/Semester%207/Business%20Analytics/Project/Review%202/hotel-dashboard/dashboard/screenshots/tab2_eda.png) | **Tab 2: Overview & EDA** | Verifies full 4,551 hotel corpus, 5 KPI cards, and all 10 coordinated charts. |
| [`tab3_models.png`](file:///d:/Semesters/Semester%207/Business%20Analytics/Project/Review%202/hotel-dashboard/dashboard/screenshots/tab3_models.png) | **Tab 3: Predictive Models** | Verifies Champion Random Forest card (Acc 81.34%, ROC-AUC 0.6990), aligned comparison table, 3 confusion matrices, ROC curves, and feature importances. |
| [`tab4_text_mining_before.png`](file:///d:/Semesters/Semester%207/Business%20Analytics/Project/Review%202/hotel-dashboard/dashboard/screenshots/tab4_text_mining_before.png) / [`tab4_text_mining.png`](file:///d:/Semesters/Semester%207/Business%20Analytics/Project/Review%202/hotel-dashboard/dashboard/screenshots/tab4_text_mining.png) | **Tab 4: Text Mining (Unfiltered)** | Verifies baseline text mining metrics across all 26,175 reviews (76.3% positive, 18.6% negative, 5.1% neutral), top pain points, experience themes, evaluation metrics chart (with Cohen's Kappa), Takeaways box, and clean word cloud. |
| [`tab4_text_mining_after.png`](file:///d:/Semesters/Semester%207/Business%20Analytics/Project/Review%202/hotel-dashboard/dashboard/screenshots/tab4_text_mining_after.png) | **Tab 4: Text Mining (Dynamic Filtered)** | Verifies dynamic recomputation on the Coimbatore 4-star slice (6 real hotels, 98 reviews, 84.7% positive, 13.3% negative, shifted pain points, and dynamically updated word cloud and confusion matrix). |
| [`tab5_association_rules.png`](file:///d:/Semesters/Semester%207/Business%20Analytics/Project/Review%202/hotel-dashboard/dashboard/screenshots/tab5_association_rules.png) | **Tab 5: Association Rules** | Verifies Apriori slider controls with Amenity Search="Spa" active, top lift bundle (*Business center + Pool → Spa*, Lift: 6.190), scatter plot, and ranked table. |
| [`tab6_insights_recommendations.png`](file:///d:/Semesters/Semester%207/Business%20Analytics/Project/Review%202/hotel-dashboard/dashboard/screenshots/tab6_insights_recommendations.png) | **Tab 6: Insights & Recommendations** | Verifies static executive summary synthesis across EDA, predictive models, text mining, and association rules: 6 business insights cards, 6 prioritized recommendations cards, and overall strategy roadmap banner. |

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

---

### 8. Quantitative Audit & Verification Reports

To ensure absolute academic rigor and 100% empirical consistency across all dashboard components:
* **Verification Report v2** (`dashboard/verification_report_v2.md`): Validated the core baselines (4,551 hotels, 3,718 successful, 26,175 reviews, 154 strong Apriori rules, Review 1 model benchmarks).
* **Verification Report v3** (`dashboard/verification_report_v3.md`): Audited all 19 quantitative claims in Tab 6 (*Insights & Recommendations*), certifying that every correlation coefficient, percentage, frequency count, and model metric matches the underlying ground-truth data with 100% accuracy.
* **Automated Test Suite** (`dashboard/test_dashboard_filters.py`): Comprehensive automated checks covering baseline integrity, authentic Review 1 Option A predictions (`0.8134` / `0.6990`), individual filters, combination filters, edge cases, reset mechanics, dynamic text mining recalculation, Tab 1 pagination/search/exports, notebook structural invariants, and Tab 6 static isolation.
