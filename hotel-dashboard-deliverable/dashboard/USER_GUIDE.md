# Hotel Success Intelligence & Decision Support Dashboard
## Comprehensive User Guide, Feature Interaction Matrix & Interpretation Manual
### Course: Business Analytics (23CSE452) • Final Review 2 Deliverable

---

## 1. Introduction & Dashboard Architecture

The **Hotel Success Intelligence & Decision Support Dashboard** is an integrated analytical platform developed to examine the key operational, pricing, geographic, and amenity-driven factors determining hotel success across the Indian hospitality market.

Built natively in Jupyter Notebook using Python, `ipywidgets`, and Plotly, the dashboard consolidates four distinct analytical workstreams into a single interactive decision-support tool:
1. **Exploratory Data Analysis (EDA)**: Market structure, pricing dynamics, geographic penetration, and amenity distributions.
2. **Predictive Modeling**: Supervised classification models (Random Forest champion, Decision Tree, Logistic Regression) with dynamic subset evaluation.
3. **Text Mining & Sentiment Analytics**: Dynamic NLP engine analyzing 26,175 customer reviews across real hotels to uncover sentiment polarity, operational pain points, and customer delight themes.
4. **Association Rule Mining (Apriori)**: Unsupervised market-basket analysis across 33 standardized amenities to discover high-lift complementary amenity bundles.
5. **Final Project Dataset & Dual CSV Export**: Direct access to the project corpus with real-time text searching, pagination, and multi-environment data export.

```
+----------------------------------------------------------------------------------------------------+
|                         HOTEL SUCCESS INTELLIGENCE & DECISION SUPPORT DASHBOARD                     |
+----------------------------------------------------------------------------------------------------+
| [GLOBAL FILTER PANEL]: Location | Category | Data Type | Price Range (₹) | Rating Range | Reset    |
+----------------------------------------------------------------------------------------------------+
| [REAL-TIME KPI ROW]: Hotels Selected | Avg Rating | Avg Price | Avg Reviews | Success Rate | Amens |
+----------------------------------------------------------------------------------------------------+
| [TAB NAVIGATION BAR]:                                                                              |
|   1. Overview & EDA | 2. Predictive Models | 3. Text Mining | 4. Association Rules | 5. Final Data |
+----------------------------------------------------------------------------------------------------+
| [ACTIVE TAB CONTENT AREA]: Dynamic Plotly Visualizations, Model Tables, NLP Graphs & Data Tables  |
+----------------------------------------------------------------------------------------------------+
```

### Core Analytical Definitions & Disclaimers

> [!IMPORTANT]
> **Target Definition ("Success")**: A hotel is categorized as **Successful (`success = 1`)** if its observed customer rating is **$\ge 4.0$** (`customer_rating >= 4.0`). Otherwise, it is categorized as **Not Successful (`success = 0`)**.

> [!NOTE]
> **Data Composition**: The usable dataset comprises **4,551 hotels**:
> * **Real Scraped Hotels**: 1,664 hotels (36.6%) scraped via SerpApi with customer reviews and verified amenity listings.
> * **Synthetic Hotels**: 2,887 hotels (63.4%) statistically generated to model diverse market scenarios.
> * *Geographic Coordinates*: Synthetic hotels sit on four fixed reference coordinate pairs. To prevent spatial artifacts, raw latitude and longitude coordinates are strictly excluded from predictive models and from the final dataset table.

> [!WARNING]
> **Association vs. Causation**: All relationships, correlations, feature importances, and association rules represent **empirical associations** in observational data. They do **not** imply that adding an amenity or altering price will causally increase customer ratings.

---

## 2. Interactive Global Filter Panel

The Global Filter Panel sits at the top of the dashboard. Modifying any filter triggers a reactive cascade that recalculates the KPI Row and every active visualization across all tabs.

```
+---------------------------------------------------------------------------------------------------+
| INTERACTIVE GLOBAL FILTERS                                                                        |
| Location: [All Locations | Coimbatore | Goa | Mumbai ...]     Category: [All Categories | 5-star...] |
| Data Type: [Both | Real | Synthetic]                          Price (₹): [====o----------------o]  |
| Rating: [o-----------------------o]                           [ Reset All Filters ]               |
| Selected Hotels: 4,551 / 4,551 (100.0%)                                                           |
+---------------------------------------------------------------------------------------------------+
```

### Filter Controls & Behaviors

| Filter Control | Widget Type | Range / Options | Interaction Behavior & Cascading Impact |
|:---|:---|:---|:---|
| **Location** | Multi-Select Box | `All Locations` (default), plus 30+ major Indian cities (e.g., Coimbatore, Goa, Mumbai, Delhi, Bengaluru, Chennai, Jaipur). | Allows selecting one or multiple cities. Selecting a city isolates hotels operating in that municipal market. |
| **Category** | Multi-Select Box | `All Categories` (default), `5-star hotel`, `4-star hotel`, `3-star hotel`, `2-star hotel`, `1-star hotel`, `Unknown`. | Slices properties by industry star-classification tier. |
| **Data Type** | Single Dropdown | `Both` (default), `Real`, `Synthetic`. | Allows auditing real scraped properties independently from synthetically generated records. |
| **Price (₹)** | Dual Float Range Slider | **₹392 to ₹61,327** (Spans full corpus minimum to maximum). | Narrows the price segment (e.g., budget hotels < ₹2,000, midscale ₹2,000–₹7,000, luxury > ₹10,000). |
| **Rating** | Dual Float Range Slider | **0.0 to 5.0** (Step: 0.1). | Slices properties by customer rating band. Setting `4.0 – 5.0` isolates the entire successful cohort. |
| **Reset All Filters** | Action Button | Single click. | Restores all 5 filters to their baseline defaults (`All Locations`, `All Categories`, `Both`, `₹392–₹61,327`, `0.0–5.0`) in one transaction. |

---

## 3. Real-Time Summary KPI Row

Directly beneath the filter panel, five dynamic KPI cards display top-level metrics for the currently filtered hotel cohort:

```
+------------------+------------------+------------------+------------------+------------------+
| HOTELS SELECTED  |  AVERAGE RATING  |  AVERAGE PRICE   |  SUCCESS RATE    |  AVG AMENITIES   |
|      4,551       |    4.45 / 5.0    |      ₹4,077      |      81.7%       |       10.5       |
| 100.0% of corpus | Range: 1.0 - 5.0 | Median: ₹2,650   | 3,718 successful | Across 15 amens  |
+------------------+------------------+------------------+------------------+------------------+
```

### How to Interpret the KPI Cards

1. **Hotels Selected**:
   * *What it shows*: Active hotel count and percentage of the total 4,551 corpus.
   * *Baseline*: **4,551 properties (100.0%)**.
   * *Interpretation*: Shows sample size strength. When this number drops below 50, statistical caution is advised.
2. **Average Rating**:
   * *What it shows*: Mean customer rating (out of 5.0) and observed range `[min, max]`.
   * *Baseline*: **4.45 / 5.0** (Range: 1.0 – 5.0).
   * *Interpretation*: Indicates customer satisfaction for the selected segment.
3. **Average Price**:
   * *What it shows*: Mean room rate in Indian Rupees (₹) alongside the median rate.
   * *Baseline*: **Mean: ₹4,077 | Median: ₹2,650**.
   * *Interpretation*: The discrepancy between mean and median reflects positive right-skewness caused by high-end luxury resorts.
4. **Average Review Count**:
   * *What it shows*: Mean review volume per hotel alongside the median count.
   * *Baseline*: **Mean: 2,257 reviews | Median: 748 reviews**.
   * *Interpretation*: High review counts signal high customer throughput and established market presence.
5. **Success Rate (≥ 4.0)**:
   * *What it shows*: Percentage and absolute count of properties meeting the success threshold (`customer_rating >= 4.0`).
   * *Baseline*: **81.7% (3,718 successful hotels)**.
   * *Interpretation*: The core benchmark proxy against which subsets can be evaluated.
6. **Avg Amenities**:
   * *What it shows*: Average count of available amenities per property across the 15 monitored hotel amenities.
   * *Baseline*: **10.5 amenities**.
   * *Interpretation*: Measures service breadth. Higher star categories consistently show higher average amenity counts.

---

## 4. Tab 1: Exploratory Data Analysis (EDA)

Tab 1 provides 10 coordinated Plotly visualizations analyzing market distributions, category breakdowns, geographic concentration, and amenity prevalence.

```
+---------------------------------------------------------------------------------------------------+
| TAB 1: EXPLORATORY DATA ANALYSIS (EDA)                                                            |
| [Chart 1: Success Donut]               [Chart 2: Real vs Synthetic Bar]                           |
| [Chart 3: Rating Distribution Hist]    [Chart 4: Category Success Rate Horizontal Bar]            |
| [Chart 5: Top 15 Locations Bar]        [Chart 6: Price vs Review Count Scatter]                   |
| [Chart 7: Amenity Market Prevalence]   [Chart 8: Success Rate Delta (With vs Without Amenity)]    |
| [Chart 9: Amenity Count Box Plot]      [Chart 10: Multi-Feature Correlation Matrix Heatmap]       |
+---------------------------------------------------------------------------------------------------+
```

### Chart-by-Chart Guide & Interpretation

#### Chart 1: Success Distribution (Rating ≥ 4.0)
* **Visual Type**: Donut chart (Green = Successful, Red = Not Successful).
* **Baseline Value**: **81.7% Successful (3,718)** vs. **18.3% Not Successful (833)**.
* **How to Interpret**: Demonstrates the overall class balance of the market. The high base success rate reflects the tendency of online-listed hotels to cluster between 3.8 and 4.6 stars.

#### Chart 2: Real vs. Synthetic Hotels
* **Visual Type**: Grouped bar chart.
* **Baseline Value**: **1,664 Real properties (36.6%)** vs. **2,887 Synthetic properties (63.4%)**.
* **How to Interpret**: Tracks data provenance in the active filter slice.

#### Chart 3: Customer Rating Distribution
* **Visual Type**: Overlaid semi-transparent histogram with a dashed red vertical line at 4.0.
* **How to Interpret**: Visualizes rating distribution shape. Real and synthetic hotels share an aligned distribution centered between 4.2 and 4.6 stars. Properties to the right of the dashed red line constitute the successful cohort.

#### Chart 4: Success Rate by Star Category (%)
* **Visual Type**: Horizontal bar chart sorted by success rate with embedded sample size count labels `(n=...)`.
* **Baseline Values**:
  * **5-star hotel**: **92.0%** success ($n=672$)
  * **4-star hotel**: **89.0%** success ($n=391$)
  * **3-star hotel**: **79.1%** success ($n=1,189$)
  * **2-star hotel**: **72.1%** success ($n=129$)
  * **1-star hotel**: **69.7%** success ($n=145$)
  * **Unknown**: **80.6%** success ($n=2,025$)
* **How to Interpret**: Demonstrates the strong positive relationship between star category and customer rating. Formal service standard enforcement and greater amenity breadth in 4-star and 5-star properties correlate with higher success rates.

#### Chart 5: Top 15 Hotel Locations
* **Visual Type**: Horizontal bar chart of hotel counts across municipal markets.
* **How to Interpret**: Identifies geographic density. Major tourist and commercial hubs (Goa, Mumbai, Delhi, Bengaluru, Chennai) dominate market volume.

#### Chart 6: Price vs. Review Count (by Success)
* **Visual Type**: Scatter plot (Green = Successful, Red = Unsuccessful) with 98th percentile outlier capping.
* **How to Interpret**: Successful hotels appear across all price points. High review counts cluster heavily in properties with ratings $\ge 4.0$, indicating that customer review volume correlates with success.

#### Chart 7: Prevalence of Monitored Amenities (%)
* **Visual Type**: Horizontal bar chart across 15 standard amenities.
* **Baseline Values**: Free Wi-Fi (88.8%), Room Service (85.0%), Free Parking (78.2%), Air Conditioning (73.7%), Restaurant (72.2%), Housekeeping (69.8%), 24-hour Front Desk (67.4%), Breakfast (60.1%), Bar (31.2%), Swimming Pool (28.0%), Fitness Center (19.6%), Airport Shuttle (17.5%), Spa (16.2%), Kitchen (6.4%), Pet-friendly (4.7%).
* **How to Interpret**: Separates "table stakes" utility amenities (Wi-Fi, parking, room service) from selective luxury amenities (pool, gym, spa).

#### Chart 8: Success Rate Δ: With vs. Without Amenity
* **Visual Type**: Diverging bar chart (Green = Positive difference, Red = Negative difference).
* **How to Interpret**: Shows the percentage point (pp) success rate difference between properties offering an amenity versus those without it:
  * Positive values indicate that properties offering the amenity have higher success rates on average.
  * *Notice*: This is an empirical association, not a causal guarantee.

#### Chart 9: Amenity Count Distribution by Success
* **Visual Type**: Box plot comparing total monitored amenities (0–15) between Successful and Unsuccessful hotels.
* **Baseline Values**: Successful hotels show higher median amenity breadth (11 amenities) compared to unsuccessful hotels (9 amenities).
* **How to Interpret**: Confirms that operational breadth across multiple amenities correlates with guest satisfaction.

#### Chart 10: Multi-Feature Correlation Matrix Heatmap
* **Visual Type**: Numerical correlation matrix heatmap between `price`, `log_price`, `review_count`, `amenity_count`, and `success`.
* **Important Note**: `customer_rating` is deliberately excluded to preserve analytical integrity, as `success` is derived from it.
* **How to Interpret**: Identifies statistical co-movements between numerical predictors.

---

## 5. Tab 2: Predictive Models

Tab 2 integrates the supervised machine learning pipeline from Review 1. Built on a stratified 80/20 train/test split (911 test hotels) using 21 predictors (19 numeric including 15 amenities, 2 categorical), this tab evaluates model performance across the full test set and on dynamically filtered subsets.

```
+---------------------------------------------------------------------------------------------------+
| TAB 2: PREDICTIVE MODELS                                                                          |
| [Champion Model Card: Tuned Random Forest (300 Trees, max_depth=10, class_weight='balanced')]     |
| [Dynamic Performance Evaluation Table: RF vs DT vs LR on filtered test subset (N=...)]           |
| [Performance Comparison Bar Chart]     [Side-by-side Confusion Matrices (LR, DT, RF)]             |
| [ROC Curves with AUC legend]           [Top 15 Random Forest Feature Importances]                 |
+---------------------------------------------------------------------------------------------------+
```

### Champion Model & Benchmark Specifications

The **Tuned Random Forest Classifier** is the selected champion model based on multi-metric optimization:

| Model Architecture | Hyperparameters | Full Test Set Accuracy | Balanced Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | ROC-AUC |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest (Champion)** | 300 Trees, `max_depth=10`, `min_samples_split=5`, `class_weight='balanced'` | **81.34%** | **0.6443** | **0.6777** | **0.6443** | **0.6571** | **0.8037** | **0.7072** |
| **Decision Tree** | `max_depth=7`, `min_samples_split=10`, `random_state=42` | 80.35% | 0.5662 | 0.6297 | 0.5662 | 0.5750 | 0.7724 | 0.6344 |
| **Logistic Regression** | $C=10$, `class_weight='balanced'`, `max_iter=2000` | 68.94% | 0.6612 | 0.6056 | 0.6612 | 0.6045 | 0.7205 | 0.6955 |

### How Filter Changes Impact Tab 2
* When global filters are adjusted, the dashboard takes the intersection between the active filter selection and the 911 holdout test hotels (`test_indices = X_test.index.intersection(filtered_df.index)`).
* All three models are re-evaluated on this exact test subset in real time.
* **Built-in Safeguards**:
  * *Sample Size Notice*: When the filtered test subset contains fewer than 50 hotels ($n < 50$), a yellow notice appears warning that small-sample metrics exhibit statistical variance.
  * *Single Class Notice*: If the filtered test subset contains only one class (e.g., all hotels are successful), precision, recall, macro metrics, and ROC-AUC are disabled gracefully to prevent mathematical division-by-zero errors.

### Interpreting Model Visualizations

1. **Confusion Matrices (Baseline 911 test hotels)**:
   * **Random Forest**: True Negatives = 63, False Positives = 104, False Negatives = 66, True Positives = 678.
   * **Logistic Regression**: High negative recall (103 TN), but lower precision (219 FN).
   * **Decision Tree**: Conservative classification favoring the majority class (700 TP, 32 TN).
2. **Top 15 Feature Importances**:
   * Top predictors: `price`, `log_price`, `review_count`, and `amenity_count`.
   * Key individual amenities with predictive power: `amenity_free_parking`, `amenity_air_conditioning`, `amenity_restaurant`, and `amenity_swimming_pool`.
   * *Interpretation*: Pricing structure, review volume, and total amenity breadth outweigh individual amenities in predicting classification success.

---

## 6. Tab 3: Text Mining Integration

Tab 3 hosts the **Dynamic NLP Engine**, analyzing customer review feedback across real hotels. While sentiment analysis in earlier iterations relied on static precomputations, this tab recomputes metrics dynamically whenever global filters are adjusted.

```
+---------------------------------------------------------------------------------------------------+
| TAB 3: TEXT MINING (DYNAMIC RECOMPUTATION ENGINE)                                                 |
| Filter Scope: Real Hotels Only (1,659 matched hotels, 26,175 reviews)                             |
| [VADER Sentiment Distribution Donut [Dynamic]]   [Star Rating vs VADER Sentiment Bar [Dynamic]]   |
| [Top Customer Pain Points [Dynamic]]             [Experience Themes (% Positive) [Dynamic]]       |
| [Top 15 Frequent Words [Dynamic]]                [Top 10 Bigrams in Reviews [Dynamic]]           |
| [Amenity Aspect Sentiment Bar [Dynamic]]         [VADER vs Rating Confusion Matrix [Dynamic]]     |
| [Sentiment Evaluation Metrics Bar [Dynamic]]     [Dynamic Review Word Cloud [Dynamic]]            |
| [Key Text Mining Takeaways Summary Box [Dynamic]]                                                 |
| [Interactive Review Explorer Table: Search & Pagination (15 reviews/page)]                       |
+---------------------------------------------------------------------------------------------------+
```

### Review Subset Derivation & Teammate Logic
* The dynamic engine derives active review IDs by matching `hotel_id` from the filtered real hotels (`rev_subset = rev_df[rev_df["hotel_id"].isin(active_real_hotel_ids)]`).
* All keyword mappings, regular expressions, sentence splitting, and stopword cleaning strictly mirror Poojitha's notebook ([`reference/review2_updated.ipynb`](file:///d:/Semesters/Semester%207/Business%20Analytics/Project/Review%202/hotel-dashboard/reference/review2_updated.ipynb)):
  * **Pain Points Denominator**: Top customer pain points are calculated as a **percentage of filtered negative reviews** ($N_{neg}$).
  * **Experience Themes**: Calculated across 10 operational themes as percentage of positive mentions ($\frac{Mentions_{pos}}{Total Mentions}$).
  * **Amenity Sentiment**: Evaluated at sentence level for reviews mentioning specific amenity keywords.

### Baseline vs. Filtered Text Mining Metrics

| Component / Metric | Baseline (All 26,175 Reviews) | Filtered: Coimbatore 4-Star (98 Reviews) | How to Interpret the Shift |
|:---|:---:|:---:|:---|
| **VADER Sentiment Distribution** | Pos: **76.3%**, Neg: **18.6%**, Neu: **5.1%** | Pos: **84.7%**, Neg: **13.3%**, Neu: **2.0%** | 4-star properties in Coimbatore outperform the national corpus sentiment by +8.4 percentage points. |
| **Sentiment vs. Rating Accuracy** | **0.8737 (87.37%)** | **0.8980 (89.80%)** | Evaluates agreement between VADER classification and customer star rating (1–2 = Neg, 3 = Neu, 4–5 = Pos). |
| **Cohen's Kappa** | **0.6801** | **0.4286** | Measures inter-rater agreement above chance. High agreement persists across diverse segments. |
| **Top Pain Point #1** | **Cleanliness / hygiene** (1,497 reviews / 30.7%) | **Price / value for money** (3 reviews / 23.1%) | Nationally, hygiene is the leading complaint; in Coimbatore 4-star hotels, complaints shift to pricing expectations and maintenance. |
| **Top Pain Point #2** | **Rude / unhelpful staff** (992 reviews / 20.4%) | **Broken / maintenance** (3 reviews / 23.1%) | Tracks secondary complaint drivers in negative feedback. |
| **Top Positive Theme** | **Location** (84.9% positive, 4,636 mentions) | **Facilities** (100.0% positive, 13 mentions) | Highlights primary satisfaction drivers. |
| **Word Cloud** | Broad national terms (*staff*, *service*, *clean*, *breakfast*, *location*) | Localized slice terms (*clean*, *service*, *food*, *maintained*, *experience*) | Word cloud regenerates dynamically from filtered token counts using teammate stopword lists. |

### Interactive Review Explorer Table (Task 4)
* Located at the bottom of Tab 3, this interactive table displays individual review records matching the active filter selection.
* **Columns**: `Hotel Name`, `City`, `Category`, `Star Rating` (★), `VADER Sentiment` (badge), and `Review Excerpt`.
* **Controls**: Real-time keyword text search and Previous / Next pagination (15 reviews per page).
* **Use Case**: Allows managers to drill down from aggregate sentiment charts into specific customer narratives.

---

## 7. Tab 4: Apriori Association Rules

Tab 4 integrates Anupama's Association Rule Mining deliverables. Using the Apriori algorithm applied to 33 standardized amenity transaction vectors across 1,430 real hotels, this tab uncovers high-lift amenity bundles.

```
+---------------------------------------------------------------------------------------------------+
| TAB 4: APRIORI ASSOCIATION RULES                                                                  |
| Controls: Min Support [0.05] | Min Conf [0.60] | Min Lift [2.0] | Search Amenity: [Spa         ]  |
| [Top 15 Standardized Amenity Support Bar Chart]                                                   |
| [Top Rules by Lift Bar Chart]          [Support vs Confidence Scatter (Marker Size = Lift)]       |
| [Top 15 Ranked Rules Table (Antecedents -> Consequents, Support, Confidence, Lift)]               |
| [Strategic Amenity Bundling Insights Box]                                                         |
+---------------------------------------------------------------------------------------------------+
```

### Understanding Apriori Metrics

1. **Support ($S$)**:
   * *Definition*: The proportion of hotels that possess both amenity sets $A$ and $B$:
     $$Support(A \rightarrow B) = P(A \cap B)$$
   * *Interpretation*: Measures how common an amenity combination is across the market. A support of 0.0818 means 8.18% of all analyzed hotels possess the bundle.
2. **Confidence ($C$)**:
   * *Definition*: The conditional probability that a hotel has amenity $B$ given that it already has amenity $A$:
     $$Confidence(A \rightarrow B) = P(B \mid A) = \frac{Support(A \cap B)}{Support(A)}$$
   * *Interpretation*: Measures rule reliability. A confidence of 74.02% means that 74.02% of hotels offering the antecedent amenity also offer the consequent.
3. **Lift ($L$)**:
   * *Definition*: The ratio of observed joint occurrence to expected independent occurrence:
     $$Lift(A \rightarrow B) = \frac{Confidence(A \rightarrow B)}{Support(B)} = \frac{P(A \cap B)}{P(A) \cdot P(B)}$$
   * *Interpretation*: Measures synergistic bundling strength:
     * $Lift = 1.0$: Antecedent and consequent appear together purely by chance.
     * $Lift > 1.0$: Positive synergy. The amenities appear together significantly more often than random chance.
     * **Lift > 5.0**: Exceptionally strong complementary bundle.

### Top Discovered Amenity Bundles

| Rank | Discovered Rule ($A \rightarrow B$) | Support (%) | Confidence (%) | Lift | Strategic Hospitality Interpretation |
|:---:|:---|:---:|:---:|:---:|:---|
| **1** | **Business center, Pool → Spa** | **6.57%** | **74.02%** | **6.190** | **The Premium Wellness Hub**: Standalone spas rarely succeed. When paired with business facilities and pools, spa likelihood jumps by **6.19×**. |
| **2** | **Fitness center, Pool → Spa** | **8.18%** | **68.82%** | **5.755** | **The Leisure & Wellness Core**: Core leisure bundle for luxury resort properties. |
| **3** | **Bar, Fitness center → Spa** | **7.90%** | **66.08%** | **5.526** | **Full-Service Lifestyle Bundle**: Complements evening entertainment with daytime wellness. |
| **4** | **Business center, Fitness center → Spa** | **6.92%** | **65.13%** | **5.447** | **Corporate Executive Wellness**: Standard bundle for upscale urban business hotels. |

### Interactive Tab 4 Controls
* **Support Slider (0.01 – 0.20)**: Filters out niche or rare bundles (default: 0.05 / 5%).
* **Confidence Slider (0.30 – 0.95)**: Sets reliability threshold (default: 0.60 / 60%).
* **Lift Slider (1.0 – 6.5)**: Focuses exclusively on high-synergy pairings (default: 2.0).
* **Search Amenity Box**: Real-time text filter. Typing `"Spa"`, `"Pool"`, or `"Bar"` filters the chart and table to rules containing that keyword.
* **Dynamic mlxtend Checkbox**: When checked, re-runs the Apriori mining algorithm on the filtered hotel transactions.

---

## 8. Tab 5: Final Project Dataset & Dual Export

Tab 5 provides transparency into the final project dataset, supporting interactive exploration and data exports.

```
+---------------------------------------------------------------------------------------------------+
| TAB 5: FINAL PROJECT DATASET                                                                      |
| Search Hotel: [Taj                ]  [◀ Previous] [Next ▶]  Page 1 of 1 (13 matching hotels)      |
| [⬇ Download Filtered Data (CSV)]    [⬇ Download Full Final Dataset (CSV)]                        |
|                                                                                                   |
| Hotel Name       | Location | Category     | Type | Price   | Rating | Reviews | Amens | Status    |
|------------------+----------+--------------+------+---------+--------+---------+-------+-----------|
| Taj Exotica Goa  | Goa      | 5-star hotel | Real | ₹31,578 | ★ 4.7  |   7,706 |    30 | 1 (Succ)  |
| Taj Fort Aguada  | Goa      | 5-star hotel | Real | ₹24,500 | ★ 4.6  |   5,210 |    29 | 1 (Succ)  |
| Showing rows 1–13 of 13 hotels (Page 1 of 1)                                                      |
+---------------------------------------------------------------------------------------------------+
```

### Table Specifications & Features
1. **Readable Column Layout**:
   * `hotel_name`: Property title.
   * `location`: Municipal market / city.
   * `hotel_category`: Official star tier (`5-star hotel` down to `1-star hotel`, `Unknown`).
   * `data_type`: Data source origin (`Real` vs. `Synthetic`).
   * `price`: Cleanly formatted room rate (`₹XX,XXX`).
   * `customer_rating`: Observed rating (`★ X.X`).
   * `review_count`: Total customer review count (`X,XXX`).
   * `amenity_count`: Breadth of available amenities (`XX`).
   * `success`: Classification status badge (`1 (Successful)` in green, `0 (Not Successful)` in gray).
   * **Strict Exclusion**: Raw latitude and longitude coordinates are deliberately excluded.
2. **Text Search on Hotel Name**:
   * Instant, case-insensitive text search. Typing a keyword (e.g., `Oberoi`, `Taj`, `Hyatt`, `Resort`) dynamically filters the table and updates the page counter.
3. **Paging Controls**:
   * Clean pagination (25 hotels per page) with Previous (`◀`) and Next (`▶`) buttons.
   * Shows active page and total pages (e.g., `Page 1 of 183` for all 4,551 hotels).
4. **Dual CSV Export Buttons**:
   * **Download Filtered Data (CSV)**: Exports the exact subset of hotels matching the active global filters.
   * **Download Full Final Dataset (CSV)**: Exports the complete 4,551-hotel corpus.
   * **Environment Auto-Detection**:
     * In **Google Colab**, automatically invokes `google.colab.files.download()` to trigger browser downloads natively.
     * In **Local Jupyter / JupyterLab**, generates a Base64 data-URI download link that opens in any browser.

---

## 9. Cascading Interaction Matrix

This matrix details how modifying specific controls cascades through the entire dashboard:

| User Action | KPI Row Response | Tab 1 (EDA) Response | Tab 2 (Models) Response | Tab 3 (Text Mining) Response | Tab 4 (Apriori) Response | Tab 5 (Dataset) Response |
|:---|:---|:---|:---|:---|:---|:---|
| **Filter by City** *(e.g. Coimbatore)* | Recomputes count ($n=224$), avg price, avg rating, and success rate for that city. | Slices all 10 charts to show local category rates, pricing, and amenities. | Evaluates models on holdout test hotels located in that city. | Filters reviews to real hotels in that city ($n=1,301$ reviews); recalculates sentiment and word cloud. | Shows corpus rules by default; if dynamic recomputation is enabled, re-mines rules on local hotels. | Filters table to properties in that city; recalculates page count. |
| **Filter by Category** *(e.g. 4-star)* | Slices metrics to 4-star segment ($n=391$, 89.0% success rate). | Displays rating distributions, pricing, and amenity rates for 4-star properties. | Evaluates RF model on 4-star test hotels. | Recalculates sentiment on 4-star reviews ($n=2,937$); updates pain points and bigrams. | Re-evaluates amenity support and bundling for 4-star listings. | Displays 4-star hotels with paging. |
| **Narrow Price Slider** *(e.g. ₹2k–₹7k)* | Recalculates median price, success rate, and active hotel count. | Updates scatter plot, category rates, and correlation matrix for the price tier. | Evaluates test accuracy for hotels within that price bracket. | Recalculates customer sentiment for the filtered budget tier. | Focuses on amenity bundles within that pricing cohort. | Limits table view to hotels within the specified rate band. |
| **Combine Filters** *(Coimbatore + 4-star + ₹2k–₹7k)* | Recomputes all cards for the intersection ($n=12$ hotels, 91.7% success, ₹4,890 avg price). | All 10 charts update to reflect the 12 matching properties. | Test subset $N=2$; displays sample size warning ($N < 50$). | Recomputes on 6 real hotels (98 reviews); shifts VADER positive to 84.7%; updates WordCloud. | Corpus rules displayed; dynamic toggle recomputes on the active transaction vector. | Table shows 12 matching rows (Page 1 of 1). |
| **Search Amenity in Tab 4** *(Type "Spa")* | No change (Tab 4 control is local to Apriori). | No change. | No change. | No change. | Instantly filters Apriori chart and table to rules containing "Spa". | No change. |
| **Click Reset All Filters** | Restores 4,551 hotels (100.0%), ₹4,077 avg price, 81.7% success. | Restores full 10-chart baseline distributions. | Restores 911 test hotels and baseline model benchmarks. | Restores baseline 26,175 reviews (76.3% positive) and full word cloud. | Restores full Apriori rule ranking. | Table resets to Page 1 of 183 showing all 4,551 hotels. |

---

## 10. Presentation Demonstration Script

Use this scenario during project presentations to demonstrate the dashboard's capabilities:

### Step-by-Step Demo Protocol

1. **Demonstrate the Baseline Default View**:
   * Point out the top KPI card: **4,551 hotels (100.0%)** and **81.7% success rate**.
   * Note the price slider span: **₹392 to ₹61,327**, ensuring that every property is included by default.
2. **Execute the Demo Scenario**:
   * In **Location**, select `Coimbatore`.
   * In **Category**, select `4-star hotel`.
   * On the **Price Slider**, narrow the range to `₹2,000 – ₹7,000`.
   * Observe the instantaneous update: The KPI row updates to **12 hotels**, **₹4,890 average price**, and a **91.7% success rate**.
3. **Inspect Tab 1 (EDA)**:
   * Show that all 10 charts have dynamically adjusted to reflect the 12 properties.
4. **Inspect Tab 2 (Predictive Models)**:
   * Highlight the **Sample Size Notice** ($n=2$ in test set), explaining statistical variance on small subsets.
5. **Inspect Tab 3 (Text Mining Engine)**:
   * Point to the dynamic banner: `(6 matched hotels, 98 reviews)`.
   * Highlight the sentiment shift: VADER positive sentiment increases to **84.7%** (vs. 76.3% baseline).
   * Review the dynamic **Word Cloud** and note that the **Top Customer Pain Points** have shifted from national hygiene complaints to property maintenance and value-for-money expectations.
   * Scroll down to the **Review Explorer** to show live review excerpts from Coimbatore 4-star properties.
6. **Inspect Tab 4 (Association Rules)**:
   * Type `"Spa"` into the Amenity Search box.
   * Point out the top bundle: **Business center, Pool → Spa** (**Lift: 6.190**, Confidence: 74.02%).
   * Deliver the strategic insight: Spas require supporting fitness and business infrastructure to succeed.
7. **Inspect Tab 5 (Final Project Dataset)**:
   * Show the filtered table displaying the 12 Coimbatore properties.
   * Demonstrate the **"Download Filtered Data (CSV)"** button.
   * Clear the hotel name search to show the full dataset export option.
8. **Click "Reset All Filters"**:
   * Demonstrate that the dashboard returns to the baseline state (4,551 hotels, 100.0%).
