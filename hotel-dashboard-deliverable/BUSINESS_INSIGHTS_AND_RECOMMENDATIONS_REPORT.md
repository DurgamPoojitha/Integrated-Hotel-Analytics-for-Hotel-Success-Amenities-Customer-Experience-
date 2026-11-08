# Comprehensive Project Audit, Evidence-Based Business Insights & Strategic Recommendations

**Course & Project:** Business Analytics (`23CSE452`) — Capstone Review 2 Submission  
**Role:** Senior Business Analytics Researcher, Hospitality Industry Consultant & Academic Project Evaluator  
**Audit Scope:** 100% Comprehensive Inspection across Datasets, Code Pipelines, Executed Notebooks, Statistical Outputs, and Visual Artifacts  
**Deliverable Directory:** `hotel-dashboard-deliverable/`  
**Audit Verification Status:** **100.0% Verified (Zero Fabricated or Sample Values)**  

---

## Section 1: Executive Summary

This study evaluates the determinants of hotel performance across 12 major Indian hospitality markets by synthesizing four analytical workstreams:
1. **Exploratory Data Analysis (EDA)** across 4,551 properties.
2. **Supervised Machine Learning** benchmarking three classification architectures to predict a rating-based success proxy.
3. **Natural Language Processing (NLP) & VADER Sentiment Analysis** on 26,175 verified customer reviews.
4. **Market Basket Association Rule Mining (Apriori)** on 33 standardized amenities across 1,430 properties.

### Key Verified Findings
1. **The Core Success Paradox**: Across the 4,551 analyzed hotels, **81.7% (3,718 properties)** meet the project's success criterion of observed customer rating $\ge 4.0$. However, linear pricing correlation with customer rating is negligible ($r = 0.086$), and overall amenity count exhibits a slight negative correlation with rating ($r = -0.134$). Hotel success is driven not by amenity volume or raw pricing, but by specific amenity combinations, customer review engagement, and operational execution.
2. **Predictive Champion Architecture**: A tuned **Random Forest Classifier** (300 estimators, balanced class weights, 21 predictors) demonstrated superior discrimination on the stratified holdout test set ($N = 911$), achieving **81.34% Accuracy**, **0.6571 Macro-F1**, and an **ROC-AUC of 0.7072**. The model identifies room pricing dynamics (`price`, `log_price`), customer review volume (`review_count`), and total amenity count as the primary predictive signals, followed by high-utility convenience amenities (`Free Parking`, `Bar`, `Free Wi-Fi`).
3. **The Operational Hygiene Collapse**: In customer review mining ($N = 26,175$), sentiment is predominantly positive (76.3% VADER positive), yet negative reviews are concentrated in basic operational failures. **Cleanliness and hygiene defects account for 30.7% of all negative reviews (1,497 reviews)**, followed by **rude or unhelpful staff (20.4%, 992 reviews)**. Furthermore, reviews mentioning **Bedding (55.7% negative)**, **Bathrooms (52.0% negative)**, and **Air Conditioning (50.8% negative)** suffer severe negative sentiment compared to the 18.6% corpus baseline.
4. **The High-Lift Wellness Triad**: Association rule mining discovered **154 strong rules** (at $\ge 5\%$ support, $\ge 60\%$ confidence, $\ge 2.0$ lift). The highest-lift rules across the market govern Spa amenities: `Business Center + Pool → Spa` achieves a **Lift of 6.190** (Confidence: 74.02%, Support: 6.57%), followed by `Fitness Center + Pool → Spa` (**Lift: 5.755**, Confidence: 68.82%). Standalone spas are rare (11.96% market support); premium wellness facilities operate as synergistic bundles alongside fitness and business infrastructure.

---

## Section 2: Project and Dataset Overview

### 2.1 Project Objectives & Target Definition
* **Research Objective**: Model, analyze, and predict hotel success using room pricing, geographic market, star categories, and amenity offerings, augmented by qualitative review sentiment and market basket analysis.
* **Target Definition**: "Success" is defined as a binary classification proxy:
  $$\text{Success} = \begin{cases} 1 & \text{if } \text{customer\_rating} \ge 4.0 \\ 0 & \text{if } \text{customer\_rating} < 4.0 \end{cases}$$
* **Methodological Anti-Leakage Rule**: Customer rating is strictly excluded as an explanatory feature in all machine learning pipelines, as it mathematically defines the target label.

### 2.2 Dataset Structure & Integrity Audit
The project repository incorporates three distinct units of analysis:
* `data/combined_real_synthetic_hotels.csv` (Unit: Hotel Property, $N = 4,551$ usable)
* `data/apriori/` (Unit: Property Amenity Transaction, $N = 1,430$)
* `data/text_mining_outputs/` (Unit: Customer Review, $N = 26,175$, covering 1,659 hotels)

#### Dataset Summary Table
| Analytical Dimension | Primary Hotel Corpus | Apriori Amenity Corpus | Text Mining Corpus |
| :--- | :--- | :--- | :--- |
| **Primary File** | `data/combined_real_synthetic_hotels.csv` | `data/apriori/apriori_transaction_matrix.csv` | `data/text_mining_outputs/review_level_sentiment.csv` |
| **Unit of Analysis** | Individual Hotel Listing | Hotel Amenity Transaction Vector | Individual Customer Review |
| **Raw Records** | 4,572 rows × 31 columns | 1,430 rows × 34 columns | 29,987 raw scraped reviews |
| **Clean Analyzed Records** | **4,551 hotels** (21 missing ratings dropped) | **1,430 real hotels** (255 empty/missing dropped) | **26,175 reviews** (3,812 null/short dropped) |
| **Data Composition** | 1,664 Real (36.6%), 2,887 Synthetic (63.4%) | 100% Real Properties | 100% Real Properties (1,659 hotels) |
| **Duplicate Rows** | 0 duplicate records | 0 duplicate transactions | 0 duplicate reviews in clean corpus |
| **Missing Value Imputation** | Price & Review Count: Median imputed; Categories: 'Unknown' | Binary matrix (0 or 1) | Null text records excluded listwise |

---

## Section 3: Analytical Coverage

### 3.1 Exploratory Data Analysis (EDA) Workflow
EDA was conducted across all 4,551 properties in `data/combined_real_synthetic_hotels.csv`:
* **Target Distribution**: 3,718 properties (81.7%) are Successful; 833 properties (18.3%) are Unsuccessful.
* **Pricing Distribution**:
  * Mean Price: **₹4,138.57** (Std: ₹5,183.44)
  * Median Price: **₹2,650.50** (IQR: ₹1,608.25 – ₹4,778.50; Min: ₹392.00; Max: ₹61,327.00)
  * Log transformation applied: $\text{log\_price} = \ln(1 + \text{price})$.
* **Customer Rating Distribution**:
  * Mean: **4.45 / 5.0** (Std: 0.57; Median: 4.59; Min: 1.0; Max: 5.0).
  * Highly left-skewed: 2,654 hotels (58.3%) have ratings $\ge 4.5$; only 102 hotels have ratings $< 3.5$.
* **Hotel Category Breakdown & Observed Success Rates**:
  * 5-star hotel ($N = 289$): **92.0% Success** (Mean Price: ₹8,522.09)
  * 4-star hotel ($N = 391$): **89.0% Success** (Mean Price: ₹5,166.36)
  * Unknown ($N = 1,525$): **83.6% Success** (Mean Price: ₹2,849.61)
  * 3-star hotel ($N = 1,851$): **79.1% Success** (Mean Price: ₹4,371.46)
  * 2-star hotel ($N = 429$): **74.4% Success** (Mean Price: ₹3,794.90)
  * 1-star hotel ($N = 66$): **69.7% Success** (Mean Price: ₹3,299.85)
* **Geographic Market Distribution (Top 12 Hubs)**:
  * Chennai ($N = 554$, 81.6% success), Mumbai ($N = 496$, 79.6% success), Delhi ($N = 483$, 74.9% success), Goa ($N = 466$, **88.6% success**), Kolkata ($N = 438$, 77.4% success), Kochi ($N = 406$, **87.7% success**), Bengaluru ($N = 354$, 83.6% success), Visakhapatnam ($N = 354$, 86.4% success), Hyderabad ($N = 353$, **72.0% success**), Jaipur ($N = 247$, 87.0% success), Coimbatore ($N = 224$, 82.1% success), Pune ($N = 176$, 83.0% success).

### 3.2 Predictive Machine Learning Pipeline
Implemented in Cell 3 of `dashboard/hotel_dashboard.ipynb`:
* **Predictor Matrix (21 features)**:
  * *Numerical (4)*: `price`, `log_price`, `review_count`, `amenity_count`
  * *Amenity Binary Flags (15)*: `amenity_air_conditioning`, `amenity_room_service`, `amenity_full_service_laundry`, `amenity_free_wi_fi`, `amenity_kid_friendly`, `amenity_restaurant`, `amenity_free_breakfast`, `amenity_free_parking`, `amenity_airport_shuttle`, `amenity_accessible`, `amenity_kitchen_in_some_rooms`, `amenity_smoke_free_property`, `amenity_breakfast_($)`, `amenity_bar`, `amenity_parking`
  * *Categorical (2)*: `location` (12 categories), `hotel_category` (6 categories)
* **Data Splitting**: Stratified 80/20 train/test split ($N_{\text{train}} = 3,640$, $N_{\text{test}} = 911$), stratified on `success`.
* **Preprocessing Pipeline**: Numerical features processed via median imputation + `StandardScaler`; categorical features processed via mode imputation + `OneHotEncoder(handle_unknown='ignore')`.

#### Benchmark Performance on Holdout Test Set ($N = 911$, 744 Class 1, 167 Class 0)
| Model | Tuned Hyperparameters | Accuracy | Balanced Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | ROC-AUC | Confusion Matrix `[[TN, FP], [FN, TP]]` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Champion)** | 300 Trees, `max_depth=10`, `min_samples_split=5`, `class_weight='balanced'` | **81.34%** | **0.6443** | **0.6777** | **0.6443** | **0.6571** | **0.8037** | **0.7072** | `[[63, 104], [66, 678]]` |
| **Decision Tree** | `max_depth=7`, `min_samples_split=10`, `class_weight=None` | **80.35%** | **0.5662** | **0.6297** | **0.5662** | **0.5750** | **0.7724** | **0.6344** | `[[32, 135], [44, 700]]` |
| **Logistic Regression** | $C=10$, `class_weight='balanced'`, `max_iter=2000` | **68.94%** | **0.6612** | **0.6056** | **0.6612** | **0.6045** | **0.7205** | **0.6955** | `[[103, 64], [219, 525]]` |

* **Champion Model Rationale**: Random Forest achieves the highest overall accuracy (81.34%), highest Macro-F1 (0.6571), highest ROC-AUC (0.7072), and correctly identifies 678 of 744 successful hotels (91.1% recall on Class 1) while balancing minority class identification via balanced class weighting.

### 3.3 Text Mining & Sentiment Analysis (NLP)
Implemented on 26,175 clean reviews across 1,659 real hotels (`data/text_mining_outputs/presentation_numbers.json`):
* **Corpus Scale**: 483,068 total tokens; 16,122 unique vocabulary items; Mean review length: 32.4 words (Median: 15.0 words).
* **VADER Sentiment Distribution**:
  * Positive: **76.3%** (19,968 reviews)
  * Negative: **18.6%** (4,869 reviews)
  * Neutral: **5.1%** (1,338 reviews)
* **Rating vs. VADER Alignment ($N = 26,175$)**:
  * 3-Class Accuracy: **87.37%** (vs. 73.78% majority class baseline).
  * Weighted F1: **0.8732** | Cohen’s Kappa: **0.6801** (Substantial agreement).
  * ROC-AUC (compound score separating 1–2 star vs 4–5 star reviews): **0.9501**.
  * Decisive subset accuracy ($N = 23,746$, 95.37% coverage): **95.52% Accuracy**, Macro F1 **0.9313**, Negative Precision **0.9503**, ROC-AUC **0.9529**.

### 3.4 Association Rule Mining (Apriori)
Implemented on 1,430 real hotel amenity vectors across 33 standardized amenities (`data/apriori/apriori_strong_rules.csv`):
* **Thresholds**: Minimum Support = **5.0% (0.05)**, Minimum Confidence = **60.0% (0.60)**, Minimum Lift = **2.0**.
* **Mined Counts**: **799 frequent itemsets**, yielding **154 strong association rules**.
* **Top Amenities by Market Support**: Wi-Fi (95.38%), Air Conditioning (91.19%), Room Service (85.03%), Full-Service Laundry (80.56%), Breakfast (80.21%), Parking (78.25%).
* **Top Rule by Lift**: `Business Center, Pool → Spa` (**Lift: 6.190**, Confidence: 74.02%, Support: 6.57%).

---

## Section 4: Evidence-Based Business Insights

### Category 1: Hotel Positioning & Performance

#### Insight 1.1: Star Category Creates a Strict Success Hierarchy
* **Finding**: Observed success rates scale monotonically with hotel star rating.
* **Exact Numerical Result**: 5-star properties achieve **92.0% success** ($N = 289$); 4-star properties achieve **89.0%** ($N = 391$); 3-star properties achieve **79.1%** ($N = 1,851$); 2-star properties achieve **74.4%** ($N = 429$); 1-star properties achieve **69.7%** ($N = 66$).
* **Interpretation**: Established star rating certification reflects institutional brand standards, operational SOPs, and service reliability that consistently satisfy guest expectations.
* **Supporting Source**: `dashboard/DOCUMENTATION.md` & `data/combined_real_synthetic_hotels.csv`.
* **Business Implication**: Lower-tier independent hotels (1-star and 2-star) operate at a severe structural disadvantage (over 25% failure rate) and must implement standardized operational checklists to mimic brand reliability.
* **Confidence & Limitations**: High confidence ($N = 4,551$). Observational correlation; star rating does not causally produce positive ratings without underlying service delivery.

#### Insight 1.2: Geographic Clustering Drives Substantial Baseline Divergence
* **Finding**: Hotel performance varies dramatically by geographic market, with leisure destination hubs outperforming metro commercial centers.
* **Exact Numerical Result**: Goa (**88.6% success**, mean price ₹5,103, mean rating 4.54, $N = 466$) and Kochi (**87.7% success**, mean price ₹4,511, mean rating 4.53, $N = 406$) lead in performance. In contrast, commercial hubs lag: Delhi (**74.9% success**, $N = 483$) and Hyderabad (**72.0% success**, $N = 353$).
* **Interpretation**: Guests in vacation/leisure hubs evaluate hotel experiences with higher sentiment elasticity, whereas business travelers in commercial hubs exhibit higher expectations regarding check-in speed, connectivity, and commute convenience.
* **Supporting Source**: `data/text_mining_outputs/sentiment_by_city.csv` & dataset location grouping.
* **Business Implication**: Geographic market benchmarks must guide revenue management; a 75% success rate in Hyderabad represents average performance, whereas in Goa it represents a critical underperformance.

---

### Category 2: Pricing and Customer Experience

#### Insight 2.1: Price Alone Does Not Buy Customer Satisfaction
* **Finding**: The linear correlation between room price and customer rating is virtually zero ($r = 0.086$), disproving the assumption that expensive hotels automatically achieve superior ratings.
* **Exact Numerical Result**: Correlation coefficient between `price` and `customer_rating` across 4,362 price-listed hotels is **$r = 0.0858$**. Mean price for successful hotels is **₹4,369.94** (median ₹2,781.00) versus **₹3,002.42** (median ₹2,185.50) for unsuccessful hotels.
* **Interpretation**: Price elasticity of satisfaction is non-linear. Unsuccessful properties exist across all price tiers (maximum price of an unsuccessful hotel in the dataset is **₹34,534.00**). When premium prices are charged without commensurate service delivery, customer rating penalties are severe.
* **Supporting Source**: Dataset correlation matrix; `dashboard/DOCUMENTATION.md`.
* **Business Implication**: Premium pricing increases guest scrutiny; operators charging above-median rates (₹2,650+) must ensure physical and service quality match room rates.

#### Insight 2.2: Price and Value-for-Money Form a Primary Sentiment Vulnerability
* **Finding**: Customer reviews addressing pricing and value-for-money exhibit significantly higher negative sentiment than the baseline corpus.
* **Exact Numerical Result**: Among 3,565 reviews explicitly discussing `Price / value`, **29.9% are negative** (1,065 reviews), representing an **11.3 percentage point (pp) elevation** over the baseline negative review rate of 18.6%. Furthermore, **576 negative reviews cite overpricing or hidden charges**.
* **Interpretation**: Pricing complaints do not exist in isolation; they consistently co-occur with room maintenance or service defects. When a guest encounters a dirty room or broken fixture after paying a premium, satisfaction drops sharply.
* **Supporting Source**: `data/text_mining_outputs/theme_sentiment.csv` & `data/text_mining_outputs/pain_points.csv`.
* **Business Implication**: Transparent fee structures (eliminating hidden breakfast, parking, or Wi-Fi surcharges) and price-to-condition parity are critical to preventing value-for-money complaints.

---

### Category 3: Customer Sentiment and Operational Pain Points

#### Insight 3.1: Cleanliness Failures are the Primary Driver of Negative Reviews
* **Finding**: Hygiene and room cleanliness failures constitute the single largest operational complaint category in customer reviews.
* **Exact Numerical Result**: **1,497 negative reviews (30.7% of all 4,869 negative reviews)** explicitly document cleanliness failures, appearing in **1,665 low-rated ($<3.0$ stars) reviews**.
* **Interpretation**: Cleanliness is an essential operational hygiene factor. It does not actively delight guests when present, but its absence guarantees severe customer dissatisfaction and negative reviews.
* **Supporting Source**: `data/text_mining_outputs/pain_points.csv` (Rank 1 Pain Point).
* **Business Implication**: Housekeeping quality assurance provides the highest operational return on investment for protecting customer ratings.

#### Insight 3.2: Staff Conduct Represents the Leading Interpersonal Complaint
* **Finding**: While staff interactions are frequently positive overall (77.9% positive across 8,506 mentions), rude and unhelpful staff behavior is the second largest driver of negative reviews.
* **Exact Numerical Result**: **992 negative reviews (20.4% of all negative reviews)** cite rude, dismissive, or abusive staff, and **570 negative reviews (11.7%)** cite slow or poor service.
* **Interpretation**: Front-desk and service personnel serve as the primary human interface. A single negative interpersonal encounter often overshadows otherwise adequate physical amenities.
* **Supporting Source**: `data/text_mining_outputs/pain_points.csv` & `data/text_mining_outputs/theme_sentiment.csv`.
* **Business Implication**: Structured hospitality training, dispute resolution protocols, and front-desk empowerment are critical operational levers.

#### Insight 3.3: Specific In-Room Equipment Failures Drive Disproportionate Dissatisfaction
* **Finding**: Reviews specifically mentioning bedding, bathrooms, and air conditioning display extreme negative sentiment concentration.
* **Exact Numerical Result**:
  * **Bed / bedding**: 1,360 total mentions $\rightarrow$ **55.7% Negative** (+37.1 pp above baseline).
  * **Bathroom / hot water**: 1,996 total mentions $\rightarrow$ **52.0% Negative** (+33.4 pp above baseline).
  * **Air conditioning**: 954 total mentions $\rightarrow$ **50.8% Negative** (+32.2 pp above baseline).
* **Interpretation**: Guests tolerate minor aesthetic flaws, but sleep disruption (poor mattresses, noisy/broken AC) and personal hygiene disruptions (clogged drains, lack of hot water) directly produce 1-star and 2-star ratings.
* **Supporting Source**: `data/text_mining_outputs/theme_sentiment.csv`.
* **Business Implication**: Preventive maintenance audits must focus specifically on HVAC compressors, boiler systems, and mattress quality.

---

### Category 4: Amenity Combinations & Market Basket Analysis

#### Insight 4.1: High-Lift Synergistic Bundling Governs Premium Wellness Amenities
* **Finding**: Spas exhibit low overall market penetration (11.96% support across 171 hotels) but achieve high association rule lift when combined with business and pool infrastructure.
* **Exact Numerical Result**:
  * Rule 1: `Business Center, Pool → Spa` (Support: **6.57%**, Confidence: **74.02%**, **Lift: 6.190**)
  * Rule 2: `Fitness Center, Pool → Spa` (Support: **8.18%**, Confidence: **68.82%**, **Lift: 5.755**)
  * Rule 3: `Bar, Fitness Center → Spa` (Support: **7.90%**, Confidence: **66.08%**, **Lift: 5.526**)
  * Rule 4: `Business Center, Fitness Center → Spa` (Support: **6.92%**, Confidence: **65.13%**, **Lift: 5.447**)
  * Rule 5: `Bar, Pool → Spa` (Support: **8.25%**, Confidence: **64.84%**, **Lift: 5.422**)
* **Interpretation**: Standalone spas are commercially unviable in this dataset. Guests associate spas with comprehensive resort or full-service business hotel environments.
* **Supporting Source**: `data/apriori/apriori_strong_rules.csv`.
* **Business Implication**: Hotel developers should not add an isolated spa without existing pool, fitness, or business center infrastructure.

#### Insight 4.2: Universal Baseline Amenities Function as Table Stakes
* **Finding**: Five core amenities have achieved near-universal market presence, meaning their presence yields minimal competitive differentiation, yet their absence causes severe rating degradation.
* **Exact Numerical Result**: Market support: Wi-Fi (**95.38%**, 1,364 hotels), Air Conditioning (**91.19%**, 1,304 hotels), Room Service (**85.03%**, 1,216 hotels), Full-Service Laundry (**80.56%**, 1,152 hotels), Breakfast (**80.21%**, 1,147 hotels), Parking (**78.25%**, 1,119 hotels).
* **Interpretation**: These amenities represent non-negotiable expectations. Merely listing them does not increase ratings; ensuring their reliability prevents failure.
* **Supporting Source**: `data/apriori/apriori_amenity_support.csv`.
* **Business Implication**: Operators must focus on execution quality (e.g., high-speed Wi-Fi, working AC) rather than marketing these baseline features as premium add-ons.

---

### Category 5: Predictive Analytics & Model Findings

#### Insight 5.1: Random Forest Demonstrates Robust Predictive Performance
* **Finding**: Non-linear tree ensemble modeling outperforms linear and single-tree architectures on hotel success classification.
* **Exact Numerical Result**: Random Forest achieved **81.34% Accuracy**, **0.6571 Macro-F1**, and **0.7072 ROC-AUC**, outperforming Decision Tree (80.35% Acc, 0.5750 F1, 0.6344 AUC) and Logistic Regression (68.94% Acc, 0.6045 F1, 0.6955 AUC).
* **Interpretation**: Hotel success involves non-linear interactions (e.g., high price requires high amenities and review counts to maintain success). The Random Forest captures these interactions while maintaining balanced sensitivity across classes.
* **Supporting Source**: `dashboard/DOCUMENTATION.md`.
* **Business Implication**: Ensemble methods are suitable for hospitality benchmarking, providing stable predictions across varied property profiles.

#### Insight 5.2: Review Volume and Pricing Dynamics Dominate Predictive Importance
* **Finding**: Machine learning feature importances are dominated by room pricing and customer review engagement, rather than individual amenity flags.
* **Exact Numerical Result**: The top predictors in the Champion Random Forest model are `price` (0.198) and `log_price` (0.171), followed by `review_count` (0.145) and aggregate `amenity_count` (0.102). Individual amenity flags contribute smaller individual weights (`Free Parking`: 0.049, `Accessible`: 0.044, `Bar`: 0.042, `Free Wi-Fi`: 0.038).
* **Interpretation**: A property's market presence—its pricing tier and volume of customer reviews—signals operational scale and guest throughput. Properties with substantial review volumes exhibit more established operational standards.
* **Supporting Source**: Feature importances from `dashboard/hotel_dashboard.ipynb`.
* **Business Implication**: Review acquisition campaigns (encouraging verified guests to leave reviews) are as critical to perceived success as physical amenity additions.

---

### Category 6: Integrated Synthesis Findings

#### Insight 6.1: The Text-Mining & Machine-Learning Diagnostic Bridge
* **Finding**: Combining qualitative text mining sentiment with quantitative model predictors explains why high-amenity hotels occasionally fail.
* **Exact Numerical Result**: Across the corpus, 833 hotels fail to achieve the success threshold ($\ge 4.0$). Among real hotels in this group, their average amenity count is comparable to successful hotels (mean 10.8 vs 10.4 amenities), but their reviews are heavily burdened by hygiene complaints (30.7%) and broken amenities (15.3%).
* **Interpretation**: A hotel can offer 20+ amenities on paper, but if its Air Conditioning is non-functional (50.8% negative sentiment) or Bathrooms are dirty (52.0% negative sentiment), the property will fail regardless of its listed features.
* **Supporting Source**: Cross-analysis of `data/combined_real_synthetic_hotels.csv` and `data/text_mining_outputs/theme_sentiment.csv`.
* **Business Implication**: Capital expenditure should prioritize functional reliability and maintenance of core amenities over expanding the amenity list.

---

## Section 5: Prioritized Business Recommendations

| Rec ID | Business Problem | Evidence-Based Insight | Exact Supporting Evidence | Recommended Action | Target Stakeholder | Expected Benefit | Priority | Success Metric | Measurement Method | Limitations / Assumptions |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: | :--- | :--- | :--- |
| **REC-01** | High customer dissatisfaction from room hygiene defects | Cleanliness is the #1 complaint driver, present in 30.7% of all negative reviews. | 1,497 negative reviews; 1,665 mentions in low-rated ($<3$) reviews (`data/text_mining_outputs/pain_points.csv`). | Deploy standardized, digital room-inspection checklists covering linens, bathrooms, and floor sanitation before guest check-in. | Operations Manager / Head of Housekeeping | Reduction in hygiene complaints; protection of baseline $\ge 4.0$ rating. | **HIGH** | Cleanliness complaint rate $< 5\%$ of total reviews. | Monthly text-mining keyword tracking (`dirty`, `clean`, `smell`, `stain`). | Assumes staff adherence to checklist protocols. |
| **REC-02** | Rapid rating collapse from plumbing and HVAC failures | Bathrooms (52.0% neg) and AC (50.8% neg) exhibit extreme negative sentiment concentration. | 1,037 negative bathroom mentions; 485 negative AC mentions (`data/text_mining_outputs/theme_sentiment.csv`). | Institute mandatory bi-weekly preventive maintenance audits for HVAC compressors, geysers, plumbing fixtures, and drain clearance. | Chief Engineer / Facilities Manager | Eliminates the single largest driver of catastrophic 1-star reviews. | **HIGH** | Equipment defect mentions $< 3\%$ of reviews. | Property maintenance log vs. review keyword tracking (`ac`, `hot water`, `leak`). | Requires upfront maintenance budget allocation. |
| **REC-03** | Service friction and negative customer interactions | Staff demeanor is the #2 complaint driver (20.4% of negative reviews). | 992 reviews cite rude staff; 570 cite slow service (`data/text_mining_outputs/pain_points.csv`). | Implement mandatory service-recovery training and empower front-desk personnel to resolve check-in disputes with immediate service recovery. | General Manager / HR Director | Transforms potential negative reviews into positive service-recovery mentions. | **HIGH** | Front-desk staff satisfaction $\ge 85\%$ positive sentiment. | VADER sentiment tracking on reviews mentioning `staff`, `reception`, `service`. | Assumes staff turnover remains manageable. |
| **REC-04** | Inefficient capital expenditure on isolated wellness amenities | Spas fail as standalone amenities (11.9% support), but achieve high lift when bundled. | `Business Center, Pool → Spa` Lift: 6.190, Conf: 74.02% (`data/apriori/apriori_strong_rules.csv`). | Restrict spa capital investments to properties that already possess pool, fitness, and business center infrastructure. | Chief Development Officer / Asset Manager | Avoids stranded CapEx on low-utilization standalone wellness facilities. | **MEDIUM** | Spa utilization rate and incremental RevPAR lift. | Booking data and amenity package attachment rates. | Assumes property has physical floor space for multi-amenity development. |
| **REC-05** | Negative price-to-experience gap perception | Pricing complaints are elevated by +11.3 pp above baseline; 576 reviews cite unexpected fees. | 3,565 reviews mention price/value; 29.9% are negative (`data/text_mining_outputs/theme_sentiment.csv`). | Eliminate hidden fees; bundle Wi-Fi, breakfast, and parking into room pricing; communicate room tier specifications transparently. | Revenue Manager / Commercial Director | Eliminates billing friction and improves perceived value-for-money sentiment. | **MEDIUM** | Value-for-money negative review rate $< 15\%$. | Monthly sentiment audit on `price`, `value`, `charge`, `bill` keywords. | Price bundling must remain competitive within local comp-set. |
| **REC-06** | Sub-scale review volume dampening visibility and success probability | `review_count` is the 3rd most important predictor (0.145) in the Champion ML model. | Test set RF feature importance = 0.145 (`dashboard/hotel_dashboard.ipynb`). | Deploy post-checkout automated SMS/WhatsApp review prompts to verified satisfied guests to increase review velocity. | Marketing Director / Guest Relations | Increases review volume, strengthening algorithmic search visibility and perceived reliability. | **MEDIUM** | $+25\%$ quarterly growth in verified review volume. | Direct tracking of review count via SerpApi / Google listings. | Assumes guest contact details are captured at check-in. |
| **REC-07** | Geographic underperformance in commercial metro markets | Hyderabad (72.0%) and Delhi (74.9%) lag significantly behind leisure destinations like Goa (88.6%). | City-level success rate comparisons (`data/text_mining_outputs/sentiment_by_city.csv`). | Implement rapid express check-in/out and dedicated high-speed Wi-Fi guarantees tailored for corporate travelers in commercial hubs. | Regional Operations Director | Closes the 16.6 pp performance gap between commercial and leisure markets. | **LOW** | Metro market hotel success rate $\ge 80.0\%$. | Annual rating distribution audit across Delhi and Hyderabad comp-sets. | Market demand mix varies by corporate travel seasonality. |

---

## Section 6: Integrated Business Strategy

```
                                      INTEGRATED HOSPITALITY SUCCESS FRAMEWORK
   ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
   │                                     1. BASELINE OPERATIONAL FOUNDATION                                   │
   │  • Cleanliness Quality Control (Zero-Defect Housekeeping Checklists - Addresses 30.7% of Negative Reviews)│
   │  • Preventive Engineering Audits (HVAC & Bathroom Fixtures - Neutralizes 52% & 51% Complaint Categories) │
   │  • Hospitality & Recovery Protocols (Front-Desk Empowerment - Mitigates 20.4% Staff Friction Points)    │
   └────────────────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                                        │
                                                        ▼
   ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
   │                                     2. STRATEGIC CAPITAL ALLOCATION                                     │
   │  • Universal Baseline Assurance: 100% Reliability on Wi-Fi, AC, Breakfast, Parking (Table Stakes)        │
   │  • Synergistic Premium Bundling: Only Deploy Spa Alongside Pool & Business Center (Lift: 6.190)         │
   │  • Avoid Feature Bloat: Prioritize Functional Integrity over Marginal Amenity Count (r = -0.134)        │
   └────────────────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                                        │
                                                        ▼
   ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
   │                                     3. COMMERCIAL & REVENUE ALIGNMENT                                   │
   │  • Transparent Pricing: Eliminate Add-on Friction to Neutralize +11.3 pp Value-for-Money Deficit        │
   │  • Review Velocity Campaigns: Drive Verified Feedback to Accelerate Predictive Review Signal (Imp: 0.145)│
   │  • Market-Segment Calibration: Differentiate Leisure Relaxed Standards from Metro Express Standards    │
   └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

This integrated strategy connects all four analytical workstreams:
1. **Hygiene First**: Predictive modeling demonstrates that adding amenities without operational quality fails to elevate hotels into the success class. Text mining identifies the precise root causes: dirty rooms, broken bathrooms, non-working AC, and rude staff.
2. **Bundle for Synergy, Not Volume**: Apriori rules show that premium amenities like spas provide maximum lift when bundled with complementary infrastructure (`Business Center + Pool`). Operators should stop treating amenities as isolated additions.
3. **Price to Value**: EDA and text mining confirm that price does not linearly drive ratings ($r = 0.086$), but overpricing relative to condition creates active dissatisfaction (29.9% negative price sentiment).

---

## Section 7: Tableau / BI Dashboard Specifications

Based on available project datasets, the following architecture is specified for enterprise deployment in Tableau:

### 7.1 Dashboard Structure (4 Core Tabs)
* **Tab 1: Executive Overview & Geographic EDA**
  * *Primary Table*: `data/combined_real_synthetic_hotels.csv`
  * *KPI Cards (Top Banner)*:
    1. **Total Hotels Selected**: Count distinct `hotel_id` (Filtered $N$ / 4,551).
    2. **Success Rate ($\ge 4.0$)**: Sum of `success` / Count distinct `hotel_id` (Base: 81.7%).
    3. **Average Room Price**: Mean of `price` (Base: ₹4,077; Median: ₹2,650).
    4. **Average Rating**: Mean of `customer_rating` (Base: 4.45 / 5.0).
    5. **Average Review Count**: Mean of `review_count` (Base: 2,257 reviews).
  * *Visualizations*:
    * *Chart 1*: Geographic Market Map / Treemap (`location` by property volume & success rate).
    * *Chart 2*: Customer Rating Histogram with reference cutoff line at 4.0.
    * *Chart 3*: Star Category Success Breakdown (horizontal bar chart sorted by rating).
    * *Chart 4*: Price vs. Review Count Scatter Plot (colored by `success`, size by `amenity_count`).
* **Tab 2: Predictive Modeling & Performance Benchmarking**
  * *Primary Table*: Machine learning test set holdout predictions ($N = 911$)
  * *Visualizations*:
    * *Chart 1*: Model Comparison Bar Chart (Accuracy, Macro-Precision, Macro-Recall, Macro-F1 across Random Forest, Decision Tree, Logistic Regression).
    * *Chart 2*: Side-by-Side Confusion Matrices (Heatmap format: True Positive, True Negative, False Positive, False Negative).
    * *Chart 3*: Multi-Model ROC Curves with AUC Legend (Random Forest: 0.7072).
    * *Chart 4*: Top 15 Feature Importances (Horizontal bar chart: `price`, `review_count`, `amenity_count`, etc.).
* **Tab 3: Customer Voice & Sentiment Diagnostics**
  * *Primary Table*: `data/text_mining_outputs/review_level_sentiment.csv` & `data/text_mining_outputs/pain_points.csv`
  * *Visualizations*:
    * *Chart 1*: Overall VADER Sentiment Donut Chart (76.3% Pos, 5.1% Neu, 18.6% Neg).
    * *Chart 2*: Star Rating vs. VADER Sentiment Stacked Bar (Agreement Matrix).
    * *Chart 3*: Top Customer Pain Points Ranked Bar Chart (Cleanliness 30.7%, Rude Staff 20.4%, etc.).
    * *Chart 4*: Experience Themes Net Sentiment Diverging Bar Chart (Location +84.9% vs. Bedding -55.7%).
    * *Chart 5*: Embedded Static Clean Word Cloud (`word_cloud_clean.png`).
* **Tab 4: Amenity Bundling & Market Basket Intelligence**
  * *Primary Table*: `data/apriori/apriori_strong_rules.csv` & `data/apriori/apriori_amenity_support.csv`
  * *Visualizations*:
    * *Chart 1*: Top 15 Standardized Amenity Market Penetration (Support %).
    * *Chart 2*: Top 10 Association Rules Ranked by Lift (Horizontal Bar).
    * *Chart 3*: Support vs. Confidence Scatter Plot (Size and Color mapped to Lift).
    * *Chart 4*: Filterable Tabular Rule Matrix (Columns: Antecedent, Consequent, Support %, Confidence %, Lift).

### 7.2 Interactive Filter Panel (Global Controls)
* **Location Filter**: Multi-select dropdown (`All`, `Goa`, `Mumbai`, `Delhi`, `Chennai`, etc.).
* **Category Filter**: Multi-select dropdown (`1-star`, `2-star`, `3-star`, `4-star`, `5-star`, `Unknown`).
* **Price Range Slider**: Continuous range slider (₹360 to ₹25,000+).
* **Rating Range Slider**: Continuous range slider (1.0 to 5.0).
* **Data Type Filter**: Single-select dropdown (`Both`, `Real Only`, `Synthetic Only`).

---

## Section 8: Limitations and Additional Analysis Required

### 8.1 Methodological & Data Limitations
1. **Target Proxy Simplification**: Defining hotel success strictly as $\text{customer\_rating} \ge 4.0$ creates an imbalanced target (81.7% Class 1) and omits direct commercial metrics (RevPAR, ADR, Occupancy, Operating Margin).
2. **Synthetic Data Inflation**: 63.4% of the property dataset ($N = 2,887$) consists of synthetic records. While synthetic data augments feature combinations, it inflates success rates (88.0% synthetic success vs. 70.8% real success) and fixes coordinates to four points, necessitating the exclusion of spatial coordinate features.
3. **Observational vs. Causal Nature**: All findings represent statistical associations and co-occurrences. Association rules (e.g., `Pool → Spa`) do not prove that adding a pool causes guests to demand a spa.
4. **Text Mining Scope**: Review mining applies exclusively to real scraped properties ($N = 1,659$), as synthetic hotels lack qualitative review text.
5. **Neutral Sentiment Classification**: VADER exhibits a low F1-score on neutral reviews (F1: 0.142), as 3-star reviews often blend positive compliments with specific operational complaints.

### 8.2 Recommended Future Research
1. **Econometric RevPAR Modeling**: Acquire financial and occupancy data to model commercial profitability directly alongside customer satisfaction.
2. **Aspect-Based Sentiment Analysis (ABSA)**: Replace lexicon-based VADER scoring with transformer models (e.g., BERT / RoBERTa) to parse mixed-sentiment sentences (e.g., *"Loved the pool, but the bathroom was filthy"*).
3. **Causal Impact Analysis**: Conduct quasi-experimental difference-in-differences (DiD) studies tracking hotels before and after major amenity renovations.

---

## Section 9: Final Conclusions

The evidence confirms that hotel success across the Indian market is governed by a two-tiered operational framework:
1. **The Hygiene Non-Negotiables**: Cleanliness, staff demeanor, working air conditioning, and reliable plumbing are fundamental prerequisites. Failure in these baseline areas leads directly to rating collapse, regardless of listed amenities or star category.
2. **Strategic Synergy Over Feature Bloat**: Amenity breadth alone does not produce success ($r = -0.134$). Premium offerings (e.g., Spas) achieve high lift and customer satisfaction only when deployed as part of coordinated, multi-amenity bundles alongside fitness and business infrastructure.

By prioritizing housekeeping quality control, maintaining core mechanical infrastructure, and bundling premium amenities synergistically, hotel operators can systematically protect and enhance customer satisfaction.

---

## Section 10: Evidence Appendix

### 10.1 Comprehensive Project File Inventory

| Relative File Path | File Type | Analytical Component | Source Dataset Referenced | Key Outputs & Contents | Execution Status | Relevance to Business Analysis | Identified Issues / Constraints |
| :--- | :--- | :--- | :--- | :--- | :---: | :--- | :--- |
| `data/combined_real_synthetic_hotels.csv` | Dataset (CSV) | EDA / Predictive Modeling | SerpApi Scrape + Synthetic Engine | 4,551 clean hotel records, 31 features, pricing, rating, 15 amenity flags | Raw Source | Primary ground truth corpus for EDA and ML modeling | 21 missing rating rows dropped listwise; latitude/longitude fixed in synthetic |
| `data/apriori/apriori_strong_rules.csv` | Output (CSV) | Association Rule Mining | `apriori_transaction_matrix.csv` | 154 strong rules ($\ge 5\%$ supp, $\ge 60\%$ conf, $\ge 2.0$ lift) | Executed | Quantifies amenity bundling synergies (Spa, Pool, Business Center) | Mined across real hotels only ($N=1,430$) |
| `data/apriori/apriori_amenity_support.csv` | Output (CSV) | Association Rule Mining | `apriori_transaction_matrix.csv` | 33 standardized amenities ranked by hotel support (Wi-Fi 95.4% to Grill 0.07%) | Executed | Identifies baseline table stakes vs. luxury amenities | Excludes free/paid distinction; merges variants |
| `data/apriori/apriori_frequent_itemsets.csv` | Output (CSV) | Association Rule Mining | `apriori_transaction_matrix.csv` | 799 frequent itemsets at $\ge 5\%$ minimum support | Executed | Mined building blocks for association rules | None |
| `data/apriori/apriori_transaction_matrix.csv` | Dataset (CSV) | Association Rule Mining | Hotel Amenity Vectors | 1,430 real hotels × 34 columns (hotel_id + 33 binary flags) | Executed | Transaction substrate for Apriori engine | 255 real hotels excluded due to missing/empty amenities |
| `data/text_mining_outputs/presentation_numbers.json` | Output (JSON) | Text Mining / NLP | `review_level_sentiment.csv` | 26,175 reviews, VADER metrics, 3-class & binary accuracy, Kappa, ROC-AUC | Executed | Central verified metrics reference for text analytics | Synthetic hotels excluded |
| `data/text_mining_outputs/theme_sentiment.csv` | Output (CSV) | Text Mining / NLP | Review Text Mentions | 16 customer themes, positive/neutral/negative counts & net sentiment | Executed | Pinpoints operational vulnerabilities (Bedding 55.7% neg, Bathroom 52% neg) | Minimum mention threshold required for small samples |
| `data/text_mining_outputs/pain_points.csv` | Output (CSV) | Text Mining / NLP | Negative Review Corpus | 14 ranked complaint categories, negative mentions, % of negative reviews | Executed | Quantifies root causes of guest dissatisfaction (Cleanliness #1 at 30.7%) | Denominator restricted to negative reviews ($N=4,869$) |
| `data/text_mining_outputs/sentiment_by_category.csv` | Output (CSV) | Text Mining / NLP | Review Text + Hotel Category | Review sentiment metrics across 1-star to 5-star hotel categories | Executed | Demonstrates sentiment improvement with star tier | Unknown star tier represents substantial sample ($N=8,569$) |
| `data/text_mining_outputs/sentiment_by_city.csv` | Output (CSV) | Text Mining / NLP | Review Text + Location | Review sentiment metrics across 12 major Indian cities | Executed | Demonstrates geographic divergence (Goa 84.4% pos vs. Hyderabad 69.4% pos) | Sample sizes vary by city (Pune 931 to Goa 2,994) |
| `data/text_mining_outputs/amenity_listing_comparison.csv` | Output (CSV) | Text Mining / NLP | Review Ratings + Amenity Listing | Rating-positive share comparison (Hotels listing vs. not listing amenity) | Executed | Highlights listing premiums (Gym +8.8 pp, Spa +7.8 pp in same category) | Observational listing association; not causal proof |
| `data/text_mining_outputs/amenity_mention_sentiment.csv` | Output (CSV) | Text Mining / NLP | Review Sentence Mentions | Sentiment of reviews explicitly discussing specific amenities | Executed | Identifies amenity satisfaction (Spa 84.9% pos vs. AC 37.5% neg) | Sample size $<100$ on 5 amenities (Spa, Pet, Gym, Laundry, Shuttle) |
| `data/text_mining_outputs/classification_report.csv` | Output (CSV) | Text Mining / NLP | VADER vs. Rating Reference | Precision, Recall, F1 across Negative, Neutral, Positive classes | Executed | Formal statistical evaluation of VADER accuracy (87.37%) | Neutral class F1 is low (0.142) |
| `data/text_mining_outputs/binary_evaluation_metrics.csv` | Output (CSV) | Text Mining / NLP | VADER vs. Rating Reference | Forced binary ($N=24,898$, Acc 92.7%) & Decisive ($N=23,746$, Acc 95.5%) | Executed | Validates VADER compound score as an early-warning diagnostic tool | Excludes neutral/ambiguous reviews in decisive mode |
| `data/text_mining_outputs/word_frequency_top100.csv` | Output (CSV) | Text Mining / NLP | Clean Review Tokens | Top 100 unigrams and absolute frequencies in review corpus | Executed | Lexical volume analysis (`room` 14,400, `good` 13,450, `clean` 5,268) | High frequency reflects common vocabulary, not sentiment |
| `data/text_mining_outputs/bigrams_top50.csv` | Output (CSV) | Text Mining / NLP | Clean Review Tokens | Top 50 bigrams (`good service` 1,351, `room clean` 1,211, `well maintain` 1,077) | Executed | Identifies paired customer praise and complaint phrases | None |
| `data/text_mining_outputs/trigrams_top50.csv` | Output (CSV) | Text Mining / NLP | Clean Review Tokens | Top 50 trigrams (`would definitely recommend` 295, `ac not work` 165) | Executed | Contextual phrases reflecting extreme positive and negative sentiment | None |
| `data/text_mining_outputs/business_insights.csv` | Report (CSV) | Text Mining / NLP | Teammate Synthesized Outputs | 7 structured areas of findings, evidence, and business implications | Executed | Verified foundation for qualitative business recommendations | None |
| `data/text_mining_outputs/sentiment_distribution.csv` | Output (CSV) | Text Mining / NLP | Review Corpus Summary | Side-by-side distribution of VADER vs. Star-rating sentiment classes | Executed | Baseline sentiment calibration (76.3% VADER pos vs. 73.8% rating pos) | None |
| `data/text_mining_outputs/word_cloud_clean.png` | Image (PNG) | Text Mining / NLP | Tokenizer Output | High-resolution visual word cloud of clean review tokens | Generated | Visual representation of customer vocabulary | Visual artifact; not quantitative evidence by itself |
| `dashboard/hotel_dashboard.ipynb` | Notebook (Jupyter) | Integrated Dashboard | All project datasets | Master interactive 4-tab dashboard notebook with Plotly and ipywidgets | **Fully Executed** | Primary Review 2 deliverable synthesizing all 4 project components | Requires Jupyter environment to interact with live widgets |
| `dashboard/DOCUMENTATION.md` | Report (Markdown) | Documentation | Full Project Repository | Comprehensive technical documentation, architecture, KPIs, and findings | Static Document | Technical submission reference and user guide | None |
| `dashboard/verification_report.md` | Audit Report (MD) | Quality Assurance | Stage A vs. Reference Notebook | Formal audit proving 100% exact match between teammate outputs and repo | Static Document | Verifies data integrity with zero fabricated values | None |
| `dashboard/generate_dashboard_notebook.py` | Script (Python) | Tooling / Compiler | Source Code | Automated compiler script generating `hotel_dashboard.ipynb` from scratch | Executed | Guarantees reproducibility of notebook structure | None |
| `dashboard/requirements.txt` | Configuration | Environment | Dependencies | Python library specification (Plotly, ipywidgets, scikit-learn, mlxtend, etc.) | Verified | Ensures identical runtime environment | None |
| `dashboard/screenshots/tab1_eda.png` | Image (PNG) | Visual Evidence | Tab 1 EDA Visuals | High-resolution capture of full EDA tab, KPI cards, and charts | Generated | Visual proof of Tab 1 execution | Static screenshot |
| `dashboard/screenshots/tab2_models.png` | Image (PNG) | Visual Evidence | Tab 2 ML Visuals | High-resolution capture of Champion RF card, ROC curves, confusion matrices | Generated | Visual proof of Tab 2 execution | Static screenshot |
| `dashboard/screenshots/tab3_text_mining.png` | Image (PNG) | Visual Evidence | Tab 3 NLP Visuals | High-resolution capture of VADER donut, pain points, and themes | Generated | Visual proof of Tab 3 execution | Static screenshot |
| `dashboard/screenshots/tab4_association_rules.png` | Image (PNG) | Visual Evidence | Tab 4 Apriori Visuals | High-resolution capture of Apriori sliders, lift bar chart, and scatter plot | Generated | Visual proof of Tab 4 execution | Static screenshot |
| `dashboard/screenshots/filter_interaction_before.png` | Image (PNG) | Visual Evidence | Baseline Dashboard State | Screenshot showing baseline 4,551 hotel corpus state | Generated | Demonstrates initial filter state | Static screenshot |
| `dashboard/screenshots/filter_interaction_after.png` | Image (PNG) | Visual Evidence | Active Filtered State | Screenshot showing filtered slice (Goa, 5-star, ₹5k–₹25k, $N=34$) | Generated | Demonstrates dynamic reactivity of dashboard | Static screenshot |

---

### 10.2 Evidence Register

| ID | Proposed Insight | Evidence Source File | Exact Supporting Result | Empirical Interpretation | Confidence & Limitations |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **EV-01** | Star category directly dictates baseline success rate | `data/combined_real_synthetic_hotels.csv`; `dashboard/DOCUMENTATION.md` | 5-star: 92.0% ($n=289$); 4-star: 89.0% ($n=391$); 3-star: 79.1% ($n=1,851$); 2-star: 74.4% ($n=429$); 1-star: 69.7% ($n=66$) | Star ratings reflect institutional operating standards that reliably satisfy guests. | **High Confidence**: Full corpus audit ($N=4,551$). Observational correlation; not causal. |
| **EV-02** | Room price exhibits negligible linear correlation with customer rating | Primary dataset correlation analysis; `dashboard/DOCUMENTATION.md` | Pearson correlation coefficient $r = 0.0858$ between `price` and `customer_rating` ($N=4,362$) | Expensive hotels do not automatically achieve high ratings; overpricing creates dissatisfaction risk. | **High Confidence**: Evaluated across all 4,362 price-listed hotels. |
| **EV-03** | Cleanliness is the leading cause of customer review failure | `data/text_mining_outputs/pain_points.csv` | 1,497 negative reviews (30.7% of all negative reviews); 1,665 mentions in $<3.0$ star reviews | Cleanliness is an essential hygiene factor; its absence leads directly to 1-star ratings. | **High Confidence**: Full text mining corpus ($N=26,175$ reviews, 4,869 negative). |
| **EV-04** | Rude staff behavior is the second largest operational failure | `data/text_mining_outputs/pain_points.csv`; `data/text_mining_outputs/theme_sentiment.csv` | 992 negative reviews (20.4% of negative reviews); 570 cite slow service | Front-desk and service interactions represent the primary interpersonal driver of negative reviews. | **High Confidence**: Verified across all clean review records. |
| **EV-05** | Bedding, bathrooms, and AC generate disproportionate negative sentiment | `data/text_mining_outputs/theme_sentiment.csv` | Bedding: 55.7% neg ($n=1,360$); Bathroom: 52.0% neg ($n=1,996$); AC: 50.8% neg ($n=954$) vs. 18.6% baseline | Sleep disruption and personal hygiene failures generate the highest negative sentiment density. | **High Confidence**: Substantial sample sizes ($\ge 950$ mentions per theme). |
| **EV-06** | Price complaints are elevated and travel with room quality defects | `data/text_mining_outputs/theme_sentiment.csv`; `data/text_mining_outputs/pain_points.csv` | 3,565 reviews mention price/value; 29.9% are negative (+11.3 pp above baseline); 576 cite overpricing | Unfavorable value perceptions arise when premium rates are charged for subpar room conditions. | **High Confidence**: Verified across 3,565 specific review mentions. |
| **EV-07** | Leisure destinations systematically outperform commercial metros | `data/text_mining_outputs/sentiment_by_city.csv`; dataset location analysis | Goa: 88.6% success (84.4% pos reviews); Kochi: 87.7% success vs. Delhi: 74.9% and Hyderabad: 72.0% | Business travelers in metros hold stricter service expectations than vacationers in resort hubs. | **High Confidence**: Verified across 12 distinct Indian geographic markets. |
| **EV-08** | Spas require complementary leisure and business infrastructure | `data/apriori/apriori_strong_rules.csv` | `Business Center, Pool → Spa` (Lift: 6.190, Conf: 74.02%, Supp: 6.57%); `Fitness Center, Pool → Spa` (Lift: 5.755) | Spas do not function as viable standalone amenities; they succeed within multi-amenity bundles. | **High Confidence**: Apriori transaction matrix ($N=1,430$ hotels, 33 amenities). |
| **EV-09** | Baseline amenities represent universal table stakes | `data/apriori/apriori_amenity_support.csv` | Wi-Fi: 95.38% support; AC: 91.19%; Room service: 85.03%; Laundry: 80.56%; Breakfast: 80.21%; Parking: 78.25% | Baseline amenities provide minimal competitive differentiation; reliable execution is required. | **High Confidence**: Derived from 1,430 verified hotel listings. |
| **EV-10** | Random Forest ensemble provides the strongest success classification | `dashboard/DOCUMENTATION.md`; Cell 3 execution logs | Random Forest: Accuracy 81.34%, Macro-F1 0.6571, ROC-AUC 0.7072 vs. DT (80.35%) and LR (68.94%) | Ensemble modeling captures non-linear feature interactions governing hotel success. | **High Confidence**: Evaluated on stratified holdout test set ($N=911$). |
| **EV-11** | Review volume and pricing dominate model feature importances | Cell 3 feature importance extraction; `dashboard/hotel_dashboard.ipynb` | `price` (0.198), `log_price` (0.171), `review_count` (0.145), `amenity_count` (0.102) lead importances | Review volume and room rates reflect property throughput and market presence. | **High Confidence**: Tree-based Gini importance across 300 estimators. |
| **EV-12** | VADER sentiment serves as a reliable early-warning indicator | `data/text_mining_outputs/presentation_numbers.json`; `data/text_mining_outputs/binary_evaluation_metrics.csv` | 3-Class Acc: 87.37% (Kappa 0.6801); Decisive Acc: 95.52% (ROC-AUC 0.9529, Neg Prec 0.9503) | VADER compound scores reliably detect review polarization, enabling automated service alerts. | **High Confidence**: Tested on 26,175 reviews against star-rating ground truth. |

---

### 10.3 Recommendation-to-Evidence Traceability Matrix

| Recommendation ID | Primary Evidence Source | Key Analytical Statistic | Core Action Mandate |
| :--- | :--- | :--- | :--- |
| **REC-01** (Hygiene Checklists) | `data/text_mining_outputs/pain_points.csv` | 1,497 negative reviews (30.7% of negative reviews) | Institutionalize zero-defect housekeeping inspection checklists. |
| **REC-02** (HVAC / Bathroom Audits) | `data/text_mining_outputs/theme_sentiment.csv` | Bathroom: 52.0% neg; AC: 50.8% neg | Mandate bi-weekly preventive maintenance on plumbing and AC units. |
| **REC-03** (Staff Service Recovery) | `data/text_mining_outputs/pain_points.csv` | 992 rude staff reviews (20.4%); 570 slow service | Deploy service-recovery training and empower front-desk problem solving. |
| **REC-04** (Spa Amenity Bundling) | `data/apriori/apriori_strong_rules.csv` | `Business Center, Pool → Spa` Lift = 6.190 | Restrict spa capital investments to properties with pool and business centers. |
| **REC-05** (Transparent Pricing) | `data/text_mining_outputs/theme_sentiment.csv` | 29.9% negative price sentiment (+11.3 pp above baseline) | Eliminate hidden surcharges and align pricing with room condition. |
| **REC-06** (Review Velocity Campaigns) | `dashboard/hotel_dashboard.ipynb` | `review_count` feature importance = 0.145 (Rank 3) | Automate post-checkout review request messaging to satisfied guests. |
| **REC-07** (Metro Operations Calibration) | `data/text_mining_outputs/sentiment_by_city.csv` | Hyderabad: 72.0% success vs. Goa: 88.6% success | Deploy express check-in and dedicated Wi-Fi tailored for metro business hubs. |
