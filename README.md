# Integrated Hotel Analytics for Hotel Success, Amenities & Customer Experience

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)
![Course](https://img.shields.io/badge/Course-23CSE452-purple)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange)
![NLP](https://img.shields.io/badge/NLP-VADER-green)
![Association Mining](https://img.shields.io/badge/Association%20Mining-Apriori-teal)
![Dashboard](https://img.shields.io/badge/Dashboard-Plotly%20%7C%20ipywidgets-blueviolet)
![Team](https://img.shields.io/badge/Team-Team%205-success)

**Business Analytics (23CSE452) – Final Project 2026**  
**Amrita Vishwa Vidyapeetham | Team 5**

An end-to-end Business Analytics project that combines Exploratory Data Analysis, synthetic data generation, predictive machine learning, customer review text mining, VADER sentiment analysis and Apriori Association Rule Mining to understand hotel success, amenities and customer experience across 12 Indian markets.

---

## Table of Contents

1. Project Overview
2. Business Problem
3. Project Objectives
4. Project Architecture
5. Repository Structure
6. Dataset Collection and Description
7. Unit 1 – Exploratory Data Analysis
8. Unit 2 – Predictive Modelling
9. Unit 3 – Text Mining and Sentiment Analysis
10. Unit 3 – Apriori Association Rule Mining
11. Comparative Analysis of Unit 3 Methods
12. Integrated Business Insights
13. Interactive Dashboard
14. Results and Key Metrics
15. Installation and Environment Setup
16. Execution Guide
17. Technologies Used
18. Limitations
19. Academic Team and Individual Contributions
20. Project Management
21. Final Submission Deliverables
22. References

---

## 1. Project Overview

The project investigates the factors associated with hotel success and customer satisfaction by analysing structured hotel information, customer reviews and amenity combinations.

The framework integrates the following analytical methods:

| Unit | Method | Purpose |
|---|---|---|
| Unit 1 | Data Preparation, EDA and Visualization | Understand hotel characteristics, distributions and relationships |
| Unit 2 | Feature Engineering and Predictive Modelling | Identify important features and predict hotel success |
| Unit 3 – Method 1 | Text Mining and Sentiment Analysis | Understand customer opinions, satisfaction and complaints |
| Unit 3 – Method 2 | Apriori Association Rule Mining | Discover frequently occurring combinations of hotel amenities |

CTGAN is additionally used for synthetic tabular data generation, and the analytical results are integrated into an interactive dashboard.

### Analytical Highlights

- 4,551 usable hotel records in the unified real and synthetic dataset.
- 12 Indian markets included in the analysis.
- 26,175 customer reviews analysed using NLP and VADER.
- 33 standardized amenities considered for association rule mining.
- 81.34% accuracy achieved by the tuned Random Forest model.
- 76.3% positive customer sentiment.
- 154 strong association rules discovered using Apriori.
- Highest reported association lift of 6.190.

---

## 2. Business Problem

The hospitality industry is highly competitive, and hotel success depends on several factors, including pricing, service quality, location, amenities and customer satisfaction.

Although hotels collect large amounts of property information and customer feedback, these sources are not always analysed together.

The project addresses the following business challenges:

- Identifying hotel characteristics associated with higher customer ratings.
- Understanding the factors influencing customer satisfaction.
- Extracting recurring complaints from customer reviews.
- Identifying relationships between hotel amenities.
- Understanding how facility combinations differ across hotels.
- Translating analytical findings into practical business recommendations.

The proposed framework integrates structured and unstructured data analysis to provide a broader understanding of hotel performance.

---

## 3. Project Objectives

1. Collect original hotel listing and customer review information across multiple Indian markets.
2. Validate, clean and integrate the collected datasets.
3. Perform exploratory data analysis and visualization.
4. Apply feature engineering and dimensionality reduction.
5. Develop and compare predictive classification models.
6. Apply Text Mining and VADER sentiment analysis to customer reviews.
7. Apply Apriori Association Rule Mining to discover amenity combinations.
8. Compare both Unit 3 methods and derive integrated business insights.
9. Develop an interactive dashboard integrating analytical outputs.
10. Provide actionable recommendations for hotel management.

---

## 4. Project Architecture

```text
             ORIGINAL DATA COLLECTION
                       |
                       v
              SerpApi – Google Hotels
                       |
              ---------------------
              |                   |
              v                   v
        Hotel Listings       Customer Reviews
              |                   |
              v                   v
        Data Cleaning       Text Preprocessing
              |                   |
              v                   v
        CTGAN Generation    Text Mining & VADER
              |                   |
              v                   |
       Unified Hotel Dataset      |
              |                   |
              v                   |
          UNIT 1 – EDA             |
              |                   |
              v                   |
       UNIT 2 – Predictive ML      |
              |                   |
              v                   v
       Hotel Performance     Customer Insights
              |                   |
              v                   v
        UNIT 3 – APRIORI    UNIT 3 – TEXT MINING
              |                   |
              ---------------------
                       |
                       v
              COMPARATIVE ANALYSIS
                       |
                       v
              BUSINESS INSIGHTS
                       |
                       v
              INTERACTIVE DASHBOARD
```

---

## 5. Repository Structure

The GitHub repository is organized into separate directories for datasets, notebooks, dashboard implementation, analytical outputs and final submission documents.

```text
Integrated-Hotel-Analytics-for-Hotel-Success-Amenities-Customer-Experience-/
│
├── dashboard/
│   └── Dashboard submission files and supporting resources
│
├── data/
│   ├── all_india_1685_unique_hotels.csv
│   ├── all_india_hotel_cleaned.csv.zip
│   ├── combined_real_synthetic_hotels.csv
│   └── Other supporting datasets
│
├── hotel-dashboard-deliverable/
│   ├── README.txt
│   │
│   ├── dashboard/
│   │   ├── hotel_dashboard.ipynb
│   │   ├── generate_dashboard_notebook.py
│   │   ├── generate_screenshots.py
│   │   ├── test_dashboard_filters.py
│   │   ├── DOCUMENTATION.md
│   │   ├── USER_GUIDE.md
│   │   ├── verification_report.md
│   │   ├── verification_report_v2.md
│   │   └── screenshots/
│   │       ├── dashboard.png
│   │       ├── dashboard_overview.png
│   │       ├── dashboard_before_filters.png
│   │       ├── dashboard_after_filters.png
│   │       ├── tab1_eda.png
│   │       ├── tab2_models.png
│   │       ├── tab3_text_mining.png
│   │       ├── tab4_association_rules.png
│   │       └── tab5_final_dataset.png
│   │
│   └── data/
│       ├── combined_real_synthetic_hotels.csv
│       ├── apriori/
│       │   ├── apriori_amenity_support.csv
│       │   ├── apriori_frequent_itemsets.csv
│       │   ├── apriori_strong_rules.csv
│       │   └── apriori_transaction_matrix.csv
│       │
│       └── text_mining_outputs/
│           └── Supporting text mining outputs
│
├── notebooks/
│   ├── BA_WebScapring.ipynb
│   ├── hotel_synthetic_data_ctgan_unique_hotel_names (1).ipynb
│   ├── BA_total_Review1_Complete_CORRECTED (2).ipynb
│   ├── Text_Mining_Hotel_Reviews_Durgam_Poojitha (1) (1).ipynb
│   └── review2_updated.ipynb
│
├── text-mining-outputs/
│   ├── presentation_numbers.json
│   ├── review_level_sentiment.csv
│   ├── hotel_level_text_sentiment.csv
│   ├── pain_points.csv
│   ├── theme_sentiment.csv
│   ├── sentiment_by_category.csv
│   ├── sentiment_by_city.csv
│   ├── amenity_mention_sentiment.csv
│   ├── amenity_listing_comparison.csv
│   ├── word_frequency_top100.csv
│   ├── bigrams_top50.csv
│   ├── trigrams_top50.csv
│   ├── classification_report.csv
│   └── Supporting CSV files and analytical outputs
│
├── visualizations/
│   └── text_mining/
│       └── Text Mining charts and figures
│
├── .gitignore
├── README.md
├── Final_Project_Report.pdf
├── Final_Project_Presentation.pptx
├── environment.yml
└── requirements.txt
```

**Repository organization:**

- `dashboard/` – Top-level dashboard submission directory.
- `data/` – Original, cleaned and combined hotel datasets.
- `hotel-dashboard-deliverable/` – Complete master dashboard package, supporting datasets, technical documentation and screenshots.
- `notebooks/` – Implementation notebooks for data collection and all analytical methods.
- `text-mining-outputs/` – Exported NLP results, sentiment metrics and supporting analytical outputs.
- `visualizations/text_mining/` – Text Mining visualization files.
- `Final_Project_Report.pdf` – Final academic project report.
- `Final_Project_Presentation.pptx` – Final project presentation.
- `requirements.txt` – Python dependencies.
- `environment.yml` – Conda environment configuration.

---

## 6. Dataset Collection and Description

### 6.1 Data Collection

The primary hotel dataset was collected using SerpApi with the Google Hotels search engine.

The collection covered 12 Indian markets:

Delhi, Mumbai, Bengaluru, Chennai, Kolkata, Hyderabad, Pune, Jaipur, Goa, Kochi, Coimbatore and Visakhapatnam.

The collected information includes hotel names, location, coordinates, star category, pricing, customer ratings, review counts, amenities and customer reviews.

### 6.2 Dataset Summary

| Dataset | Records |
|---|---:|
| Original hotel listings | 1,685 |
| Usable real hotel records | 1,664 |
| Original customer reviews | 29,987 |
| Clean customer reviews | 26,175 |
| Synthetic hotel records | 2,887 |
| Unified usable hotel records | 4,551 |
| Real hotel transactions for Apriori | 1,430 |

### 6.3 Data Preprocessing

The preprocessing stage included:

- Missing-value analysis.
- Duplicate-record checking.
- Data type conversion.
- Categorical standardization.
- Price and rating validation.
- Amenity standardization.
- Review text cleaning.
- Removal of unusable review records.

### 6.4 Synthetic Data Generation

CTGAN was used to generate 2,887 synthetic hotel records using the original hotel dataset as the training reference.

The unified dataset contains 1,664 usable real records and 2,887 synthetic records.

Synthetic records are distinguished from independently collected observations. They are used for analytical augmentation and are not treated as additional original data.

Synthetic coordinates are excluded from predictive modelling because they are generated around fixed reference locations.

### 6.5 Target Variable

Hotel success is defined using a rating-based proxy:

```python
success = 1 if customer_rating >= 4.0 else 0
```

Customer rating is excluded from the predictor variables to avoid target leakage.

---

## 7. Unit 1 – Exploratory Data Analysis

### Objective

To understand the structure, distributions and relationships in the hotel dataset before applying predictive models.

### Analysis Performed

- Dataset structure and statistical summaries.
- Missing-value and duplicate analysis.
- City-wise hotel distribution.
- Hotel category distribution.
- Price distribution.
- Customer rating distribution.
- Review-count analysis.
- Amenity availability.
- Correlation analysis.
- PCA-based dimensionality exploration.

### Important Findings

Across 4,551 usable hotel records:

- 3,718 hotels (81.7%) meet the rating-based success threshold.
- Price has a weak positive linear correlation with customer rating (r = 0.086).
- Amenity count has a slight negative correlation with customer rating (r = -0.134).

These results indicate that higher prices and larger amenity counts do not automatically translate into higher customer ratings.

### Visualizations

The EDA implementation includes distribution plots, bar charts, histograms, boxplots, correlation heatmaps, city-wise comparisons and PCA visualizations.

---

## 8. Unit 2 – Predictive Modelling

### Objective

To identify hotel characteristics associated with the rating-based success classification.

### Feature Engineering

The model uses 21 predictors, including numerical and categorical variables.

Important features include:

- Price.
- Log-transformed price.
- Review count.
- Amenity count.
- Hotel category.
- Location.
- Individual amenity indicators.

Price transformation:

```python
log_price = np.log1p(price)
```

Customer rating is excluded from the predictor set.

### Train-Test Split

A stratified 80:20 split was used.

- Training records: 3,640.
- Testing records: 911.

### Models Implemented

1. Logistic Regression.
2. Decision Tree.
3. Random Forest.

### Model Comparison

| Model | Accuracy | Balanced Accuracy | Macro-F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 68.94% | 0.6612 | 0.6045 | 0.6955 |
| Decision Tree | 80.35% | 0.5662 | 0.5750 | 0.6344 |
| Random Forest | 81.34% | 0.6443 | 0.6571 | 0.7072 |

### Best Model: Random Forest

The tuned Random Forest achieved the highest accuracy and Macro-F1 among the evaluated models.

Configuration:

- 300 estimators.
- Maximum depth: 10.
- Minimum samples split: 5.
- Balanced class weighting.

Evaluation:

- Accuracy: 81.34%.
- Balanced Accuracy: 0.6443.
- Macro Precision: 0.6777.
- Macro Recall: 0.6443.
- Macro-F1: 0.6571.
- Weighted-F1: 0.8037.
- ROC-AUC: 0.7072.

### Feature Importance

| Feature | Importance |
|---|---:|
| Price | 0.198 |
| Log Price | 0.171 |
| Review Count | 0.145 |
| Amenity Count | 0.102 |
| Free Parking | 0.049 |
| Accessible | 0.044 |
| Bar | 0.042 |
| Free Wi-Fi | 0.038 |

These results show the contribution of different predictors to the Random Forest's decisions. Feature importance does not establish causation.

---

## 9. Unit 3 – Text Mining and Sentiment Analysis

**Implementation:** `notebooks/Text_Mining_Hotel_Reviews_Durgam_Poojitha (1) (1).ipynb`

### Objective

To understand customer experiences, opinions and recurring complaints through unstructured hotel reviews.

### Text Preprocessing

The NLP pipeline includes:

1. Review corpus loading.
2. Missing and unusable review removal.
3. Text normalization.
4. Tokenization.
5. Stop-word removal for frequency analysis.
6. Lemmatization.
7. Clean text generation.

Separate cleaned representations are maintained for frequency and sentiment analysis to preserve sentiment-related information such as negation.

### Word Frequency Analysis

Frequently discussed terms include room, good, stay, hotel, staff, service, experience, clean and location.

### N-Gram Analysis

Bigrams and trigrams were used to identify recurring word combinations and customer expressions.

Examples include:

- Good service.
- Room clean.
- Well maintain.
- Would definitely recommend.
- Staff friendly helpful.
- AC not work.

### VADER Sentiment Analysis

VADER calculates positive, negative, neutral and compound sentiment scores.

Classification:

```python
if compound >= 0.05:
    sentiment = "Positive"
elif compound <= -0.05:
    sentiment = "Negative"
else:
    sentiment = "Neutral"
```

### Sentiment Results

| Sentiment | Percentage |
|---|---:|
| Positive | 76.3% |
| Negative | 18.6% |
| Neutral | 5.1% |

Total reviews analysed: 26,175.

### Customer Pain Points

| Pain Point | Mentions |
|---|---:|
| Cleanliness / Hygiene | 1,497 |
| Rude / Unhelpful Staff | 992 |
| Broken / Poorly Maintained | 747 |
| Bathroom / Hot-Water Issues | 599 |
| Price / Value for Money | 576 |

Cleanliness and hygiene emerged as the most frequently identified complaint category.

A single review may contain multiple complaint categories.

### Rating-Based Validation

| Metric | Result |
|---|---:|
| Three-class Accuracy | 87.37% |
| Weighted F1 | 0.8732 |
| Cohen's Kappa | 0.6801 |
| Spearman Correlation | 0.6157 |

These results measure agreement between VADER classifications and rating-derived sentiment labels.

---

## 10. Unit 3 – Apriori Association Rule Mining

**Implementation:** `notebooks/review2_updated.ipynb`

### Objective

To discover frequently occurring combinations of hotel amenities using Association Rule Mining.

### Transaction Preparation

The Apriori analysis uses:

- 1,430 real hotel transactions.
- 33 standardized amenities.
- Binary amenity representation (1 = present, 0 = absent).

### Apriori Algorithm

The algorithm identifies frequent itemsets and generates association rules using support, confidence and lift.

### Evaluation Metrics

**Support**

\[
Support(A\rightarrow B)=P(A\cup B)
\]

**Confidence**

\[
Confidence(A\rightarrow B)=\frac{Support(A\cup B)}{Support(A)}
\]

**Lift**

\[
Lift(A\rightarrow B)=\frac{Confidence(A\rightarrow B)}{Support(B)}
\]

Lift interpretation:

- Lift greater than 1: Positive association.
- Lift equal to 1: Independence.
- Lift less than 1: Negative association.

### Parameters

| Parameter | Value |
|---|---:|
| Minimum Support | 0.05 |
| Minimum Confidence | 0.60 |
| Minimum Lift | 2.0 |
| Amenities | 33 |
| Transactions | 1,430 |

### Results

- 799 frequent itemsets.
- 154 strong association rules.

### Strongest Associations

| Antecedents | Consequent | Support | Confidence | Lift |
|---|---|---:|---:|---:|
| Business Center, Pool | Spa | 6.57% | 74.02% | 6.190 |
| Fitness Center, Pool | Spa | 8.18% | 68.82% | 5.755 |
| Bar, Fitness Center | Spa | 7.90% | 66.08% | 5.526 |
| Business Center, Fitness Center | Spa | 6.92% | 65.13% | 5.447 |
| Bar, Pool | Spa | 8.25% | 64.84% | 5.422 |

These associations suggest that wellness facilities often occur alongside other business and leisure facilities.

Association rules identify co-occurrence, not causation.

---

## 11. Comparative Analysis of Unit 3 Methods

| Aspect | Text Mining | Apriori |
|---|---|---|
| Data | Customer review text | Hotel amenity transactions |
| Technique | NLP and VADER | Frequent itemset mining |
| Objective | Understand customer experience | Discover amenity combinations |
| Outputs | Sentiment, themes and pain points | Frequent itemsets and rules |
| Metrics | Accuracy, F1, Kappa and sentiment scores | Support, confidence and lift |
| Business application | Service improvement | Facility planning |

Text Mining identifies what customers experience and complain about.

Apriori identifies which amenities frequently occur together.

Combining both methods provides a more complete understanding of hotel performance. Hotels must not only offer suitable facilities but also maintain the quality of their services.

---

## 12. Integrated Business Insights

### 12.1 Improve Cleanliness

Cleanliness and hygiene account for 1,497 negative-review mentions.

Hotels should strengthen housekeeping inspections and maintenance procedures.

### 12.2 Improve Staff Behaviour

Rude and unhelpful staff account for 992 negative-review mentions.

Customer-service training and complaint-handling procedures should be prioritized.

### 12.3 Use Data-Driven Pricing

The weak correlation between price and customer rating indicates that higher pricing alone does not guarantee customer satisfaction.

### 12.4 Plan Complementary Amenity Bundles

Apriori identifies strong relationships among Spa, Pool, Fitness Center, Bar and Business Center.

Hotels can use these patterns when evaluating facility combinations.

### 12.5 Monitor Customer Feedback

Regular sentiment analysis can help identify emerging service problems and recurring complaints.

### 12.6 Integrate Predictive and Descriptive Insights

Predictive models help identify features associated with hotel success, while Text Mining and Apriori provide additional customer and amenity-level insights.

---

## 13. Interactive Dashboard

### Overview

The dashboard integrates analytical results using Python, Plotly and ipywidgets.

**Master implementation:**

`hotel-dashboard-deliverable/dashboard/hotel_dashboard.ipynb`

### Dashboard Components

| Dashboard Section | Functionality |
|---|---|
| Tab 1 – EDA | Hotel distributions, ratings, pricing and amenity analysis |
| Tab 2 – Predictive Models | Model comparison, confusion matrices, ROC curves and feature importance |
| Tab 3 – Text Mining | Sentiment distribution, pain points and thematic analysis |
| Tab 4 – Association Rules | Frequent itemsets, amenity relationships and association metrics |
| Dataset View | Final dataset exploration and export functionality |

### Interactive Features

The dashboard supports dynamic filtering by:

- Location.
- Hotel category.
- Price range.
- Rating range.
- Data type (Real or Synthetic).

The dashboard recalculates relevant visualizations and KPIs based on selected filters.

### Dashboard Evidence

Screenshots and supporting HTML visualizations are maintained in:

`hotel-dashboard-deliverable/dashboard/screenshots/`

Technical documentation:

- `hotel-dashboard-deliverable/dashboard/DOCUMENTATION.md`
- `hotel-dashboard-deliverable/dashboard/USER_GUIDE.md`
- `hotel-dashboard-deliverable/dashboard/verification_report.md`

The dashboard is currently provided as a Jupyter-based interactive application.

---

## 14. Results and Key Metrics

| Component | Result |
|---|---|
| Usable hotel records | 4,551 |
| Indian markets | 12 |
| Real hotel records | 1,664 |
| Synthetic records | 2,887 |
| Clean reviews | 26,175 |
| Apriori transactions | 1,430 |
| Standardized amenities | 33 |
| Random Forest accuracy | 81.34% |
| Random Forest Macro-F1 | 0.6571 |
| Positive sentiment | 76.3% |
| Negative sentiment | 18.6% |
| VADER three-class accuracy | 87.37% |
| Frequent itemsets | 799 |
| Strong association rules | 154 |
| Highest lift | 6.190 |

The combined results demonstrate the value of integrating predictive modelling, customer feedback analysis and amenity relationship discovery.

---

## 15. Installation and Environment Setup

### Requirements

| Requirement | Specification |
|---|---|
| Operating System | macOS, Windows or Linux |
| Python | 3.10–3.12 |
| RAM | 8 GB minimum; 16 GB recommended |
| Notebook Environment | JupyterLab / Jupyter Notebook |
| Browser | Chrome, Firefox, Safari or Edge |

### Clone Using SSH

```bash
git clone git@github.com:DurgamPoojitha/Integrated-Hotel-Analytics-for-Hotel-Success-Amenities-Customer-Experience-.git

cd Integrated-Hotel-Analytics-for-Hotel-Success-Amenities-Customer-Experience-
```

### Create Virtual Environment

```bash
python3 -m venv .venv
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

### Install Dependencies

```bash
python -m pip install --upgrade pip

pip install -r requirements.txt
```

### Download NLTK Resources

```bash
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

---

## 16. Execution Guide

### Step 1 – Open the Notebooks

From the repository root:

```bash
cd notebooks
jupyter lab
```

### Step 2 – Execute the Analytical Notebooks

Recommended execution order:

| Order | Notebook | Purpose |
|---|---|---|
| 1 | BA_WebScapring.ipynb | Hotel data collection |
| 2 | hotel_synthetic_data_ctgan_unique_hotel_names (1).ipynb | Synthetic data generation |
| 3 | BA_total_Review1_Complete_CORRECTED (2).ipynb | EDA and predictive modelling |
| 4 | Text_Mining_Hotel_Reviews_Durgam_Poojitha (1) (1).ipynb | Text Mining and sentiment analysis |
| 5 | review2_updated.ipynb | Integrated Text Mining and Apriori |

Ensure that the required input datasets are available before running each notebook.

### Step 3 – Launch the Master Dashboard

From the repository root:

```bash
cd hotel-dashboard-deliverable/dashboard
```

Run:

```bash
jupyter lab hotel_dashboard.ipynb
```

Execute the notebook cells to render the interactive dashboard.

### Step 4 – Dashboard Testing

```bash
python test_dashboard_filters.py
```

Regenerate the notebook if necessary:

```bash
python generate_dashboard_notebook.py
```

Generate screenshots:

```bash
python generate_screenshots.py
```

---

## 17. Technologies Used

| Technology | Application |
|---|---|
| Python | Main programming language |
| Pandas | Data preprocessing |
| NumPy | Numerical operations |
| Matplotlib | Visualization |
| Seaborn | Exploratory analysis |
| Plotly | Interactive visualizations |
| ipywidgets | Dashboard controls |
| Scikit-learn | Predictive modelling |
| CTGAN | Synthetic data generation |
| NLTK | NLP preprocessing |
| VADER | Sentiment analysis |
| mlxtend | Apriori algorithm |
| Jupyter Notebook | Analytical implementation |
| SerpApi | Data collection |
| Git and GitHub | Version control |

---

## 18. Limitations

### Dataset Limitations

- The dataset covers selected Indian markets rather than the entire hospitality industry.
- Scraped information reflects the availability of listings and reviews during collection.
- Synthetic records cannot replace independently collected real-world observations.

### Predictive Modelling Limitations

- Success is defined using a rating-based proxy.
- Model performance depends on the available features and class distribution.
- Feature importance does not establish causality.
- Performance may vary across new markets and future datasets.

### Text Mining Limitations

- VADER is lexicon-based.
- Sarcasm and context-dependent expressions may be difficult to classify.
- Rating-derived labels may not perfectly reflect review sentiment.

### Apriori Limitations

- Association rules identify co-occurrence rather than causation.
- Results depend on selected thresholds.
- Amenity presence does not establish amenity quality or customer usage.

### Dashboard Limitations

- The application requires a compatible local Python and Jupyter environment.
- Dashboard results depend on the available datasets.
- Live hotel information is not automatically refreshed.

---

## 19. Academic Team and Individual Contributions

This project was developed by Team 5 as part of the Business Analytics (23CSE452) Final Project 2026 at Amrita Vishwa Vidyapeetham.

### Durgam Poojitha

**Register Number:** CB.SC.U4CSE23320

**Contributions:**

- Dataset consolidation and synthetic data generation.
- Predictive modelling and feature engineering.
- Text Mining and NLP preprocessing.
- VADER-based sentiment modelling.
- Customer pain-point identification.
- Thematic extraction and sentiment evaluation.
- Jira task tracking and analytical documentation.

### Anupama Nair

**Register Number:** CB.SC.U4CSE23207

**Contributions:**

- Hotel data collection.
- Data cleaning and preprocessing.
- Dataset documentation.
- Market Basket Analysis.
- Apriori Association Rule Mining.

### Sathya Roopan M

**Register Number:** CB.SC.U4CSE23256

**Contributions:**

- Interactive dashboard development.
- Integration of analytical outputs.
- Implementation of dashboard filters.
- Dashboard testing.
- Dataset collection.

### S. Manish

**Register Number:** CB.SC.U4CSE23125

**Contributions:**

- Amenity analysis using Principal Component Analysis.
- Feature engineering.
- Amenity pattern visualization.
- Dataset collection.

### P. Tirupathi Reddy

**Register Number:** CB.SC.U4CSE23568

**Contributions:**

- Predictive model evaluation.
- Confusion matrix analysis.
- ROC curve analysis.
- Feature importance assessment.
- Business recommendations.
- Dataset collection.

The team collaboratively contributed to analytical integration, dashboard development, interpretation and final project documentation.

---

## 20. Project Management

Project activities were organized and tracked using a project management board.

The board supported coordination and tracking of:

- Dataset collection.
- Data preprocessing.
- Exploratory analysis.
- Predictive modelling.
- Text Mining.
- Apriori Association Rule Mining.
- Dashboard development.
- Documentation and report preparation.
- Final presentation and submission.

**Project Board:** [Insert actual GitHub Projects / Jira board URL]

Individual responsibilities and contributions are documented in Section 19.

---

## 21. Final Submission Deliverables

The repository includes the deliverables required for the Business Analytics Final Project 2026.

| Deliverable | Repository Location |
|---|---|
| Project overview and documentation | `README.md` |
| Original and cleaned datasets | `data/` |
| Analytical notebooks | `notebooks/` |
| Dashboard submission files | `dashboard/` |
| Master dashboard implementation | `hotel-dashboard-deliverable/` |
| Text Mining outputs | `text-mining-outputs/` |
| Visualization files | `visualizations/text_mining/` |
| Final project report | `Final_Project_Report.pdf` |
| Final project presentation | `Final_Project_Presentation.pptx` |
| Python dependencies | `requirements.txt` |
| Environment configuration | `environment.yml` |

All final project deliverables are maintained in the team's public GitHub repository.

---

## 22. References

1. SerpApi – Google Hotels API Documentation. https://serpapi.com/google-hotels-api
2. Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825–2830.
3. Hutto, C. J., & Gilbert, E. (2014). VADER: A Parsimonious Rule-Based Model for Sentiment Analysis of Social Media Text. Proceedings of the International AAAI Conference on Web and Social Media.
4. Agrawal, R., & Srikant, R. (1994). Fast Algorithms for Mining Association Rules. Proceedings of the 20th International Conference on Very Large Data Bases.
5. Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K. (2019). Modeling Tabular Data using Conditional GAN. Advances in Neural Information Processing Systems.
6. Pandas Documentation. https://pandas.pydata.org/docs/
7. Plotly Python Documentation. https://plotly.com/python/
8. NLTK Documentation. https://www.nltk.org/
9. mlxtend Documentation. https://rasbt.github.io/mlxtend/

---

## Project Information

**Project Title:** Integrated Hotel Analytics for Hotel Success, Amenities & Customer Experience

**Course:** Business Analytics (23CSE452)

**Institution:** Amrita Vishwa Vidyapeetham

**Team:** Team 5

**Academic Year:** 2026

**GitHub Repository:**  
https://github.com/DurgamPoojitha/Integrated-Hotel-Analytics-for-Hotel-Success-Amenities-Customer-Experience-

**Developed as part of the Business Analytics Final Project 2026.**
