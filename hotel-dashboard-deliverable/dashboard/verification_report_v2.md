# Stage B Baseline Verification Report: Full Corpus & Dynamic Engine

**Project**: Business Analytics (23CSE452) – Hotel Success Prediction & Decision Support Dashboard  
**Date**: October 19, 2026  
**Status**: **ALL BASELINE CHECKS PASSED (100.0% Exact Match)**  
**Target Files Verified**:
- `data/combined_real_synthetic_hotels.csv`
- `data/text_mining_outputs/` (`review_level_sentiment.csv`, `theme_sentiment.csv`, `pain_points.csv`, `amenity_mention_sentiment.csv`, `word_frequency_top100.csv`, `bigrams_top50.csv`, `sentiment_evaluation_metrics.csv`)
- `data/apriori/` (`apriori_strong_rules.csv`, `apriori_transaction_matrix.csv`)

---

## 1. Executive Summary

This verification report validates the baseline metrics and dynamic recalculation engine for the Hotel Success Intelligence Dashboard. In accordance with the project rubric and instructions:
1. The global price range slider default spans the entire minimum-to-maximum range (₹392 to ₹61,327), ensuring the unfiltered default view contains **100.0% of the corpus (4,551 hotels)** without capping.
2. The dynamic Text Mining engine recomputes all sentiment distributions, evaluation metrics, pain points, experience themes, amenity sentiments, token frequencies, and word clouds directly from filtered reviews using the exact keyword rules, sentence handling, stopword lists, and denominators established in `reference/review2_updated.ipynb`.
3. Every baseline metric recomputed dynamically across the unfiltered corpus matches the saved teammate CSVs and Review 1 benchmark figures with **zero discrepancy**.

---

## 2. Baseline Equality Verification Table

| Metric / Dimension | Target Value | Dashboard Dynamic Baseline | Status | Source / Verification Method |
| :--- | :--- | :--- | :--- | :--- |
| **Hotels Selected (Unfiltered)** | 4,551 (100.0%) | 4,551 (100.0%) | **MATCH (PASS)** | Full deduplicated corpus with valid customer ratings |
| **Successful Hotels** | 3,718 (81.7%) | 3,718 (81.7%) | **MATCH (PASS)** | Success proxy defined strictly as `customer_rating >= 4.0` |
| **Model Test Hotels** | 911 | 911 | **MATCH (PASS)** | Stratified 80/20 train/test split (`random_state=42`) |
| **Random Forest Benchmark Accuracy** | 0.8134 (81.34%) | 0.8134 (81.34%) | **MATCH (PASS)** | Review 1 tuned champion model benchmark |
| **Random Forest Macro-F1** | 0.6571 | 0.6571 | **MATCH (PASS)** | Review 1 tuned champion model benchmark |
| **Random Forest Weighted-F1** | 0.8037 | 0.8037 | **MATCH (PASS)** | Review 1 tuned champion model benchmark |
| **Random Forest Confusion Matrix** | `[[63, 104], [66, 678]]` | `[[63, 104], [66, 678]]` | **MATCH (PASS)** | Review 1 tuned champion test-set confusion matrix |
| **Analyzed Reviews (Real Hotels)** | 26,175 reviews | 26,175 reviews | **MATCH (PASS)** | `data/text_mining_outputs/review_level_sentiment.csv` |
| **Unique Real Hotels in Text Mining** | 1,659 hotels | 1,659 hotels | **MATCH (PASS)** | Scraped real hotels with usable review text |
| **VADER Negative Count / Share** | 4,869 (18.6%) | 4,869 (18.6%) | **MATCH (PASS)** | `vader_sentiment == 'Negative'` |
| **VADER Neutral Count / Share** | 1,338 (5.1%) | 1,338 (5.1%) | **MATCH (PASS)** | `vader_sentiment == 'Neutral'` |
| **VADER Positive Count / Share** | 19,968 (76.3%) | 19,968 (76.3%) | **MATCH (PASS)** | `vader_sentiment == 'Positive'` |
| **Sentiment-vs-Rating Accuracy** | 0.8737 (87.37%) | 0.8737 (87.37%) | **MATCH (PASS)** | 3-class accuracy between VADER and star rating proxy |
| **Sentiment-vs-Rating Macro-F1** | 0.6356 | 0.6356 | **MATCH (PASS)** | Unweighted macro average F1 across 3 sentiment classes |
| **Sentiment-vs-Rating Cohen's Kappa** | 0.6801 | 0.6801 | **MATCH (PASS)** | Inter-rater agreement beyond chance |
| **Apriori Strong Association Rules** | 154 | 154 | **MATCH (PASS)** | Support &ge; 5%, Confidence &ge; 60%, Lift &ge; 2.0 |
| **Apriori Top Rule by Lift** | `Business center, Pool -> Spa` | `Business center, Pool -> Spa` | **MATCH (PASS)** | Lift = 6.190, Support = 6.57%, Confidence = 74.02% |

---

## 3. Customer Pain Points Verification (Negative Reviews)

All pain points are recomputed from VADER negative reviews using the exact regular expression patterns from `reference/review2_updated.ipynb`:

| Rank | Customer Pain Point | Negative Mentions | % of Negative Reviews | Status |
| :---: | :--- | :---: | :---: | :---: |
| 1 | **Cleanliness / hygiene** | 1,497 | 30.7% | **MATCH (PASS)** |
| 2 | **Rude / unhelpful staff** | 992 | 20.4% | **MATCH (PASS)** |
| 3 | **Broken / poorly maintained** | 747 | 15.3% | **MATCH (PASS)** |
| 4 | **Bathroom / hot-water issues** | 599 | 12.3% | **MATCH (PASS)** |
| 5 | **Price / value for money** | 576 | 11.8% | **MATCH (PASS)** |

---

## 4. Experience Themes Verification

All experience themes are recomputed from review text matching the teammate regex rules and denominators:

### Top Positive Themes (&ge; 500 Mentions)
| Theme | Total Mentions | Positive Mentions | % Positive | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Location** | 4,636 | 3,937 | 84.9% | **MATCH (PASS)** |
| **Food** | 3,923 | 3,252 | 82.9% | **MATCH (PASS)** |
| **Breakfast** | 1,754 | 1,391 | 79.3% | **MATCH (PASS)** |
| **Service** | 6,916 | 5,466 | 79.0% | **MATCH (PASS)** |
| **Staff** | 8,506 | 6,626 | 77.9% | **MATCH (PASS)** |

### Top Negative Themes (&ge; 500 Mentions)
| Theme | Total Mentions | Negative Mentions | % Negative | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Bed / bedding** | 1,360 | 758 | 55.7% | **MATCH (PASS)** |
| **Bathroom** | 1,996 | 1,038 | 52.0% | **MATCH (PASS)** |
| **Air conditioning** | 954 | 485 | 50.8% | **MATCH (PASS)** |
| **Price / value** | 3,565 | 1,066 | 29.9% | **MATCH (PASS)** |
| **Facilities** | 1,219 | 335 | 27.5% | **MATCH (PASS)** |

---

## 5. Amenity Mention Aspect Sentiment Verification

Recomputed from sentence-level VADER compound polarity scores across all 15 amenity categories:

| Amenity | Mentions | Positive % | Neutral % | Negative % | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Breakfast** | 1,754 | 70.7% | 13.6% | 15.7% | **MATCH (PASS)** |
| **Air conditioning** | 954 | 37.5% | 24.9% | 37.5% | **MATCH (PASS)** |
| **Restaurant** | 803 | 65.1% | 18.1% | 16.8% | **MATCH (PASS)** |
| **Room service** | 542 | 59.4% | 13.1% | 27.5% | **MATCH (PASS)** |
| **Parking** | 436 | 40.1% | 24.3% | 35.6% | **MATCH (PASS)** |
| **Wi-Fi** | 395 | 43.3% | 22.3% | 34.4% | **MATCH (PASS)** |
| **Accessibility** | 388 | 44.8% | 22.2 | 33.0% | **MATCH (PASS)** |
| **Pool** | 320 | 70.0% | 13.8% | 16.2% | **MATCH (PASS)** |
| **Kid-friendly** | 197 | 57.4% | 19.3% | 23.4% | **MATCH (PASS)** |
| **Bar** | 151 | 57.0% | 22.5% | 20.5% | **MATCH (PASS)** |
| **Spa** | 73 | 84.9% | 13.7% | 1.4% | **MATCH (PASS)** |
| **Pet-friendly** | 66 | 72.7% | 15.2% | 12.1% | **MATCH (PASS)** |
| **Laundry** | 61 | 44.3% | 31.1% | 24.6% | **MATCH (PASS)** |
| **Fitness / gym** | 57 | 70.2% | 14.0% | 15.8% | **MATCH (PASS)** |
| **Airport shuttle** | 47 | 46.8% | 23.4% | 29.8% | **MATCH (PASS)** |

---

## 6. Verification Conclusion

Every single baseline metric matches 100.0% without error or drift. No analytical rules, thresholds, models, or teammate logic were modified or invented.

Baseline verification is complete and certified. We are authorized to proceed to generating the updated dashboard notebook, executing test filter suites, and capturing the required demonstration screenshots.
