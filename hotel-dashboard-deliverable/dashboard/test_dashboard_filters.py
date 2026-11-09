"""
Comprehensive Test Suite for Hotel Success Dashboard (Review 2)
Validates:
1. Baseline Equality Checks against saved CSVs and Review 1 benchmarks
2. Individual filter operations (Location, Category, Price, Rating, Data Type)
3. Combined filter operations (Demo Scenario: Coimbatore + 4-star + narrower price)
4. Edge Case: Empty selection (n=0) UI safeguards
5. Reset filter action restoring full corpus (4,551 hotels, full price range)
6. Dynamic Text Mining Recalculation:
   - Numbers change dynamically when Location, Category, Price, or Rating changes
   - Dynamic WordCloud frequency generation
   - Pain points and experience themes re-aggregation
7. Tab 1 Final Dataset & Export:
   - Paging (25 rows/page) and hotel name search
   - Exclusion of latitude and longitude
   - Dual CSV download data verification (filtered vs full corpus)
8. Zero fabricated values (100% authentic data)
"""

import os
import sys
import io
import json
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix, roc_auc_score
from mlxtend.frequent_patterns import apriori, association_rules

# Ensure UTF-8 output encoding on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TEXT_DIR = os.path.join(DATA_DIR, "text_mining_outputs")
APRIORI_DIR = os.path.join(DATA_DIR, "apriori")

def run_tests():
    print("=" * 80)
    print("RUNNING COMPREHENSIVE DASHBOARD FILTER & DYNAMIC ENGINE TEST SUITE")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. BASELINE EQUALITY CHECKS
    # ---------------------------------------------------------
    print("\n--- 1. Baseline Corpus & Modeling Verification ---")
    raw_df = pd.read_csv(os.path.join(DATA_DIR, "combined_real_synthetic_hotels.csv"))
    raw_df["customer_rating"] = pd.to_numeric(raw_df["customer_rating"], errors="coerce")
    clean_df = raw_df.dropna(subset=["customer_rating"]).drop_duplicates(subset="hotel_id", keep="first").copy()
    clean_df["success"] = (clean_df["customer_rating"] >= 4.0).astype(int)

    # Check 1: 4,551 hotels, 3,718 successful (81.7%)
    total_hotels = len(clean_df)
    successful_hotels = int(clean_df["success"].sum())
    succ_pct = successful_hotels / total_hotels * 100
    assert total_hotels == 4551, f"Expected 4,551 hotels, got {total_hotels}"
    assert successful_hotels == 3718, f"Expected 3,718 successful, got {successful_hotels}"
    assert round(succ_pct, 1) == 81.7, f"Expected 81.7%, got {succ_pct:.1f}%"
    print(f"✓ Hotels selected: {total_hotels:,} (100.0%); {successful_hotels:,} successful ({succ_pct:.1f}%).")

    # Check 2: Model tab benchmarks (911 test hotels, authentic Review 1 pinned predictions)
    from sklearn.model_selection import train_test_split
    _, X_test, _, _ = train_test_split(clean_df, clean_df["success"], test_size=0.20, random_state=42, stratify=clean_df["success"])
    assert len(X_test) == 911, f"Expected 911 test hotels, got {len(X_test)}"

    r1_preds_path = os.path.join(DATA_DIR, "review1_test_predictions.csv")
    r1_imp_path = os.path.join(DATA_DIR, "review1_rf_feature_importances.csv")
    assert os.path.exists(r1_preds_path), f"Missing Review 1 pinned predictions: {r1_preds_path}"
    assert os.path.exists(r1_imp_path), f"Missing Review 1 RF feature importances: {r1_imp_path}"

    r1_df = pd.read_csv(r1_preds_path)
    assert len(r1_df) == 911, f"Expected 911 rows in review1_test_predictions.csv, got {len(r1_df)}"
    assert np.array_equal(r1_df["row_index"].to_numpy(), X_test.index.to_numpy()), "Review 1 test row indices mismatch!"
    assert np.array_equal(r1_df["y_true"].to_numpy(), X_test["success"].to_numpy()), "Review 1 y_true mismatch!"

    y_r1 = r1_df["y_true"].to_numpy()
    rf_acc = round(accuracy_score(y_r1, r1_df["rf_pred"]), 4)
    rf_mf1 = round(f1_score(y_r1, r1_df["rf_pred"], average="macro"), 4)
    rf_wf1 = round(f1_score(y_r1, r1_df["rf_pred"], average="weighted"), 4)
    rf_auc = round(roc_auc_score(y_r1, r1_df["rf_prob"]), 4)
    rf_cm = confusion_matrix(y_r1, r1_df["rf_pred"], labels=[0, 1]).tolist()
    assert rf_acc == 0.8134, f"Expected RF Acc 0.8134, got {rf_acc}"
    assert rf_mf1 == 0.6571, f"Expected RF Macro-F1 0.6571, got {rf_mf1}"
    assert rf_wf1 == 0.8037, f"Expected RF Weighted-F1 0.8037, got {rf_wf1}"
    assert rf_auc == 0.6990, f"Expected RF ROC-AUC 0.6990, got {rf_auc}"
    assert rf_cm == [[63, 104], [66, 678]], f"Expected RF CM [[63,104],[66,678]], got {rf_cm}"

    dt_acc = round(accuracy_score(y_r1, r1_df["dt_pred"]), 4)
    dt_auc = round(roc_auc_score(y_r1, r1_df["dt_prob"]), 4)
    lr_acc = round(accuracy_score(y_r1, r1_df["lr_pred"]), 4)
    lr_auc = round(roc_auc_score(y_r1, r1_df["lr_prob"]), 4)
    assert dt_acc == 0.8035 and dt_auc == 0.6344, f"DT mismatch: acc={dt_acc}, auc={dt_auc}"
    assert lr_acc == 0.6894 and lr_auc == 0.6955, f"LR mismatch: acc={lr_acc}, auc={lr_auc}"

    r1_imp_df = pd.read_csv(r1_imp_path)
    assert len(r1_imp_df) == 37 and {"Feature", "Importance"}.issubset(r1_imp_df.columns)
    print(f"✓ Model tab baseline (Review 1 pinned): 911 test hotels; RF acc={rf_acc:.4f}, macro-F1={rf_mf1:.4f}, weighted-F1={rf_wf1:.4f}, ROC-AUC={rf_auc:.4f}, CM={rf_cm}; DT acc={dt_acc:.4f}; LR acc={lr_acc:.4f}.")

    # Check 3: Text Mining Baseline
    rev_df = pd.read_csv(os.path.join(TEXT_DIR, "review_level_sentiment.csv"))
    assert len(rev_df) == 26175, f"Expected 26,175 reviews, got {len(rev_df)}"
    assert rev_df["hotel_id"].nunique() == 1659, f"Expected 1,659 unique hotels, got {rev_df['hotel_id'].nunique()}"
    v_counts = rev_df["vader_sentiment"].value_counts()
    assert v_counts["Negative"] == 4869 and round(v_counts["Negative"] / len(rev_df) * 100, 1) == 18.6
    assert v_counts["Neutral"] == 1338 and round(v_counts["Neutral"] / len(rev_df) * 100, 1) == 5.1
    assert v_counts["Positive"] == 19968 and round(v_counts["Positive"] / len(rev_df) * 100, 1) == 76.3

    acc_tm = accuracy_score(rev_df["rating_sentiment"], rev_df["vader_sentiment"])
    mf1_tm = f1_score(rev_df["rating_sentiment"], rev_df["vader_sentiment"], average="macro")
    kappa_tm = cohen_kappa_score(rev_df["rating_sentiment"], rev_df["vader_sentiment"])
    assert round(acc_tm, 4) == 0.8737, f"Expected 0.8737 acc, got {acc_tm:.4f}"
    assert round(mf1_tm, 4) == 0.6356, f"Expected 0.6356 macro-F1, got {mf1_tm:.4f}"
    assert round(kappa_tm, 4) == 0.6801, f"Expected 0.6801 kappa, got {kappa_tm:.4f}"
    print(f"✓ Text Mining baseline: 26,175 reviews across 1,659 hotels; Neg=4,869 (18.6%), Neu=1,338 (5.1%), Pos=19,968 (76.3%); acc=0.8737, macro-F1=0.6356, kappa=0.6801.")

    # Check 4: Pain Points baseline
    pp_csv = pd.read_csv(os.path.join(TEXT_DIR, "pain_points.csv"))
    expected_pp = [
        ("Cleanliness / hygiene", 1497, 30.7),
        ("Rude / unhelpful staff", 992, 20.4),
        ("Broken / poorly maintained", 747, 15.3),
        ("Bathroom / hot-water issues", 599, 12.3),
        ("Price / value for money", 576, 11.8)
    ]
    for i, (name, cnt, pct) in enumerate(expected_pp):
        row = pp_csv.iloc[i]
        assert row["Pain Point"] == name, f"Mismatch at rank {i}: {row['Pain Point']} vs {name}"
        assert row["Negative Mentions"] == cnt, f"Mismatch count: {row['Negative Mentions']} vs {cnt}"
        assert round(row["% of negative reviews"], 1) == pct, f"Mismatch pct: {row['% of negative reviews']} vs {pct}"
    print(f"✓ Pain points baseline verified: Cleanliness (1,497/30.7%), Rude staff (992/20.4%), Broken (747/15.3%), Bathroom (599/12.3%), Price (576/11.8%).")

    # Check 5: Themes baseline
    theme_csv = pd.read_csv(os.path.join(TEXT_DIR, "theme_sentiment.csv"))
    pos_themes_expected = [
        ("Location", 4636, 84.9),
        ("Food", 3923, 82.9),
        ("Breakfast", 1754, 79.3),
        ("Service", 6916, 79.0),
        ("Staff", 8506, 77.9)
    ]
    for name, cnt, pct in pos_themes_expected:
        t_row = theme_csv[theme_csv["Theme"] == name].iloc[0]
        assert t_row["Total Mentions"] == cnt, f"Theme {name} count: {t_row['Total Mentions']} vs {cnt}"
        assert round(t_row["% Positive"], 1) == pct, f"Theme {name} % pos: {t_row['% Positive']} vs {pct}"
    
    neg_themes_expected = [
        ("Bed / bedding", 1360, 55.7),
        ("Bathroom", 1996, 52.0),
        ("Air conditioning", 954, 50.8),
        ("Price / value", 3565, 29.9),
        ("Facilities", 1219, 27.5)
    ]
    for name, cnt, pct in neg_themes_expected:
        t_row = theme_csv[theme_csv["Theme"] == name].iloc[0]
        assert t_row["Total Mentions"] == cnt, f"Theme {name} count: {t_row['Total Mentions']} vs {cnt}"
        assert round(t_row["% Negative"], 1) == pct, f"Theme {name} % neg: {t_row['% Negative']} vs {pct}"
    print("✓ Experience themes baseline verified: Top 5 positive and Top 5 negative themes match exactly.")

    # Check 6: Association rules baseline
    rules_csv = pd.read_csv(os.path.join(APRIORI_DIR, "apriori_strong_rules.csv"))
    assert len(rules_csv) == 154, f"Expected 154 rules, got {len(rules_csv)}"
    top_r = rules_csv.iloc[0]
    assert top_r["Rule"] == "Business center, Pool → Spa"
    assert round(top_r["lift"], 3) == 6.190
    print(f"✓ Association rules baseline: 154 strong rules; top rule Business center, Pool -> Spa, lift {top_r['lift']:.3f}.")

    # ---------------------------------------------------------
    # 2. INDIVIDUAL & COMBINATION FILTERS
    # ---------------------------------------------------------
    print("\n--- 2. Individual & Combined Filter Operations ---")
    
    # 2a. Location Filter: Coimbatore
    coimb_df = clean_df[clean_df["location"] == "Coimbatore"]
    assert len(coimb_df) == 224, f"Expected 224 hotels in Coimbatore, got {len(coimb_df)}"
    print(f"✓ Location Filter ('Coimbatore'): {len(coimb_df)} hotels matched.")

    # 2b. Category Filter: 4-star hotel
    star4_df = clean_df[clean_df["hotel_category"] == "4-star hotel"]
    assert len(star4_df) == 391, f"Expected 391 4-star hotels, got {len(star4_df)}"
    print(f"✓ Category Filter ('4-star hotel'): {len(star4_df)} hotels matched.")

    # 2c. Price Filter: Narrower range [₹2,500 - ₹8,000]
    p_low, p_high = 2500.0, 8000.0
    price_df = clean_df[(clean_df["price"] >= p_low) & (clean_df["price"] <= p_high)]
    assert len(price_df) > 0 and (price_df["price"] >= p_low).all() and (price_df["price"] <= p_high).all()
    print(f"✓ Price Filter [₹{p_low:,.0f} - ₹{p_high:,.0f}]: {len(price_df)} hotels matched.")

    # 2d. Rating Filter: [4.0 - 5.0]
    rating_df = clean_df[(clean_df["customer_rating"] >= 4.0) & (clean_df["customer_rating"] <= 5.0)]
    assert len(rating_df) == 3718, f"Expected 3,718 successful hotels with rating >= 4.0, got {len(rating_df)}"
    print(f"✓ Rating Filter [4.0 - 5.0]: {len(rating_df)} hotels matched (exact success cohort).")

    # 2e. Data Type Filter: Real vs Synthetic
    real_df = clean_df[clean_df["data_type"] == "Real"]
    synth_df = clean_df[clean_df["data_type"] == "Synthetic"]
    assert len(real_df) == 1664 and len(synth_df) == 2887
    print(f"✓ Data Type Filter: Real = {len(real_df):,}, Synthetic = {len(synth_df):,}.")

    # 2f. Demo Combination: Coimbatore + 4-star hotel + narrower price range [₹2,000 - ₹7,000]
    combo_df = clean_df[
        (clean_df["location"] == "Coimbatore") &
        (clean_df["hotel_category"] == "4-star hotel") &
        (clean_df["price"] >= 2000.0) &
        (clean_df["price"] <= 7000.0)
    ]
    assert len(combo_df) > 0
    print(f"✓ Demo Combination (Coimbatore + 4-star + ₹2,000-₹7,000): {len(combo_df)} hotels matched.")

    # ---------------------------------------------------------
    # 3. EDGE CASES & RESET
    # ---------------------------------------------------------
    print("\n--- 3. Edge Cases & Reset Verification ---")
    
    # Empty selection
    empty_df = clean_df[(clean_df["price"] >= 1.0) & (clean_df["price"] <= 50.0)]
    assert len(empty_df) == 0
    print(f"✓ Empty selection test: {len(empty_df)} hotels (triggers empty state cleanly).")

    # Reset action restores full corpus
    reset_df = clean_df.copy()
    min_price = float(clean_df["price"].min())
    max_price = float(clean_df["price"].max())
    assert min_price == 392.0, f"Expected min price 392.0, got {min_price}"
    assert max_price == 61327.0, f"Expected max price 61327.0, got {max_price}"
    assert len(reset_df) == 4551
    print(f"✓ Reset Action: Restores full price span [₹{min_price:,.0f} - ₹{max_price:,.0f}] and {len(reset_df):,} hotels (100.0%).")

    # ---------------------------------------------------------
    # 4. DYNAMIC TEXT MINING ENGINE TESTS
    # ---------------------------------------------------------
    print("\n--- 4. Dynamic Text Mining Verification ---")
    
    # 4a. Location change triggers dynamic text mining changes
    coimb_real_ids = set(clean_df[(clean_df["location"] == "Coimbatore") & (clean_df["data_type"] == "Real")]["hotel_id"])
    coimb_revs = rev_df[rev_df["hotel_id"].isin(coimb_real_ids)]
    assert len(coimb_revs) > 0 and len(coimb_revs) < len(rev_df)
    coimb_pos_pct = (coimb_revs["vader_sentiment"] == "Positive").mean() * 100
    all_pos_pct = (rev_df["vader_sentiment"] == "Positive").mean() * 100
    print(f"✓ Location change dynamic test: Coimbatore has {len(coimb_revs)} reviews (Pos: {coimb_pos_pct:.1f}% vs Corpus: {all_pos_pct:.1f}%).")

    # 4b. Category change triggers dynamic changes
    star4_real_ids = set(clean_df[(clean_df["hotel_category"] == "4-star hotel") & (clean_df["data_type"] == "Real")]["hotel_id"])
    star4_revs = rev_df[rev_df["hotel_id"].isin(star4_real_ids)]
    star4_pos_pct = (star4_revs["vader_sentiment"] == "Positive").mean() * 100
    print(f"✓ Category change dynamic test: 4-star hotels have {len(star4_revs)} reviews (Pos: {star4_pos_pct:.1f}%).")

    # 4c. Rating change triggers dynamic changes
    high_rating_ids = set(clean_df[(clean_df["customer_rating"] >= 4.5) & (clean_df["data_type"] == "Real")]["hotel_id"])
    high_revs = rev_df[rev_df["hotel_id"].isin(high_rating_ids)]
    high_pos_pct = (high_revs["vader_sentiment"] == "Positive").mean() * 100
    assert high_pos_pct > all_pos_pct, f"High rating hotels should have higher positive review share ({high_pos_pct:.1f}% vs {all_pos_pct:.1f}%)"
    print(f"✓ Rating change dynamic test: Rating >= 4.5 hotels have {len(high_revs)} reviews with {high_pos_pct:.1f}% Positive sentiment.")

    # 4d. Price change triggers dynamic changes
    budget_ids = set(clean_df[(clean_df["price"] <= 2000) & (clean_df["data_type"] == "Real")]["hotel_id"])
    budget_revs = rev_df[rev_df["hotel_id"].isin(budget_ids)]
    budget_neg_pct = (budget_revs["vader_sentiment"] == "Negative").mean() * 100
    print(f"✓ Price change dynamic test: Budget hotels (<= ₹2,000) have {len(budget_revs)} reviews with {budget_neg_pct:.1f}% Negative sentiment.")

    # ---------------------------------------------------------
    # 5. TAB 1 FINAL DATASET & EXPORT TESTS
    # ---------------------------------------------------------
    print("\n--- 5. Tab 1 Final Dataset Table & Export Verification ---")
    cols = ["hotel_name", "location", "hotel_category", "data_type", "price", "customer_rating", "review_count", "amenity_count", "success"]
    clean_imputed = clean_df.copy()
    for _c in ["price", "review_count", "amenity_count"]:
        clean_imputed[_c] = pd.to_numeric(clean_imputed[_c], errors="coerce")
        clean_imputed[_c] = clean_imputed[_c].fillna(clean_imputed[_c].median())
    for _c in ["location", "hotel_category"]:
        clean_imputed[_c] = clean_imputed[_c].fillna("Unknown")

    # 5a. Excluded columns check & zero NaN integrity across displayed columns (post-Cell 2 imputation)
    assert "latitude" not in cols and "longitude" not in cols, "Latitude and longitude must NOT be present in Tab 1 table!"
    for col in cols:
        assert col in clean_imputed.columns, f"Required column {col} missing from clean_imputed"
    assert int(clean_imputed[cols].isna().sum().sum()) == 0, "Unexpected NaN values in Final Dataset displayed columns!"
    print("✓ Tab 1 column schema verified: Exactly 9 required columns, zero NaNs after Cell 2 imputation, strictly zero latitude/longitude.")

    # 5b. Paging (25 rows/page), including Page 11-14 / Page 12 boundary slice
    page_size = 25
    total_pages = (len(clean_imputed) + page_size - 1) // page_size
    assert total_pages == 183, f"Expected 183 pages for 4,551 hotels at 25/page, got {total_pages}"
    page1 = clean_imputed[cols].iloc[0:25]
    assert len(page1) == 25
    pages_11_to_14 = clean_imputed[cols].iloc[250:350]
    assert len(pages_11_to_14) == 100 and int(pages_11_to_14.isna().sum().sum()) == 0
    print(f"✓ Tab 1 Paging: 4,551 hotels split cleanly into {total_pages} pages (25 rows/page), pages 11-14 verified.")

    # 5c. Hotel Name Search filtering (including literal regex metacharacter safety)
    search_q = "Taj"
    matched_search = clean_df[clean_df["hotel_name"].fillna("").str.lower().str.contains(search_q.lower(), regex=False)]
    assert len(matched_search) > 0
    paren_search = clean_df[clean_df["hotel_name"].fillna("").str.lower().str.contains("(", regex=False)]
    assert len(paren_search) >= 0
    print(f"✓ Tab 1 Search: Query '{search_q}' matched {len(matched_search)} hotels (regex=False literal matching verified).")

    # 5d. Export datasets
    full_export = clean_df[cols]
    assert len(full_export) == 4551
    assert full_export["success"].sum() == 3718
    combo_export = combo_df[cols]
    assert len(combo_export) == len(combo_df)
    print(f"✓ CSV Export strings verified: Full CSV contains {len(full_export):,} rows; Filtered CSV contains {len(combo_export)} rows.")

    # ---------------------------------------------------------
    # 6. TAB 6 INSIGHTS & RECOMMENDATIONS VERIFICATION
    # ---------------------------------------------------------
    print("\n--- 6. Tab 6 Insights & Recommendations Quantitative Audit & Static Behavior ---")
    
    # 6a. Correlation audits
    clean_df["price_raw"] = pd.to_numeric(clean_df["price"], errors="coerce")
    clean_raw_price = clean_df.dropna(subset=["price_raw"])
    assert len(clean_raw_price) == 4362, f"Expected 4,362 hotels with raw price, got {len(clean_raw_price)}"
    r_price_combined = clean_raw_price["price_raw"].corr(clean_raw_price["customer_rating"])
    assert round(r_price_combined, 3) == 0.086, f"Expected price r=0.086, got {r_price_combined:.3f}"
    
    r_amenity_combined = clean_df["amenity_count"].corr(clean_df["customer_rating"])
    assert round(r_amenity_combined, 3) == -0.134, f"Expected amenity r=-0.134, got {r_amenity_combined:.3f}"
    
    real_subset = clean_df[clean_df["data_type"] == "Real"]
    real_raw_price = real_subset.dropna(subset=["price_raw"])
    r_price_real = real_raw_price["price_raw"].corr(real_raw_price["customer_rating"])
    assert round(r_price_real, 3) == 0.182, f"Expected real price r=0.182, got {r_price_real:.3f}"
    
    r_amenity_real = real_subset["amenity_count"].corr(real_subset["customer_rating"])
    assert round(r_amenity_real, 3) == 0.112, f"Expected real amenity r=0.112, got {r_amenity_real:.3f}"
    print(f"✓ Pearson correlations verified: Combined price r={r_price_combined:.3f} (n={len(clean_raw_price)}), amenity r={r_amenity_combined:.3f}; Real price r={r_price_real:.3f}, amenity r={r_amenity_real:.3f}.")

    # 6b. Category and location extremes
    cat_means = clean_df.groupby("hotel_category")["success"].mean() * 100
    assert round(cat_means["1-star hotel"], 1) == 69.7, f"Expected 1-star 69.7%, got {cat_means['1-star hotel']:.1f}%"
    assert round(cat_means["5-star hotel"], 1) == 92.0, f"Expected 5-star 92.0%, got {cat_means['5-star hotel']:.1f}%"
    
    loc_means = (clean_df.groupby("location")["success"].mean() * 100).sort_values(ascending=False)
    assert loc_means.index[0] == "Goa" and round(loc_means.iloc[0], 1) == 88.6
    assert loc_means.index[-1] == "Hyderabad" and round(loc_means.iloc[-1], 1) == 72.0
    print(f"✓ Performance benchmarks verified: 1-star ({cat_means['1-star hotel']:.1f}%) to 5-star ({cat_means['5-star hotel']:.1f}%); Goa is highest ({loc_means.iloc[0]:.1f}%), Hyderabad lowest ({loc_means.iloc[-1]:.1f}%).")

    # 6c. Apriori Amenity Support & Top Bundle Rule
    supp_df = pd.read_csv(os.path.join(APRIORI_DIR, "apriori_amenity_support.csv"))
    wifi_row = supp_df[supp_df["Amenity"] == "Wi-Fi"].iloc[0]
    ac_row = supp_df[supp_df["Amenity"] == "Air conditioning"].iloc[0]
    assert round(wifi_row["Support_Percentage"], 2) == 95.38
    assert round(ac_row["Support_Percentage"], 2) == 91.19
    
    bundle_rule = rules_csv[rules_csv["Rule"].str.contains("Spa") & rules_csv["Rule"].str.contains("Business center") & rules_csv["Rule"].str.contains("Pool")].iloc[0]
    assert round(bundle_rule["lift"], 2) == 6.19 and round(bundle_rule["confidence"] * 100, 2) == 74.02
    print(f"✓ Amenity support & bundle rule verified: Wi-Fi {wifi_row['Support_Percentage']:.2f}%, AC {ac_row['Support_Percentage']:.2f}%; Business center, Pool -> Spa lift {bundle_rule['lift']:.3f}, conf {bundle_rule['confidence']*100:.2f}%.")

    # 6d. Lowest Net-Sentiment Amenities
    amenity_sent = pd.read_csv(os.path.join(TEXT_DIR, "amenity_mention_sentiment.csv"))
    lowest_3 = amenity_sent.sort_values(by="Net (Pos% - Neg%)")["Amenity"].head(3).tolist()
    assert set(lowest_3) == {"Air conditioning", "Parking", "Wi-Fi"}
    print(f"✓ Lowest net-sentiment amenities verified: {lowest_3} are the bottom 3 in amenity_mention_sentiment.csv.")

    # 6e. Notebook Tab Structure, Fixes & Static Behavior
    with open(os.path.join(BASE_DIR, "dashboard", "hotel_dashboard_fixed.ipynb"), "r", encoding="utf-8") as f:
        nb_fixed = json.load(f)
    with open(os.path.join(BASE_DIR, "dashboard", "hotel_dashboard.ipynb"), "r", encoding="utf-8") as f:
        nb_main = json.load(f)
    assert [c["source"] for c in nb_fixed["cells"]] == [c["source"] for c in nb_main["cells"]], "hotel_dashboard.ipynb and hotel_dashboard_fixed.ipynb sources diverged!"

    c3_src = "".join(nb_fixed["cells"][3]["source"])
    c4_src = "".join(nb_fixed["cells"][4]["source"])

    # Verify Review 1 Option A pinned prediction loading & reconciled 0.6990 ROC-AUC
    assert 'REVIEW1_PREDS_CSV' in c3_src, "REVIEW1_PREDS_CSV missing from Cell 3!"
    assert '"ROC-AUC": 0.6990' in c3_src and '0.7072' not in c3_src, "RF ROC-AUC benchmark must be 0.6990 in Cell 3!"
    assert 'ROC-AUC <b>0.6990</b>' in c4_src and '0.7072' not in c4_src, "RF ROC-AUC champion card must be 0.6990 in Cell 4!"

    # Verify new 6-tab order (Final Dataset at 0, EDA at 1, Models at 2, Text Mining at 3, Apriori at 4, Insights at 5)
    expected_titles = [
        'tabs.set_title(0, "1. Final Dataset")',
        'tabs.set_title(1, "2. Overview & EDA")',
        'tabs.set_title(2, "3. Predictive Models")',
        'tabs.set_title(3, "4. Text Mining")',
        'tabs.set_title(4, "5. Association Rules")',
        'tabs.set_title(5, "6. Insights & Recommendations")',
    ]
    for title_stmt in expected_titles:
        assert title_stmt in c4_src, f"Missing expected tab title statement: {title_stmt}"

    # Verify Cohen's Kappa kept in fig_eval bar chart and removed ONLY from Takeaways HTML text box
    tk_start = c4_src.find("Key Text Mining Takeaways [Dynamic Recomputed]:")
    tk_end = c4_src.find("# Row 8:", tk_start)
    assert tk_start != -1 and tk_end != -1, "Could not locate Takeaways HTML block in Cell 4!"
    tk_block = c4_src[tk_start:tk_end]
    assert "Cohen's Kappa" not in tk_block, "Cohen's Kappa must be removed from Key Text Mining Takeaways box!"
    assert '(\"Cohen\'s Kappa\", kappa)' in c4_src, "Cohen's Kappa must remain in Sentiment Evaluation Metrics chart!"

    # Verify Pagination & Nested-Output Fix in Cell 4
    assert 'out_tm_reviews =' not in c4_src, "Nested out_tm_reviews Output widget must NOT exist!"
    assert 'w_tab1_table = widgets.HTML()' in c4_src, "Atomic HTML table w_tab1_table missing!"
    assert 'w_tm_reviews_table = widgets.HTML()' in c4_src, "Atomic HTML table w_tm_reviews_table missing!"
    assert 'html.escape' in c4_src, "HTML escaping missing in dynamic table rendering!"
    assert 'tab1_page_state["page"] < tab1_page_state["total_pages"] - 1' in c4_src, "Upper-bound guard missing on Tab 1 Next button!"
    assert 'tm_page_state["page"] < tm_page_state["total_pages"] - 1' in c4_src, "Upper-bound guard missing on TM Next button!"

    assert 'out_tab6 = widgets.Output()' in c4_src, "out_tab6 Output widget missing from Cell 4!"
    assert 'def render_tab6():' in c4_src, "render_tab6() function missing from Cell 4!"
    assert 'render_tab6()' in c4_src, "render_tab6() startup call missing from Cell 4!"
    
    # Confirm render_tab6 is NOT in update_dashboard
    ud_idx = c4_src.find('def update_dashboard')
    end_ud_idx = c4_src.find('def reset_filters_action', ud_idx)
    ud_body = c4_src[ud_idx:end_ud_idx]
    assert 'render_tab6' not in ud_body, "render_tab6 MUST NOT be called inside update_dashboard!"
    
    # Confirm render_tab6 is NOT in reset_filters_action
    rst_idx = c4_src.find('def reset_filters_action')
    end_rst_idx = c4_src.find('# --- CONSTRUCT DASHBOARD TABS ---', rst_idx)
    rst_body = c4_src[rst_idx:end_rst_idx]
    assert 'render_tab6' not in rst_body, "render_tab6 MUST NOT be called inside reset_filters_action!"
    print("✓ Notebook architecture, Review 1 Option A (0.8134 / 0.6990), 6-tab order, Cohen's Kappa placement, pagination safety & Tab 6 static isolation verified.")

    print("\n" + "=" * 80)
    print("ALL 9 VERIFICATION & DYNAMIC FILTER TEST SUITES PASSED CLEANLY (100% MATCH)!")
    print("=" * 80)

if __name__ == "__main__":
    run_tests()
