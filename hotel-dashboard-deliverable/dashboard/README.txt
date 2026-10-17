================================================================================
BUSINESS ANALYTICS (23CSE452) - REVIEW 2: INTEGRATED HOTEL DASHBOARD
PROJECT DELIVERABLE & FILE GUIDE (Dashboard Integrator)
================================================================================

This package contains the complete, self-contained Review 2 interactive dashboard 
integrating Exploratory Data Analysis (EDA), Predictive ML Models (Review 1), 
Text Mining and Apriori Association Rules.

--------------------------------------------------------------------------------
DASHBOARD FOLDER (dashboard/)
--------------------------------------------------------------------------------

1. hotel_dashboard.ipynb
   - The master interactive dashboard notebook combining all 4 project components with dynamic global filters (Location, Star Category, Price, Rating, Data Type).
   - Pre-executed with live interactive Plotly charts and KPI cards saved, so you can explore the entire dashboard immediately without having to rerun heavy models.

2. requirements.txt
   - Specifies the exact Python library dependencies and versions (Plotly, ipywidgets, scikit-learn, mlxtend, pandas, etc.) needed to run the environment.
   - Install using: pip install -r dashboard/requirements.txt

3. DOCUMENTATION.md
   - Comprehensive technical documentation explaining the dashboard architecture, methodology (rating-based success proxy >= 4.0), KPI definitions, filter behavior, and strategic business takeaways.
   - Serves as the complete submission report and user manual for evaluators and teammates.

4. verification_report.md
   - Formal audit and verification log proving that 100% of teammate metrics match original notebook results with zero discrepancies.
   - Guarantees complete data integrity with zero placeholder or fabricated values.

5. generate_dashboard_notebook.py
   - Automated Python script that programmatically compiles the clean 5-cell master notebook from scratch.
   - Ensures full reproducibility and allows regenerating the notebook anytime if needed.

6. screenshots/
   - Directory containing high-resolution evidence screenshots of each dashboard tab (tab1_eda.png, tab2_models.png, tab3_text_mining.png, tab4_association_rules.png) plus before/after filter interaction proof.
   - Useful for presentations, final report slides, or viewing results on devices without Jupyter.

--------------------------------------------------------------------------------
DATA FOLDER (data/)
--------------------------------------------------------------------------------

1. data/combined_real_synthetic_hotels.csv
   - The primary hotel dataset containing 4,551 hotels (1,660 real and 2,851 synthetic properties) with pricing, categories, customer ratings, review counts, and 15 monitored amenity binary flags.
   - Used across Tab 1 (EDA), Tab 2 (Predictive Models), and Global Filter controls.

2. data/text_mining_outputs/
   - Directory housing all 36 text mining artifacts (including review_level_sentiment.csv, hotel_level_text_sentiment.csv, and presentation_numbers.json).
   - Powers the dynamic sentiment distributions, agreement matrices, and word clouds in Tab 3.

3. data/apriori/
   - Directory housing all 4 association rule mining artifacts (including apriori_strong_rules.csv, apriori_frequent_itemsets.csv, and apriori_amenity_support.csv).
   - Powers the interactive lift rankings, scatter plots, and dynamic mining engine in Tab 4.

--------------------------------------------------------------------------------
QUICK START INSTRUCTIONS
--------------------------------------------------------------------------------

Option A (Immediate View):
1. Open dashboard/hotel_dashboard.ipynb in VS Code or Jupyter Notebook.
2. The notebook is already executed—scroll down to Cell 4 to interact with the dashboard, switch tabs, and test filters.

Option B (Run From Scratch):
1. Open a terminal in the project folder and run:
   pip install -r dashboard/requirements.txt
2. Open dashboard/hotel_dashboard.ipynb and click "Run All" (or "Restart & Run All").
================================================================================
