# DATA 200 — Applied Statistical Analysis
## Project Assignment Brief

---

## Objective

Students will apply statistical and predictive modeling techniques to analyze a real-world dataset relevant to their areas of interest (e.g., Healthcare, Marketing, Sports, Finance, Education, etc.). The project will focus on:

- Cleaning and structuring data
- Applying statistical analysis techniques such as Linear Regression, ANOVA, or Logistic Regression
- Developing a simple end-to-end application
- Generating a comprehensive report to document choices, insights, and conclusions

---

## Week 2 — Literature Review and Dataset Selection

**Tasks:**
- Conduct a literature review to establish context and relevance
- Select a suitable dataset

**Deliverables:**
- Submit at least 3 literature reviews of related works

**Our deliverable:**
- Selected dataset: `EPL_combined.csv` — 760 EPL matches from 2023/24 and 2024/25 seasons
- Problem: Predict match outcome (Home Win / Draw / Away Win) from in-match statistics

---

## Week 3 — Exploratory Data Analysis (EDA)

**Tasks:**
- Perform EDA using descriptive statistics and visualizations to understand data patterns and relationships
- Preprocess the data: Handle missing values, outliers, and duplicates, and perform other data wrangling techniques
- Check relationships in data

**Deliverables:**
- EDA summary with key insights supported by visualizations (e.g., scatter plots, histograms, box plots)

**Our deliverable:**
- `Week3_EDA.ipynb`
- Charts: `week3_ftr_distribution.png`, `week3_goals_distribution.png`, `week3_boxplots.png`, `week3_correlation_heatmap.png`, `week3_scatter_multicollinearity.png`, `week3_outliers.png`, `week3_htr_ftr_heatmap.png`

**Key findings:**
- 0 missing values, 0 duplicates — no preprocessing required
- Natural class imbalance: H=43.4%, D=23.0%, A=33.6%
- Strong multicollinearity: r(HS, HST) > 0.85 and r(AS, AST) > 0.85
- Half-time result is the strongest predictor of full-time result

---

## Week 4 — Statistical Model Selection and Hypothesis Development

**Tasks:**
- Select appropriate statistical techniques (e.g., Regression, ANOVA, Logistic Regression) based on the problem statement
- Perform feature selection and develop hypotheses

**Deliverables:**
- A short slide deck with justification of model and feature choices

**Our deliverable:**
- `Week4_Model_Selection.ipynb`
- Charts: `week4_ftr_distribution.png`, `week4_multicollinearity.png`, `week4_hypotheses.png`

**Key decisions:**
- Model: Multinomial Logistic Regression (3-class target; interpretable coefficients)
- Features kept (10): HST, AST, HC, AC, HY, AY, HR, AR, HTHG, HTAG
- Features dropped (2): HS, AS — multicollinearity confirmed via Pearson r and statsmodels p-values
- Hypotheses developed:
  - H1: HST differs significantly across FTR classes → test with One-Way ANOVA
  - H2: HTHG/HTAG is a significant predictor of FTR → test with logistic regression coefficient
  - H3: Red cards shift match outcome probabilities → test with coefficient + proportion analysis

---

## Week 5 — Statistical Analysis and Validation

**Tasks:**
- Conduct descriptive and inferential statistical analysis tests using Python
- Perform diagnostics measures
- Interpret results and validate hypotheses
- If necessary, choose sampling methods or collect additional data

**Our deliverable:**
- `Week5_Statistical_Analysis.ipynb`
- Charts: `week5_anova.png`, `week5_coefficients.png`, `week5_confusion_matrix.png`, `week5_roc_curve.png`
- Saved model: `epl_model.pkl`, `epl_scaler.pkl`

**Key results:**
- ANOVA: HST, AST, HC, AC all significant (p < 0.001) — H1 SUPPORTED
- VIF: all 10 features < 5 — no multicollinearity in final feature set
- Model accuracy: 62.5% (upper end of 55–65% literature range for 3-class football prediction)
- H1: SUPPORTED | H2: SUPPORTED | H3: PARTIALLY SUPPORTED
- Sampling: no additional data collection required; natural class imbalance retained by design

---

## Week 6 — Statistical Modeling (Continued)

**Tasks:**
- Build on the analysis from Week 5
- Finalize insights and begin compiling the project report

**Deliverables:**
- Present statistical analysis results and insights
- Present the report draft for progress tracking

**Our deliverable:**
- `Week6_Modelling_Final.ipynb`
- Charts: `week6_confusion_matrix_final.png`, `week6_all_coefficients.png`, `week6_results_dashboard.png`
- `Week6_Report_Draft.md` — all 8 sections complete

**Key results:**
- Final accuracy: 62.5%
- Top feature: HTHG (highest Home Win coefficient)
- Draw F1 = 0.22 — expected limitation, not a model error
- 6 key insights documented with computed statistics

---

## Week 7 — Application Development

**Tasks:**
- Simple Python Application Creation for project demonstration using libraries of student's choice

**Deliverables:**
- Locally running Python application

**Our deliverable:**
- `app.py` — Streamlit web application
- Run with: `streamlit run app.py` (requires `EPL_combined.csv` in same folder)

**App features:**
- 10 input fields for in-match statistics (defaults set to dataset means)
- Predicts match outcome (H/D/A) with probability bar chart
- Shows model test accuracy (62.5%)
- Input validation warnings for unusual stat combinations
- Draw confidence note (F1 ≈ 0.22)

---

## Week 8 — Peer Evaluation and Final Presentation

**Tasks:**
- Conduct peer evaluations (review at least two projects)
- Prepare and present findings in a 10-minute presentation
- Submit the final report with documentation

**Deliverables:**
- Final report with detailed documentation
- Presentation slides
- Peer evaluation feedback

**Our deliverable (to complete):**
- Final polished report (based on `Week6_Report_Draft.md`)
- Presentation slides covering Weeks 2–7
- Peer evaluation of two classmates' projects

---

## Evaluation Criteria

| Criterion | Weight |
|---|---|
| Dataset and Problem Definition | 10% |
| Exploratory Data Analysis and Preprocessing | 20% |
| Statistical Modeling and Validation | 40% |
| Python Application Development | 10% |
| Presentation and Collaboration | 20% |

---

## Project Files Summary

| File | Week | Description |
|---|---|---|
| `EPL_combined.csv` | 2 | Dataset — 760 EPL matches |
| `Week3_EDA.ipynb` | 3 | Exploratory Data Analysis |
| `Week4_Model_Selection.ipynb` | 4 | Model selection + hypothesis development |
| `Week5_Statistical_Analysis.ipynb` | 5 | ANOVA, VIF, model training, validation |
| `Week6_Modelling_Final.ipynb` | 6 | Final model + results dashboard |
| `Week6_Report_Draft.md` | 6 | Report draft — all 8 sections |
| `app.py` | 7 | Streamlit prediction app |
| `requirements_app.txt` | 7 | App dependencies |
