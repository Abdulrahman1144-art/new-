# UAE Cancer Data Explorer

## Project summary and model report

**Prepared:** 25 November 2026  
**Purpose:** Educational data-science project

> We cleaned and explored a 200,000-row workbook, prepared diagnosis-time features, compared classifiers, tuned logistic regression, and built a Streamlit dashboard. The model predicts the workbook's recorded `Outcome` label only; it is not a clinical or survival model.

## 1. Project overview

The project asks whether selected attributes recorded around diagnosis can classify the dataset's recorded `Outcome` category (`Recovered`, `Under Treatment`, or `Deceased`). The workflow includes data cleaning, feature preparation, model comparison and tuning, held-out evaluation, and an interactive Streamlit dashboard.

The outcome observation time is not defined in the workbook, so the task is classification of an existing label rather than prediction of a future clinical event.

| Project item | Summary |
| --- | --- |
| Input | Workbook described as 200,000 records and 20 source columns; real-world source and license are unverified. |
| Target | `Outcome`: the recorded label in the supplied dataset. |
| Prepared data | 21 columns after deriving BMI; 200,000 rows remained after cleaning. |
| Application | Streamlit dashboard for filtering and exploring the dataset, plus an exploratory model interface. |

## 2. Data cleaning and preparation

- Trimmed whitespace from column names and text values; converted empty text values to missing values.
- Parsed date columns and coerced age, weight, and height to numeric values. Unparseable entries become missing rather than being treated as valid.
- Checked for exact duplicate rows and repeated `Patient_ID` values. The checks found zero of each, so no records were removed for duplication.
- Checked age, weight, and height against broad plausibility bounds (age 0–120, weight 20–300, height 100–230). No rows failed, so none were removed by these checks.
- Derived BMI as weight divided by height in metres squared. Weight and height were then excluded from model predictors.
- Changed missing `Comorbidities` values to `Unknown`, because missing does not mean that a person had no comorbidity. Left `Death_Date` and `Cause_of_Death` missing because their absence is meaningful for many non-deceased records.
- Used a preprocessing pipeline: median imputation and standard scaling for numeric features; most-frequent imputation and one-hot encoding for categorical features. Unknown categories at prediction time are ignored safely.
- Used diagnosis-time features only. `Patient_ID`, diagnosis date, treatment, death details, hospital, and physician were excluded to reduce identifier use and leakage from later-recorded information.

## 3. Data quality findings

| Issue | Finding and handling | Status |
| --- | --- | --- |
| Missing data | 80,660 missing `Comorbidities` values were represented as `Unknown`. `Death_Date` and `Cause_of_Death` each had 171,300 missing values and were retained as missing. Total missing cells changed from 423,260 to 342,600. | Partly handled; death fields remain missing by design. |
| Duplicates | No exact duplicate rows and no duplicate patient IDs were found. | Checked; no correction needed. |
| Implausible measurements | No values fell outside configured age, weight, or height bounds. | Checked; zero rows removed. |
| Definitions and units | No authoritative data dictionary was supplied. Column meanings are inferred; weight and height units are assumed, not verified. | Unresolved. |
| Provenance and license | The workbook's source, generation method, collection protocol, and sharing license are undocumented. | Unresolved; limits public claims and release. |
| Outcome meaning | `Outcome` is recorded at an unspecified observation time and does not establish survival or treatment effectiveness. | Unresolved; target is label classification only. |

The project documentation reports no demonstrated category spelling inconsistencies; observed dates parsed successfully, and treatment start was not before diagnosis for observed records. These checks do not establish that the dataset represents real-world patients.

## 4. Model selection and evaluation

A stratified 80/20 split produced 160,000 training records and 40,000 held-out test records. Logistic regression and a decision tree were compared. Logistic regression was then tuned with randomized search over six settings using three-fold cross-validation. The selected settings were `class_weight="balanced"` and `C=3.0`. Macro F1 was the tuning objective to give each outcome class more equal influence.

| Model | Test accuracy | Test macro F1 | Interpretation |
| --- | ---: | ---: | --- |
| Baseline logistic regression | 58.7% | 0.367 | Higher accuracy, weaker class-balanced F1. |
| Decision tree | 55.1% | 0.418 | Better macro F1 than untuned logistic regression. |
| Tuned logistic regression (selected) | 53.2% | 0.452 | Highest macro F1 among these results; selected for more balanced class performance. |

Per-class test results for the selected model:

| Recorded class | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Deceased | 0.279 | 0.565 | 0.374 |
| Recovered | 0.709 | 0.698 | 0.703 |
| Under Treatment | 0.397 | 0.217 | 0.280 |

Overall held-out accuracy was 0.532 and macro F1 was 0.452. The model is notably weak at identifying the `Under Treatment` class (recall 0.217). It should be presented as an exploratory benchmark, not as a dependable patient-level prediction.

## 5. Is the project sufficient?

| Intended use | Assessment |
| --- | --- |
| Class or portfolio demonstration | Yes, as an educational end-to-end prototype: it shows cleaning, preprocessing, model comparison, tuning, evaluation, and a dashboard. |
| Real-world claims about UAE cancer outcomes | No. Dataset provenance, collection method, and label definition are unverified. |
| Clinical decision-making or patient risk prediction | No. The target is not a defined future outcome; performance is modest; external validation, calibration, fairness assessment, and clinical review are absent. |
| Public deployment | Not ready. Data sharing rights are unknown, generated model/data artifacts are not included, and GitHub Pages cannot execute the Streamlit Python app. |

Before making stronger claims, verify the dataset source and license, obtain an authoritative data dictionary and measurement units, define exactly when `Outcome` is observed, assess missingness and subgroup performance, and validate on an independent dataset from the intended population. Prediction-model reporting guidance treats model development and validation as distinct tasks; this project currently has only an internal random hold-out evaluation. See [TRIPOD+AI (BMJ, 2024)](https://www.bmj.com/content/385/bmj-2023-078378.full).

## 6. One-sentence conclusion

This is a useful educational cancer-data exploration and classification prototype, but its unverified data source, undefined outcome timing, modest and uneven model performance, and lack of external validation make it insufficient for clinical use or real-world claims.
