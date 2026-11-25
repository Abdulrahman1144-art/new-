"""Data preparation and outcome-status modeling for the UAE cancer dataset.

This is an educational benchmark over recorded status, not a clinical predictor.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path(os.environ.get("CANCER_DATA_PATH", ROOT / "data/raw/uae_cancer_dataset_200k.xlsx"))
MODEL_PATH = ROOT / "models/outcome_status_model.joblib"
METRICS_PATH = ROOT / "reports/model_metrics.json"
TARGET = "Outcome"
NUMERIC_FEATURES = ["Age", "BMI"]
CATEGORICAL_FEATURES = ["Gender", "Nationality", "Emirate", "Cancer_Type",
                        "Cancer_Stage", "Smoking_Status", "Comorbidities", "Ethnicity"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Set CANCER_DATA_PATH.")
    return pd.read_excel(path)


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Normalize formats and remove invalid/duplicate identifiers; keep meaningful nulls."""
    df = raw.copy()
    before = {"rows": len(df), "columns": len(df.columns),
              "exact_duplicates": int(df.duplicated().sum()),
              "duplicate_patient_ids": int(df.Patient_ID.duplicated().sum()),
              "missing_cells": int(df.isna().sum().sum())}
    df.columns = [str(c).strip() for c in df.columns]
    text_cols = df.select_dtypes(include="object").columns
    for col in text_cols:
        df[col] = df[col].astype("string").str.strip().replace("", pd.NA)
    for col in ("Diagnosis_Date", "Treatment_Start_Date", "Death_Date"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ("Age", "Weight", "Height"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset="Patient_ID", keep="first")
    # Values outside defensible human ranges are treated as invalid, not capped.
    invalid = ((df.Age.notna() & ~df.Age.between(0, 120)) |
               (df.Weight.notna() & ~df.Weight.between(20, 300)) |
               (df.Height.notna() & ~df.Height.between(100, 230)))
    invalid_count = int(invalid.sum())
    df = df.loc[~invalid].copy()
    df["BMI"] = df["Weight"] / (df["Height"] / 100) ** 2
    # A missing comorbidity is unknown, not evidence of no comorbidity.
    df["Comorbidities"] = df["Comorbidities"].fillna("Unknown")
    after = {"rows": len(df), "columns": len(df.columns),
             "exact_duplicates": int(df.duplicated().sum()),
             "duplicate_patient_ids": int(df.Patient_ID.duplicated().sum()),
             "missing_cells": int(df.isna().sum().sum()),
             "invalid_range_rows_removed": invalid_count,
             "comorbidity_unknown_count": int(df.Comorbidities.eq("Unknown").sum())}
    return df, {"before": before, "after": after}


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("numeric", numeric, NUMERIC_FEATURES),
                              ("categorical", categorical, CATEGORICAL_FEATURES)])


def train_model(df: pd.DataFrame, tune: bool = True) -> tuple[Pipeline, dict]:
    """Compare two models, tune logistic regression using CV, then evaluate a held-out set."""
    data = df.loc[df[TARGET].notna()].copy()
    X, y = data[FEATURES], data[TARGET].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    base = Pipeline([("prep", make_preprocessor()),
                     ("model", LogisticRegression(max_iter=1200, solver="lbfgs"))])
    tree = Pipeline([("prep", make_preprocessor()),
                     ("model", DecisionTreeClassifier(random_state=42, min_samples_leaf=20))])
    comparisons = {}
    for name, estimator in (("Logistic regression", base), ("Decision tree", tree)):
        estimator.fit(X_train, y_train)
        pred = estimator.predict(X_test)
        comparisons[name] = {"macro_f1": float(f1_score(y_test, pred, average="macro")),
                             "accuracy": float(accuracy_score(y_test, pred))}
    best = base
    search_summary = "No hyperparameter search requested."
    if tune:
        search = RandomizedSearchCV(
            base, {"model__C": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                   "model__class_weight": [None, "balanced"]},
            n_iter=6, scoring="f1_macro", cv=3, random_state=42,
            n_jobs=-1, refit=True, verbose=0)
        search.fit(X_train, y_train)
        best = search.best_estimator_
        search_summary = {"best_params": search.best_params_,
                          "best_cv_macro_f1": float(search.best_score_),
                          "trials": int(len(search.cv_results_["params"])),
                          "folds": 3}
    pred = best.predict(X_test)
    metrics = {
        "target": TARGET,
        "target_definition": "recorded status in this dataset; not a future clinical outcome",
        "feature_columns": FEATURES,
        "excluded_columns": ["Patient_ID", "Diagnosis_Date", "Treatment_Type",
                             "Treatment_Start_Date", "Death_Date", "Cause_of_Death",
                             "Primary_Physician", "Hospital", "Weight", "Height"],
        "rows": int(len(data)), "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "comparison": comparisons, "tuning": search_summary,
        "final_model": "Logistic regression",
        "test_metrics": {"accuracy": float(accuracy_score(y_test, pred)),
                         "macro_f1": float(f1_score(y_test, pred, average="macro")),
                         "macro_precision": float(precision_score(y_test, pred, average="macro")),
                         "macro_recall": float(recall_score(y_test, pred, average="macro")),
                         "classification_report": classification_report(y_test, pred, output_dict=True),
                         "confusion_matrix": confusion_matrix(y_test, pred, labels=best.classes_).tolist(),
                         "class_order": best.classes_.tolist()}}
    return best, metrics


def run_training(tune: bool = True) -> dict:
    raw = load_raw()
    clean, quality = clean_data(raw)
    model, metrics = train_model(clean, tune=tune)
    processed_path = ROOT / "data/processed/cleaned_dataset.csv"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(processed_path, index=False)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    metrics["quality_checks"] = quality
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    print(json.dumps(run_training(), indent=2))
