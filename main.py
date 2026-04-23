"""
ModelForge Backend — FastAPI + Real sklearn Training
Senior ML Engineer Grade — Google Standards
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import io
import time
import traceback

# ── ML Libraries ─────────────────────────────────────────────────
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, KFold,
    GridSearchCV, train_test_split
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, r2_score, mean_squared_error,
    mean_absolute_error, classification_report, roc_auc_score
)

# Classifiers
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier,
    RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
)
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

app = FastAPI(title="ModelForge API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Constants ─────────────────────────────────────────────────────
MAX_ROWS       = 50_000   # cap for training speed
SAMPLE_ROWS    = 20_000   # sample if above this
CV_FOLDS       = 5
MAX_CATS       = 50       # max unique values for OHE
TIMEOUT_MODELS = ["SVC", "SVR", "KNeighborsClassifier"]  # skip if large data

# ── Helpers ───────────────────────────────────────────────────────
def infer_task(series: pd.Series) -> str:
    nums = pd.to_numeric(series, errors="coerce")
    if nums.notna().mean() > 0.9 and series.nunique() > 20:
        return "regression"
    unique = series.nunique()
    if unique == 2:
        return "binary_classification"
    return "multiclass_classification"

def build_preprocessor(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Cap high cardinality cats
    cat_cols = [c for c in cat_cols if X[c].nunique() <= MAX_CATS]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    transformers = []
    if num_cols:
        transformers.append(("num", numeric_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", categorical_pipe, cat_cols))

    preprocessor = ColumnTransformer(transformers, remainder="drop")
    return preprocessor, num_cols, cat_cols

def get_classifiers(n_rows: int) -> Dict:
    models = {}
    if HAS_LGBM:
        models["LGBMClassifier"] = LGBMClassifier(
            n_estimators=100, random_state=42, n_jobs=-1, verbose=-1
        )
    if HAS_XGB:
        models["XGBClassifier"] = XGBClassifier(
            n_estimators=100, random_state=42, n_jobs=-1,
            eval_metric="logloss", verbosity=0
        )
    models["RandomForestClassifier"]     = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    models["GradientBoostingClassifier"] = GradientBoostingClassifier(n_estimators=100, random_state=42)
    models["ExtraTreesClassifier"]       = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    models["LogisticRegression"]         = LogisticRegression(max_iter=500, random_state=42, n_jobs=-1)
    models["DecisionTreeClassifier"]     = DecisionTreeClassifier(random_state=42)
    models["GaussianNB"]                 = GaussianNB()
    if n_rows < 5000:
        models["SVC"]                    = SVC(probability=True, random_state=42)
        models["KNeighborsClassifier"]   = KNeighborsClassifier(n_jobs=-1)
    return models

def get_regressors(n_rows: int) -> Dict:
    models = {}
    if HAS_LGBM:
        models["LGBMRegressor"]  = LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
    if HAS_XGB:
        models["XGBRegressor"]   = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0)
    models["RandomForestRegressor"]     = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    models["GradientBoostingRegressor"] = GradientBoostingRegressor(n_estimators=100, random_state=42)
    models["ExtraTreesRegressor"]       = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    models["Ridge"]                     = Ridge()
    models["Lasso"]                     = Lasso(max_iter=2000)
    models["ElasticNet"]                = ElasticNet(max_iter=2000)
    models["DecisionTreeRegressor"]     = DecisionTreeRegressor(random_state=42)
    if n_rows < 5000:
        models["SVR"]                   = SVR()
    return models

HYPERPARAM_GRIDS = {
    "LGBMClassifier":             {"model__num_leaves": [31, 63], "model__learning_rate": [0.05, 0.1]},
    "XGBClassifier":              {"model__max_depth": [4, 6], "model__learning_rate": [0.05, 0.1]},
    "RandomForestClassifier":     {"model__n_estimators": [100, 200], "model__max_depth": [None, 10]},
    "LogisticRegression":         {"model__C": [0.1, 1.0, 10.0]},
    "LGBMRegressor":              {"model__num_leaves": [31, 63], "model__learning_rate": [0.05, 0.1]},
    "XGBRegressor":               {"model__max_depth": [4, 6], "model__learning_rate": [0.05, 0.1]},
    "RandomForestRegressor":      {"model__n_estimators": [100, 200], "model__max_depth": [None, 10]},
    "Ridge":                      {"model__alpha": [0.1, 1.0, 10.0]},
}

# ── Routes ────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ModelForge API running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "lgbm": HAS_LGBM,
        "xgb": HAS_XGB,
    }

@app.post("/analyze")
async def analyze(file: UploadFile = File(...), target: str = ""):
    """
    Full ML Pipeline:
    1. Parse CSV
    2. Detect task type
    3. Build preprocessor
    4. CV benchmark all models
    5. Tune top 2 with GridSearchCV
    6. Return real results
    """
    t0 = time.time()

    # ── 1. Read CSV ───────────────────────────────────────────────
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content), low_memory=False)
    except Exception as e:
        raise HTTPException(400, f"CSV parse error: {e}")

    total_rows = len(df)

    # Sample if too large
    if len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=42)
    elif len(df) > SAMPLE_ROWS:
        df = df.sample(n=SAMPLE_ROWS, random_state=42)

    n_rows = len(df)

    # ── 2. Target Column ──────────────────────────────────────────
    if not target or target not in df.columns:
        target = df.columns[-1]

    X = df.drop(columns=[target])
    y = df[target].copy()

    # Drop high-cardinality object cols from X
    for col in X.select_dtypes(include="object").columns:
        if X[col].nunique() > MAX_CATS:
            X = X.drop(columns=[col])

    # ── 3. Task Detection ─────────────────────────────────────────
    task_type = infer_task(y)
    is_reg    = task_type == "regression"

    le = None
    if not is_reg:
        le = LabelEncoder()
        y  = le.fit_transform(y.astype(str))
        classes = le.classes_.tolist()
    else:
        y = pd.to_numeric(y, errors="coerce").fillna(y.median() if pd.to_numeric(y, errors="coerce").notna().any() else 0)
        classes = []

    # ── 4. Preprocessor ───────────────────────────────────────────
    preprocessor, num_cols, cat_cols = build_preprocessor(X)

    # ── 5. CV Benchmark ───────────────────────────────────────────
    models     = get_regressors(n_rows) if is_reg else get_classifiers(n_rows)
    scoring    = "r2" if is_reg else ("roc_auc" if task_type == "binary_classification" else "accuracy")
    cv_splitter = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42) if is_reg \
                  else StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

    cv_results = []
    for name, model in models.items():
        try:
            pipe   = Pipeline([("pre", preprocessor), ("model", model)])
            scores = cross_val_score(pipe, X, y, cv=cv_splitter, scoring=scoring, n_jobs=-1)
            cv_results.append({
                "name":     name,
                "cv_mean":  round(float(scores.mean()), 4),
                "cv_std":   round(float(scores.std()),  4),
                "cv_scores": [round(float(s), 4) for s in scores],
                "scoring":  scoring,
            })
        except Exception as ex:
            cv_results.append({
                "name":    name,
                "cv_mean": 0.0,
                "cv_std":  0.0,
                "cv_scores": [],
                "error":   str(ex),
                "scoring": scoring,
            })

    cv_results.sort(key=lambda x: x["cv_mean"], reverse=True)
    top2 = [r["name"] for r in cv_results[:2] if r["cv_mean"] > 0]

    # ── 6. Tune Top 2 ─────────────────────────────────────────────
    tuned_results = []
    best_pipeline = None

    for name in top2:
        model = models.get(name)
        if not model:
            continue
        grid = HYPERPARAM_GRIDS.get(name, {})
        pipe = Pipeline([("pre", preprocessor), ("model", model)])

        try:
            if grid:
                gs = GridSearchCV(
                    pipe, grid, cv=cv_splitter,
                    scoring=scoring, n_jobs=-1, refit=True
                )
                gs.fit(X, y)
                tuned_score  = round(float(gs.best_score_), 4)
                best_params  = {k.replace("model__", ""): v for k, v in gs.best_params_.items()}
                fitted_pipe  = gs.best_estimator_
            else:
                pipe.fit(X, y)
                scores      = cross_val_score(pipe, X, y, cv=cv_splitter, scoring=scoring, n_jobs=-1)
                tuned_score = round(float(scores.mean()), 4)
                best_params = {}
                fitted_pipe = pipe

            base_score = next((r["cv_mean"] for r in cv_results if r["name"] == name), 0)

            tuned_results.append({
                "name":        name,
                "base_score":  base_score,
                "tuned_score": tuned_score,
                "improvement": round(tuned_score - base_score, 4),
                "best_params": best_params,
            })

            if best_pipeline is None:
                best_pipeline = fitted_pipe

        except Exception as ex:
            tuned_results.append({"name": name, "error": str(ex)})

    # ── 7. Feature Importance ─────────────────────────────────────
    feature_importance = []
    if best_pipeline:
        try:
            model_step = best_pipeline.named_steps["model"]
            if hasattr(model_step, "feature_importances_"):
                importances = model_step.feature_importances_
                pre_step    = best_pipeline.named_steps["pre"]
                feat_names  = []
                for name_t, _, cols in pre_step.transformers_:
                    feat_names.extend(cols)
                feat_names = feat_names[:len(importances)]
                total      = importances.sum()
                feature_importance = sorted([
                    {
                        "feature":    str(fn),
                        "importance": round(float(imp / total), 4),
                        "type": "numeric" if fn in num_cols else "categorical"
                    }
                    for fn, imp in zip(feat_names, importances)
                ], key=lambda x: x["importance"], reverse=True)[:15]
        except Exception:
            pass

    # ── 8. Test Set Evaluation ────────────────────────────────────
    test_metrics = {}
    if best_pipeline and tuned_results:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42,
                stratify=y if not is_reg else None
            )
            best_pipeline.fit(X_train, y_train)
            y_pred = best_pipeline.predict(X_test)

            if is_reg:
                test_metrics = {
                    "r2":   round(float(r2_score(y_test, y_pred)), 4),
                    "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
                    "mae":  round(float(mean_absolute_error(y_test, y_pred)), 4),
                }
            else:
                test_metrics = {
                    "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
                }
                if task_type == "binary_classification" and hasattr(best_pipeline, "predict_proba"):
                    proba = best_pipeline.predict_proba(X_test)[:, 1]
                    test_metrics["roc_auc"] = round(float(roc_auc_score(y_test, proba)), 4)
        except Exception as ex:
            test_metrics = {"error": str(ex)}

    elapsed = round(time.time() - t0, 2)

    return {
        "status":          "success",
        "total_rows":      total_rows,
        "trained_rows":    n_rows,
        "n_features":      len(X.columns),
        "target":          target,
        "task_type":       task_type,
        "classes":         classes,
        "scoring_metric":  scoring,
        "cv_results":      cv_results,
        "tuned_results":   tuned_results,
        "feature_importance": feature_importance,
        "test_metrics":    test_metrics,
        "best_model":      tuned_results[0]["name"] if tuned_results else None,
        "elapsed_seconds": elapsed,
        "num_cols":        num_cols,
        "cat_cols":        cat_cols,
    }


@app.post("/predict")
async def predict_single(file: UploadFile = File(...), target: str = "", row_json: str = "{}"):
    """Train on full data, predict single row"""
    import json
    content = await file.read()
    df      = pd.read_csv(io.BytesIO(content), low_memory=False)
    if len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=42)

    if not target or target not in df.columns:
        target = df.columns[-1]

    X = df.drop(columns=[target])
    y = df[target].copy()

    task_type = infer_task(y)
    is_reg    = task_type == "regression"

    le = None
    if not is_reg:
        le = LabelEncoder()
        y  = le.fit_transform(y.astype(str))

    preprocessor, num_cols, cat_cols = build_preprocessor(X)

    # Use best model — LGBM if available else RF
    if HAS_LGBM:
        model = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1) if is_reg \
                else LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42) if is_reg \
                else RandomForestClassifier(n_estimators=100, random_state=42)

    pipe = Pipeline([("pre", preprocessor), ("model", model)])
    pipe.fit(X, y)

    row_data  = json.loads(row_json)
    row_df    = pd.DataFrame([row_data])

    # Fill missing cols with mode/mean
    for col in X.columns:
        if col not in row_df.columns:
            if col in num_cols:
                row_df[col] = float(X[col].median())
            else:
                row_df[col] = X[col].mode()[0] if not X[col].mode().empty else "unknown"

    row_df = row_df[X.columns]
    pred   = pipe.predict(row_df)[0]

    result = {}
    if is_reg:
        result = {"prediction": round(float(pred), 4), "type": "regression"}
    else:
        label = le.inverse_transform([int(pred)])[0]
        proba = {}
        if hasattr(pipe, "predict_proba"):
            probas = pipe.predict_proba(row_df)[0]
            proba  = {le.inverse_transform([i])[0]: round(float(p), 4)
                      for i, p in enumerate(probas)}
        result = {"prediction": str(label), "probabilities": proba, "type": task_type}

    return result
