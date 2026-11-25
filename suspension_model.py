"""
Project Handa — Random Forest Classification for class suspension prediction.

Per the proposal (Capsule Proposal-3.pdf): benchmark Random Forest
Classification on historical weather + suspension data for Olongapo City,
evaluate with accuracy/precision/recall/F1, and inspect feature importances.

Data:
  - "Cubi Point Daily Data.csv": daily weather 2022-01-01 to 2024-12-31.
  - "May-November Suspensions - Sheet1.csv": FB-scraped suspension
    announcements. IMPORTANT — this file only lists days suspensions
    WERE declared (all positive cases). There is no explicit "no
    suspension" record.

Labeling strategy (see build_labeled_dataset):
  - Every date in the suspension sheet is a positive (SUSPENDED=1) example.
  - Every OTHER date in the weather data is treated as a negative
    (SUSPENDED=0) example. By user instruction, this assumes any date
    NOT in the scraped suspension sheet had no weather-related
    suspension — so the full 2023-01-01 to 2024-12-31 weather window
    is used (not just from the first scraped record, 2023-06-13).
    2022 is still excluded (per user request to use 2023-2024 only).
  - 2025 suspension dates (Jul-Nov 2025) are excluded — there's no
    matching weather data for them yet. Add 2025 Cubi Point data to
    unlock that period.

This is a small, heavily imbalanced dataset (24 positive days out of
731 in the usable window), so treat metrics as a rough benchmark, not
a production-grade result.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

# macOS's Accelerate BLAS backend raises spurious "invalid value encountered in
# matmul" RuntimeWarnings on some logistic-regression matrix products even when
# the actual result is fully finite (verified: no NaN/Inf in coef_ or output).
# Silence just that known false positive rather than real numerical issues.
warnings.filterwarnings("ignore", message=".*encountered in matmul.*", category=RuntimeWarning)

_DATA_DIR = Path(__file__).resolve().parent
WEATHER_CSV = _DATA_DIR / "Cubi Point Daily Data.csv"
SUSPENSIONS_CSV = _DATA_DIR / "May-November Suspensions - Sheet1.csv"

FEATURE_COLS = [
    "RAINFALL",
    "RAIN_3DAY_SUM",
    "RAIN_7DAY_SUM",
    "TMAX",
    "TMIN",
    "RH",
    "WIND_SPEED",
    "MONTH",
]

SCHOOL_COLS = ["Pre-School", "Elementary", "High School", "College"]

SUGGESTION_BANDS = [
    (0.60, "Suspension Likely — recommend monitoring PAGASA advisories closely"),
    (0.30, "Elevated Risk — possible partial/afternoon suspension, keep watching"),
    (0.00, "No Suspension Likely — normal conditions"),
]


def load_weather(path=WEATHER_CSV):
    df = pd.read_csv(path)
    df["DATE"] = pd.to_datetime(dict(year=df.YEAR, month=df.MONTH, day=df.DAY))
    df["RAINFALL"] = df["RAINFALL"].clip(lower=0)  # -1 marks trace rainfall, not missing
    return df.sort_values("DATE").reset_index(drop=True)


def load_suspensions(path=SUSPENSIONS_CSV):
    df = pd.read_csv(path)
    for col in ["Rainfall Level", "Forecast Risk", "Time Risk"]:
        df[col] = df[col].astype(str).str.strip().replace({"nan": np.nan})
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=False)

    # A handful of scraped rows have a blank Suspension Decision but a
    # school level marked suspended — backfill those as suspended.
    any_level_suspended = (df[SCHOOL_COLS].fillna(0).sum(axis=1) > 0).astype(int)
    df["Suspension Decision"] = df["Suspension Decision"].fillna(any_level_suspended)
    df["Suspension Decision"] = df["Suspension Decision"].astype(int)

    # Same calendar date scraped twice from separate FB posts (e.g. 7/24/2024) -> dedup.
    df = df.sort_values("Date").drop_duplicates(subset="Date", keep="last")
    return df.reset_index(drop=True)


def build_labeled_dataset(weather_df, suspensions_df, start=None, end=None):
    """Label every weather day in [start, end] as suspended or not.

    By default, restricts to [earliest scraped suspension date, latest weather
    date] since dates outside that window can't be trusted as true negatives.
    Pass explicit `start`/`end` to override that — e.g. to assume every
    unlisted date truly had no weather-related suspension, per the user's
    stated assumption, and use the full 2023-2024 weather range instead of
    starting from the first scraped record (2023-06-13)."""
    coverage_start = pd.Timestamp(start) if start is not None else suspensions_df["Date"].min()
    coverage_end = pd.Timestamp(end) if end is not None else weather_df["DATE"].max()

    window = weather_df[
        (weather_df.DATE >= coverage_start) & (weather_df.DATE <= coverage_end)
    ].copy()

    suspended_dates = set(
        suspensions_df.loc[suspensions_df["Suspension Decision"] == 1, "Date"]
    )
    window["SUSPENDED"] = window["DATE"].isin(suspended_dates).astype(int)

    dropped_out_of_window = suspensions_df[
        (suspensions_df["Date"] < coverage_start) | (suspensions_df["Date"] > coverage_end)
    ]
    return window.reset_index(drop=True), coverage_start, coverage_end, dropped_out_of_window


def build_features(df):
    feats = df.copy()
    feats["RAIN_3DAY_SUM"] = feats.RAINFALL.rolling(3, min_periods=1).sum()
    feats["RAIN_7DAY_SUM"] = feats.RAINFALL.rolling(7, min_periods=1).sum()
    feats["MONTH"] = feats.DATE.dt.month
    return feats


def train_random_forest(features_df, feature_cols=FEATURE_COLS, label_col="SUSPENDED",
                         test_size=0.25, random_state=42, n_estimators=300):
    X = features_df[feature_cols]
    y = features_df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred, zero_division=0),
    }
    return model, metrics, (X_test, y_test, y_pred)


def feature_importance(model, feature_cols=FEATURE_COLS):
    return pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)


# Gradient Boosting has no class_weight param, unlike RF/LogReg — balance it
# manually via sample_weight so all three models get the same rare-class boost.
MODEL_FACTORIES = {
    # With only ~5 positives per fold, RAINFALL alone can nearly perfectly
    # separate the classes. lbfgs (sklearn's default solver) overflows trying
    # to fit that boundary even with C=0.1; liblinear's coordinate descent
    # stays numerically stable under the same regularization.
    "Logistic Regression": lambda random_state: Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=0.1, class_weight="balanced",
                                    solver="liblinear", random_state=random_state)),
    ]),
    "Random Forest": lambda random_state: RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=random_state
    ),
    "Gradient Boosting": lambda random_state: GradientBoostingClassifier(random_state=random_state),
}

# Same models, but depth/leaf-constrained so a tree can't carve out a leaf per
# training positive (the direct cause of the 100% train scores seen earlier).
# With only ~18 positives per training fold, every split has to keep at least
# 4 samples together instead of isolating individual points.
REGULARIZED_MODEL_FACTORIES = {
    "Random Forest (regularized)": lambda random_state: RandomForestClassifier(
        n_estimators=300, max_depth=4, min_samples_leaf=4,
        class_weight="balanced", random_state=random_state,
    ),
    "Gradient Boosting (regularized)": lambda random_state: GradientBoostingClassifier(
        n_estimators=100, max_depth=2, min_samples_leaf=4,
        learning_rate=0.05, random_state=random_state,
    ),
}


def _fit(name, model, X_train, y_train):
    if name.startswith("Gradient Boosting"):
        model.fit(X_train, y_train, sample_weight=compute_sample_weight("balanced", y_train))
    else:
        model.fit(X_train, y_train)
    return model


def _score(model, X, y):
    pred = model.predict(X)
    return dict(
        accuracy=accuracy_score(y, pred),
        precision=precision_score(y, pred, zero_division=0),
        recall=recall_score(y, pred, zero_division=0),
        f1=f1_score(y, pred, zero_division=0),
    )


def train_vs_test_diagnostic(features_df, model_factories=MODEL_FACTORIES, feature_cols=FEATURE_COLS,
                              label_col="SUSPENDED", test_size=0.25, random_state=42):
    """Overfitting check: a model that scores near-perfect on train but drops sharply on
    test has memorized the (few) training positives rather than learned a general pattern."""
    X = features_df[feature_cols]
    y = features_df[label_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    rows = []
    for name, factory in model_factories.items():
        model = _fit(name, factory(random_state), X_train, y_train)
        train_scores = _score(model, X_train, y_train)
        test_scores = _score(model, X_test, y_test)
        rows.append({"model": name, **{f"train_{k}": v for k, v in train_scores.items()},
                     **{f"test_{k}": v for k, v in test_scores.items()}})
    return pd.DataFrame(rows)


def cross_validate_models(features_df, model_factories=MODEL_FACTORIES, feature_cols=FEATURE_COLS,
                           label_col="SUSPENDED", n_splits=5, random_state=42):
    """Stratified k-fold comparison. A single 142-row/6-positive test split is too small to
    trust one number from — averaging over folds shows how much precision/recall swing."""
    X = features_df[feature_cols].to_numpy()
    y = features_df[label_col].to_numpy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    rows = []
    for name, factory in model_factories.items():
        fold_scores = []
        for train_idx, test_idx in skf.split(X, y):
            model = _fit(name, factory(random_state), X[train_idx], y[train_idx])
            fold_scores.append(_score(model, X[test_idx], y[test_idx]))
        fold_df = pd.DataFrame(fold_scores)
        rows.append({
            "model": name,
            **{f"{col}_mean": fold_df[col].mean() for col in fold_df.columns},
            **{f"{col}_std": fold_df[col].std() for col in fold_df.columns},
        })
    return pd.DataFrame(rows)


def leave_one_out_cv(features_df, model_factories=MODEL_FACTORIES, feature_cols=FEATURE_COLS,
                      label_col="SUSPENDED", random_state=42):
    """With only 24 positives, 5-fold CV leaves ~5 positives per test fold — too few to
    trust. Leave-one-out holds out a single day at a time (n folds), so every prediction
    comes from a model trained on effectively 'all the rest of the data.' Metrics are
    computed once over all n held-out predictions pooled together, not averaged per-fold
    (a per-fold precision/recall is undefined when a fold has just one sample)."""
    X = features_df[feature_cols].to_numpy()
    y = features_df[label_col].to_numpy()
    loo = LeaveOneOut()

    rows = []
    for name, factory in model_factories.items():
        y_true, y_pred = [], []
        for train_idx, test_idx in loo.split(X):
            model = _fit(name, factory(random_state), X[train_idx], y[train_idx])
            y_pred.append(model.predict(X[test_idx])[0])
            y_true.append(y[test_idx][0])
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        })
    return pd.DataFrame(rows)


def suggest_suspension(model, rainfall, tmax, tmin, rh, wind_speed, month,
                        rain_3day_sum=None, rain_7day_sum=None):
    """Suggestion model: given a day's weather inputs, return (label, probability)."""
    rain_3day_sum = rainfall if rain_3day_sum is None else rain_3day_sum
    rain_7day_sum = rainfall if rain_7day_sum is None else rain_7day_sum

    row = pd.DataFrame([{
        "RAINFALL": rainfall,
        "RAIN_3DAY_SUM": rain_3day_sum,
        "RAIN_7DAY_SUM": rain_7day_sum,
        "TMAX": tmax,
        "TMIN": tmin,
        "RH": rh,
        "WIND_SPEED": wind_speed,
        "MONTH": month,
    }])[FEATURE_COLS]

    proba = model.predict_proba(row)[0][1]
    label = next(text for threshold, text in SUGGESTION_BANDS if proba >= threshold)
    return label, proba


if __name__ == "__main__":
    weather = load_weather()
    suspensions = load_suspensions()

    # Per user's assumption: any date not in the scraped suspension sheet had
    # no weather-related suspension. Use the full 2023-2024 weather range.
    dataset, coverage_start, coverage_end, dropped = build_labeled_dataset(
        weather, suspensions, start="2023-01-01", end="2024-12-31"
    )
    features = build_features(dataset)

    print(f"Usable training window: {coverage_start.date()} to {coverage_end.date()}")
    print(f"Total days in window: {len(features)}")
    print(f"Suspended days in window: {features.SUSPENDED.sum()} "
          f"({features.SUSPENDED.mean():.1%} of window)")
    if len(dropped):
        print(f"\nExcluded {len(dropped)} scraped suspension date(s) outside the weather "
              f"data window (no matching weather row): {dropped['Date'].dt.date.tolist()}")

    model, metrics, (X_test, y_test, y_pred) = train_random_forest(features)

    print("\n=== Random Forest (default, unconstrained) — held-out test split ===")
    print(f"Accuracy:  {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1-score:  {metrics['f1']:.3f}")
    print("\nConfusion matrix ([[TN, FP], [FN, TP]]):")
    print(metrics["confusion_matrix"])
    print("\nFull report:")
    print(metrics["report"])

    print("=== Feature importances (default, unconstrained RF) ===")
    print(feature_importance(model))

    all_models = {**MODEL_FACTORIES, **REGULARIZED_MODEL_FACTORIES}

    print("\n=== Train vs. test scores (overfitting check) ===")
    diag = train_vs_test_diagnostic(features, model_factories=all_models)
    print(diag.to_string(index=False))

    print("\n=== 5-fold stratified cross-validation (default vs. regularized) ===")
    cv = cross_validate_models(features, model_factories=all_models)
    print(cv[["model", "accuracy_mean", "precision_mean", "recall_mean", "f1_mean",
              "precision_std", "recall_std"]].to_string(index=False))

    print("\n=== Leave-one-out CV: default vs. depth/leaf-regularized trees ===")
    loo = leave_one_out_cv(features, model_factories=all_models)
    print(loo.to_string(index=False))

    # Examples use the model the API serves: regularized RF fit on all labeled days.
    served_model = REGULARIZED_MODEL_FACTORIES["Random Forest (regularized)"](random_state=42)
    served_model.fit(features[FEATURE_COLS], features["SUSPENDED"])

    print("\n=== Example suggestions (regularized RF, as served) ===")
    examples = [
        dict(rainfall=150, tmax=28, tmin=24, rh=92, wind_speed=6, month=8, rain_3day_sum=300, rain_7day_sum=400),
        dict(rainfall=5, tmax=31, tmin=23, rh=60, wind_speed=4, month=3, rain_3day_sum=5, rain_7day_sum=10),
        dict(rainfall=40, tmax=28, tmin=24, rh=85, wind_speed=5, month=9, rain_3day_sum=70, rain_7day_sum=90),
    ]
    for ex in examples:
        label, proba = suggest_suspension(served_model, **ex)
        print(f"{ex} -> {label} (p={proba:.2f})")
