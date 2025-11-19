from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
)
from sklearn.model_selection import (
    ParameterGrid,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Updated_Data.csv"
RESULTS_PATH = BASE_DIR / "model_search_results.csv"

TARGET_NAME = "tier_rank_combined"
EXCLUDED_FEATURES = {
    "tier",
    "rank",
    "tier_int",
    "rank_int",
    "total_lp",
    "lp",
    "leaguepoints",
    "totallp",
}

TEST_SIZE = 0.2
CV_SPLITS = 5
RANDOM_STATE = 42
N_JOBS = -1  # for sklearn CV where applicable

XGB_PARAM_GRID = {
    "model__n_estimators": [2000, 5000],
    "model__max_depth": [4, 6],
    "model__learning_rate": [0.05, 0.1],
    "model__subsample": [0.8, 1.0],
    "model__colsample_bytree": [0.7],
    "model__reg_lambda": [1.0, 1.5],
}

RF_PARAM_GRID = {
    "model__n_estimators": [3000, 5000],
    "model__max_depth": [30, None],
    "model__max_features": ["sqrt", 0.6],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 3],
}

BASE_RESULT_COLUMNS = [
    "algorithm",
    "mean_accuracy",
    "std_accuracy",
    "mean_misclassification",
    "std_misclassification",
    "mean_macro_f1",
    "std_macro_f1",
    "mean_log_loss",
    "std_log_loss",
]
PARAM_COLUMNS = sorted(set(XGB_PARAM_GRID.keys()) | set(RF_PARAM_GRID.keys()))


def load_dataset() -> Tuple[pd.DataFrame, np.ndarray, LabelEncoder]:
    data = pd.read_csv(DATA_PATH, low_memory=False)

    if not {"tier", "rank"}.issubset(data.columns):
        raise ValueError("Both 'tier' and 'rank' columns must be present.")

    tier = data["tier"].astype(str).str.strip().str.upper()
    rank = data["rank"].astype(str).str.strip().str.upper()
    combined = tier + "_" + rank
    data[TARGET_NAME] = combined.replace({"NAN_NAN": np.nan})
    data = data.dropna(subset=[TARGET_NAME])

    feature_columns = [
        col
        for col in data.columns
        if col != TARGET_NAME and col.lower() not in EXCLUDED_FEATURES
    ]
    if not feature_columns:
        raise ValueError("No usable feature columns remain after exclusions.")

    X = data[feature_columns].copy()
    y_text = data[TARGET_NAME].astype(str).values

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_text)

    return X, y_encoded, label_encoder


def split_features(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [col for col in X.columns if col not in categorical]
    return numeric, categorical


def build_preprocessor(
    numeric_features: List[str], categorical_features: List[str]
) -> ColumnTransformer:
    transformers = []
    if numeric_features:
        transformers.append(
            ("num", SimpleImputer(strategy="median"), numeric_features)
        )
    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
                    ]
                ),
                categorical_features,
            )
        )
    return ColumnTransformer(transformers)


def make_pipeline(
    algorithm: str, num_classes: int, preprocessor: ColumnTransformer
) -> Pipeline:
    if algorithm == "xgboost":
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=num_classes,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        )
    elif algorithm == "random_forest":
        model = RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            class_weight="balanced",
        )
    else:  # pragma: no cover
        raise ValueError(f"Unsupported algorithm '{algorithm}'.")

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def append_partial_result(row: dict, filepath: Path, header_written: bool) -> bool:
    ordered = {col: row.get(col, np.nan) for col in BASE_RESULT_COLUMNS}
    ordered.update({col: row.get(col, np.nan) for col in PARAM_COLUMNS})
    pd.DataFrame([ordered]).to_csv(
        filepath,
        mode="a",
        header=not header_written,
        index=False,
    )
    return True


def run_search(
    algorithm: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    param_grid: Dict[str, Iterable],
    preprocessor: ColumnTransformer,
    num_classes: int,
    results_path: Path,
    header_written: bool,
) -> Tuple[pd.DataFrame, Dict[str, object], bool]:
    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "neg_log_loss": "neg_log_loss",
    }
    cv = StratifiedKFold(
        n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )

    results = []
    grid = list(ParameterGrid(param_grid))
    total = len(grid)
    print(f"\n=== {algorithm.upper()} hyperparameter search ({total} combinations) ===")
    for idx, params in enumerate(grid, start=1):
        pipeline = make_pipeline(algorithm, num_classes, preprocessor)
        pipeline.set_params(**params)

        cv_scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            scoring=scoring,
            cv=cv,
            n_jobs=N_JOBS,
            error_score="raise",
        )

        accuracy_scores = cv_scores["test_accuracy"]
        misclassification_scores = 1 - accuracy_scores
        f1_scores = cv_scores["test_f1_macro"]
        logloss_scores = -cv_scores["test_neg_log_loss"]

        row = {
            "algorithm": algorithm,
            "mean_accuracy": accuracy_scores.mean(),
            "std_accuracy": accuracy_scores.std(),
            "mean_misclassification": misclassification_scores.mean(),
            "std_misclassification": misclassification_scores.std(),
            "mean_macro_f1": f1_scores.mean(),
            "std_macro_f1": f1_scores.std(),
            "mean_log_loss": logloss_scores.mean(),
            "std_log_loss": logloss_scores.std(),
        }
        row.update(params)
        results.append(row)
        header_written = append_partial_result(row, results_path, header_written)

        print(
            f"[{algorithm} {idx:02d}/{total}] "
            f"acc={row['mean_accuracy']:.4f}, f1={row['mean_macro_f1']:.4f}, "
            f"logloss={row['mean_log_loss']:.4f}"
        )

    results_df = pd.DataFrame(results).sort_values(
        by=["mean_accuracy", "mean_macro_f1"], ascending=False
    )
    best_params = results_df.iloc[0].drop(labels=["algorithm"]).to_dict()
    return results_df.reset_index(drop=True), best_params, header_written


def evaluate_holdout(
    algorithm: str,
    params: Dict[str, object],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    preprocessor: ColumnTransformer,
    num_classes: int,
) -> Dict[str, float]:
    pipeline = make_pipeline(algorithm, num_classes, preprocessor)
    pipeline.set_params(**params)
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    probas = pipeline.predict_proba(X_test)

    accuracy = accuracy_score(y_test, preds)
    misclassification = 1.0 - accuracy
    macro_f1 = f1_score(y_test, preds, average="macro")
    loss = log_loss(y_test, probas)

    metrics = {
        "holdout_accuracy": accuracy,
        "holdout_misclassification": misclassification,
        "holdout_macro_f1": macro_f1,
        "holdout_log_loss": loss,
    }
    return metrics


def persist_results(results: List[pd.DataFrame], filepath: Path) -> None:
    merged = pd.concat(results, ignore_index=True)
    merged.to_csv(filepath, index=False)
    print(f"\nSaved {len(merged)} rows to {filepath}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank/Tier classifier trainer.")
    parser.add_argument(
        "--model-type",
        choices=["xgboost", "random_forest", "both"],
        default="both",
        help="Which algorithm(s) to evaluate.",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=RESULTS_PATH,
        help="Where to save the CSV of all evaluated configurations.",
    )
    args = parser.parse_args()

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    args.results_path.unlink(missing_ok=True)
    header_written = False

    X, y, label_encoder = load_dataset()
    numeric_features, categorical_features = split_features(X)
    print(
        f"Loaded {len(X)} samples "
        f"({len(numeric_features)} numeric, {len(categorical_features)} categorical features)."
    )
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    num_classes = len(label_encoder.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    algo_sequence = (
        ["xgboost", "random_forest"]
        if args.model_type == "both"
        else [args.model_type]
    )

    all_results = []
    for algo in algo_sequence:
        grid = XGB_PARAM_GRID if algo == "xgboost" else RF_PARAM_GRID
        results_df, best_params, header_written = run_search(
            algo,
            X_train,
            y_train,
            grid,
            preprocessor,
            num_classes,
            args.results_path,
            header_written,
        )
        all_results.append(results_df)

        holdout_metrics = evaluate_holdout(
            algo,
            {k: v for k, v in best_params.items() if k.startswith("model__")},
            X_train,
            y_train,
            X_test,
            y_test,
            preprocessor,
            num_classes,
        )
        print(
            f"\nBest {algo} hold-out metrics: "
            f"acc={holdout_metrics['holdout_accuracy']:.4f}, "
            f"misclass={holdout_metrics['holdout_misclassification']:.4f}, "
            f"macro_f1={holdout_metrics['holdout_macro_f1']:.4f}, "
            f"log_loss={holdout_metrics['holdout_log_loss']:.4f}\n"
        )

    print(f"\nIncremental results saved to {args.results_path}")


if __name__ == "__main__":
    main()