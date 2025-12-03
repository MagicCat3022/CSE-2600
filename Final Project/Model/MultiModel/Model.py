from pathlib import Path
import argparse
import joblib
import shutil
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    make_scorer,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import KFold, ParameterGrid, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from catboost import CatBoostClassifier, CatBoostRegressor

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "Data" / "Updated_Data.csv"

TARGET = "TOTAL_LP"
EXCLUDED_FEATURES = {"tier", "rank", "tier_int", "rank_int", "leaguepoints", "lp", "total_lp"}

RANDOM_STATE = 67
TEST_SIZE = 0.2
N_SPLITS = 5
TOP_K = 10

# Classifier parameter grid
CLASSIFIER_PARAM_GRID = {
    "classifier__iterations": [5000],
    "classifier__depth": [9],
    "classifier__learning_rate": [0.07],
    "classifier__l2_leaf_reg": [0],
}

# Regressor parameter grid (shared across all bins)
REGRESSOR_PARAM_GRID = {
    "regressor__iterations": [500],
    "regressor__depth": [8, 10],
    "regressor__learning_rate": [0.1],
    "regressor__l2_leaf_reg": [0],
    "regressor__subsample": [0.7],
}

CLEAN_CLASSIFIER_PARAM_NAMES = [
    key.replace("classifier__", "") for key in CLASSIFIER_PARAM_GRID
]
CLEAN_REGRESSOR_PARAM_NAMES = [
    key.replace("regressor__", "") for key in REGRESSOR_PARAM_GRID
]


class MultiModelPredictor(BaseEstimator, RegressorMixin):
    """
    Combined model that first classifies into bins, then uses bin-specific regressor.
    Compatible with Check.py for evaluation.
    """

    def __init__(self, classifier_pipeline, regressor_pipelines, bin_edges, 
                 classifier_features, regressor_features):
        self.classifier_pipeline = classifier_pipeline
        self.regressor_pipelines = regressor_pipelines
        self.bin_edges = bin_edges
        self.classifier_features = classifier_features
        self.regressor_features = regressor_features

    def fit(self, X, y):
        # Already fitted during training
        return self

    def predict(self, X):
        # Select features for classifier
        X_classifier = X[self.classifier_features]
        
        # Predict which bin each sample belongs to
        bin_predictions = self.classifier_pipeline.predict(X_classifier)

        # Initialize predictions array
        predictions = np.zeros(len(X))

        # For each bin, use the corresponding regressor
        for bin_idx in range(len(self.regressor_pipelines)):
            mask = bin_predictions == bin_idx
            if mask.sum() > 0:
                # Select features for this bin's regressor and filter by mask
                X_regressor = X.loc[mask, self.regressor_features[bin_idx]]
                predictions[mask] = self.regressor_pipelines[bin_idx].predict(
                    X_regressor
                )

        return predictions

    def get_params(self, deep=True):
        return {
            "classifier_pipeline": self.classifier_pipeline,
            "regressor_pipelines": self.regressor_pipelines,
            "bin_edges": self.bin_edges,
            "classifier_features": self.classifier_features,
            "regressor_features": self.regressor_features,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(DATA_PATH, low_memory=False)
    if TARGET not in data.columns:
        raise ValueError(f"{TARGET} not found in {DATA_PATH.name}.")
    data = data.dropna(subset=[TARGET])

    feature_cols = [
        col
        for col in data.columns
        if col != TARGET and col.lower() not in EXCLUDED_FEATURES
    ]
    if not feature_cols:
        raise ValueError("No features remain after exclusions.")

    excluded_cols = [
        col
        for col in data.columns
        if col.lower() in EXCLUDED_FEATURES and col != TARGET
    ]
    print(f"Loaded {len(data)} rows.")
    if excluded_cols:
        print(f"Excluded LP/rank columns: {', '.join(excluded_cols)}")

    X = data[feature_cols].copy()
    y = data[TARGET].astype(float)
    return X, y


def determine_feature_types(
    feature_frame: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    categorical_features = feature_frame.select_dtypes(
        include=["object"]
    ).columns.tolist()
    numeric_features = [
        col for col in feature_frame.columns if col not in categorical_features
    ]
    return numeric_features, categorical_features


def create_bins(y_train: pd.Series, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Create equal-width bins based on training data only.
    Returns bin edges and binned labels.
    """
    min_val = y_train.min()
    max_val = y_train.max()
    bin_edges = np.linspace(min_val, max_val, n_bins + 1)
    y_binned = pd.cut(y_train, bins=bin_edges, labels=False, include_lowest=True)
    print(f"\nCreated {n_bins} equal-width bins:")
    for i in range(n_bins):
        print(f"  Bin {i}: [{bin_edges[i]:.1f}, {bin_edges[i+1]:.1f}]")
    return bin_edges, y_binned.values


def assign_bins(y: pd.Series, bin_edges: np.ndarray) -> np.ndarray:
    """Assign data to bins using pre-defined bin edges."""
    return pd.cut(y, bins=bin_edges, labels=False, include_lowest=True).values


def select_top_features_classifier(
    X: pd.DataFrame, y: np.ndarray, n_features: int
) -> pd.DataFrame:
    """
    Select top features for classification using CatBoost feature importance.
    """
    print(f"\n--- Feature Selection for Classifier (Target: Top {n_features}) ---")
    numeric_cols, cat_cols = determine_feature_types(X)

    if len(X.columns) <= n_features:
        print(
            f"Dataset has {len(X.columns)} features, which is <= {n_features}. No reduction needed."
        )
        return X

    # Prepare data
    X_prep = X.copy()
    for col in numeric_cols:
        X_prep[col] = X_prep[col].fillna(X_prep[col].median())
    for col in cat_cols:
        X_prep[col] = X_prep[col].fillna("missing").astype(str)

    cat_feature_indices = [X_prep.columns.get_loc(col) for col in cat_cols]

    # Lightweight CatBoost for feature selection
    selector_model = CatBoostClassifier(
        iterations=250,
        depth=6,
        learning_rate=0.15,
        loss_function="MultiClass",
        cat_features=cat_feature_indices,
        random_seed=RANDOM_STATE,
        verbose=False,
    )

    print("Training classifier scout model for feature importance...")
    selector_model.fit(X_prep, y)

    importances = selector_model.get_feature_importance()
    feature_names = X_prep.columns.tolist()

    feat_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
    feat_imp = feat_imp.sort_values(by="importance", ascending=False)

    top_features = feat_imp.head(n_features)["feature"].tolist()

    print(f"Top 5 features: {top_features[:5]}")
    print(
        f"Reduced feature space from {len(X.columns)} to {len(top_features)} for classifier."
    )

    return X[top_features]


def select_top_features_regressor(
    X: pd.DataFrame, y: pd.Series, n_features: int, bin_idx: int
) -> pd.DataFrame:
    """
    Select top features for a specific bin's regressor using CatBoost feature importance.
    """
    print(
        f"\n--- Feature Selection for Bin {bin_idx} Regressor (Target: Top {n_features}) ---"
    )
    numeric_cols, cat_cols = determine_feature_types(X)

    if len(X.columns) <= n_features:
        print(
            f"Dataset has {len(X.columns)} features, which is <= {n_features}. No reduction needed."
        )
        return X

    # Prepare data
    X_prep = X.copy()
    for col in numeric_cols:
        X_prep[col] = X_prep[col].fillna(X_prep[col].median())
    for col in cat_cols:
        X_prep[col] = X_prep[col].fillna("missing").astype(str)

    cat_feature_indices = [X_prep.columns.get_loc(col) for col in cat_cols]

    # Lightweight CatBoost for feature selection
    selector_model = CatBoostRegressor(
        iterations=250,
        depth=6,
        learning_rate=0.15,
        loss_function="RMSE",
        cat_features=cat_feature_indices,
        random_seed=RANDOM_STATE,
        verbose=False,
    )

    print(f"Training regressor scout model for bin {bin_idx} feature importance...")
    selector_model.fit(X_prep, y)

    importances = selector_model.get_feature_importance()
    feature_names = X_prep.columns.tolist()

    feat_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
    feat_imp = feat_imp.sort_values(by="importance", ascending=False)

    top_features = feat_imp.head(n_features)["feature"].tolist()

    print(f"Top 5 features: {top_features[:5]}")
    print(
        f"Reduced feature space from {len(X.columns)} to {len(top_features)} for bin {bin_idx}."
    )

    return X[top_features]


def build_classifier_pipeline(
    numeric_features: list[str], categorical_features: list[str]
) -> Pipeline:
    """Build a pipeline for CatBoost classification."""
    transformers = []
    if numeric_features:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric_features))
    if categorical_features:
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                (
                    "encoder",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    ),
                ),
            ]
        )
        transformers.append(("cat", categorical_transformer, categorical_features))
    if not transformers:
        raise ValueError("No columns available for preprocessing.")
    preprocessor = ColumnTransformer(transformers)

    classifier = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="Accuracy",
        random_seed=RANDOM_STATE,
        verbose=False,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("classifier", classifier)])


def build_regressor_pipeline(
    numeric_features: list[str], categorical_features: list[str]
) -> Pipeline:
    """Build a pipeline for CatBoost regression."""
    transformers = []
    if numeric_features:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric_features))
    if categorical_features:
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                (
                    "encoder",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    ),
                ),
            ]
        )
        transformers.append(("cat", categorical_transformer, categorical_features))
    if not transformers:
        raise ValueError("No columns available for preprocessing.")
    preprocessor = ColumnTransformer(transformers)

    regressor = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=RANDOM_STATE,
        verbose=False,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("regressor", regressor)])


def rmse_score(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def append_classifier_result(row: dict, filepath: Path, header_written: bool) -> bool:
    ordered_row = {}
    if "model_filename" in row:
        ordered_row["model_filename"] = row["model_filename"]
    for col in ("mean_cv_accuracy", "std_cv_accuracy", "mean_cv_f1", "mean_cv_precision", "mean_cv_recall"):
        if col in row:
            ordered_row[col] = row[col]
    for col in CLEAN_CLASSIFIER_PARAM_NAMES:
        if col in row:
            ordered_row[col] = row[col]
    pd.DataFrame([ordered_row]).to_csv(
        filepath,
        mode="a",
        header=not header_written,
        index=False,
    )
    return True


def append_regressor_result(row: dict, filepath: Path, header_written: bool) -> bool:
    ordered_row = {}
    if "model_filename" in row:
        ordered_row["model_filename"] = row["model_filename"]
    for col in ("mean_cv_rmse", "std_cv_rmse", "mean_cv_mae", "mean_cv_r2"):
        if col in row:
            ordered_row[col] = row[col]
    for col in CLEAN_REGRESSOR_PARAM_NAMES:
        if col in row:
            ordered_row[col] = row[col]
    pd.DataFrame([ordered_row]).to_csv(
        filepath,
        mode="a",
        header=not header_written,
        index=False,
    )
    return True


def train_classifier(
    X_train: pd.DataFrame,
    y_train_binned: np.ndarray,
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[pd.DataFrame, dict, Pipeline]:
    """
    Train and evaluate classifier with hyperparameter search.
    """
    print("\n" + "=" * 70)
    print("TRAINING CLASSIFIER (Bin Prediction)")
    print("=" * 70)

    results_path = BASE_DIR / "multimodel_classifier_results.csv"
    results_path.unlink(missing_ok=True)
    header_written = False

    models_dir = BASE_DIR / "all_models" / "classifier"
    models_dir.mkdir(parents=True, exist_ok=True)

    param_grid = list(ParameterGrid(CLASSIFIER_PARAM_GRID))
    total_configs = len(param_grid)
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    results = []
    print(f"Evaluating {total_configs} hyperparameter combinations for classifier...")

    for idx, params in enumerate(param_grid, start=1):
        pipeline = build_classifier_pipeline(numeric_features, categorical_features)
        pipeline.set_params(**params)

        cv_scores = cross_validate(
            pipeline,
            X_train,
            y_train_binned,
            cv=kfold,
            scoring={
                "accuracy": "accuracy",
                "f1": make_scorer(f1_score, average="weighted"),
                "precision": make_scorer(precision_score, average="weighted", zero_division=0),
                "recall": make_scorer(recall_score, average="weighted"),
            },
            n_jobs=-1,
        )

        # Fit on full training set and save
        pipeline.fit(X_train, y_train_binned)
        model_filename = f"classifier_{idx}.joblib"
        joblib.dump(pipeline, models_dir / model_filename)

        row = {
            "model_filename": model_filename,
            "mean_cv_accuracy": cv_scores["test_accuracy"].mean(),
            "std_cv_accuracy": cv_scores["test_accuracy"].std(),
            "mean_cv_f1": cv_scores["test_f1"].mean(),
            "mean_cv_precision": cv_scores["test_precision"].mean(),
            "mean_cv_recall": cv_scores["test_recall"].mean(),
            "params_prefixed": params,
        }
        clean_params = {k.replace("classifier__", ""): v for k, v in params.items()}
        row.update(clean_params)
        results.append(row)

        header_written = append_classifier_result(row, results_path, header_written)

        print(
            f"[{idx:02d}/{total_configs}] Accuracy: {cv_scores['test_accuracy'].mean():.3f}, "
            f"F1: {cv_scores['test_f1'].mean():.3f}, params: {clean_params}"
        )

    results_df = pd.DataFrame(results).sort_values("mean_cv_accuracy", ascending=False).reset_index(drop=True)
    best_params = results_df.iloc[0]["params_prefixed"]

    # Train best classifier on full data
    best_pipeline = build_classifier_pipeline(numeric_features, categorical_features)
    best_pipeline.set_params(**best_params)
    best_pipeline.fit(X_train, y_train_binned)

    # Display top models
    print("\nTop classifier models by cross-validated accuracy:")
    cols_to_show = ["model_filename", "mean_cv_accuracy", "mean_cv_f1", "mean_cv_precision", "mean_cv_recall"] + CLEAN_CLASSIFIER_PARAM_NAMES
    cols_to_show = [col for col in cols_to_show if col in results_df.columns]
    print(results_df.head(TOP_K)[cols_to_show].to_string(index=False))

    # Save results
    cols = [col for col in results_df.columns if col != "params_prefixed"]
    results_df[cols].to_csv(results_path, index=False)
    print(f"\nClassifier results saved to {results_path}")

    return results_df, best_params, best_pipeline


def train_regressor_for_bin(
    X_train_bin: pd.DataFrame,
    y_train_bin: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    bin_idx: int,
) -> tuple[pd.DataFrame, dict, Pipeline]:
    """
    Train and evaluate regressor for a specific bin with hyperparameter search.
    """
    print("\n" + "=" * 70)
    print(f"TRAINING REGRESSOR FOR BIN {bin_idx} ({len(y_train_bin)} samples)")
    print("=" * 70)

    results_path = BASE_DIR / f"multimodel_regressor_bin_{bin_idx}_results.csv"
    results_path.unlink(missing_ok=True)
    header_written = False

    models_dir = BASE_DIR / "all_models" / f"regressor_bin_{bin_idx}"
    models_dir.mkdir(parents=True, exist_ok=True)

    param_grid = list(ParameterGrid(REGRESSOR_PARAM_GRID))
    total_configs = len(param_grid)
    kfold = KFold(n_splits=min(N_SPLITS, len(y_train_bin)), shuffle=True, random_state=RANDOM_STATE)

    rmse_scorer = make_scorer(rmse_score, greater_is_better=False)
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    results = []
    print(f"Evaluating {total_configs} hyperparameter combinations for bin {bin_idx} regressor...")

    for idx, params in enumerate(param_grid, start=1):
        pipeline = build_regressor_pipeline(numeric_features, categorical_features)
        pipeline.set_params(**params)

        cv_scores = cross_validate(
            pipeline,
            X_train_bin,
            y_train_bin,
            cv=kfold,
            scoring={"rmse": rmse_scorer, "mae": mae_scorer, "r2": "r2"},
            n_jobs=-1,
        )

        # Fit on full bin training set and save
        pipeline.fit(X_train_bin, y_train_bin)
        model_filename = f"regressor_bin_{bin_idx}_{idx}.joblib"
        joblib.dump(pipeline, models_dir / model_filename)

        rmse_scores = -cv_scores["test_rmse"]
        mae_scores = -cv_scores["test_mae"]

        row = {
            "model_filename": model_filename,
            "mean_cv_rmse": rmse_scores.mean(),
            "std_cv_rmse": rmse_scores.std(),
            "mean_cv_mae": mae_scores.mean(),
            "mean_cv_r2": cv_scores["test_r2"].mean(),
            "params_prefixed": params,
        }
        clean_params = {k.replace("regressor__", ""): v for k, v in params.items()}
        row.update(clean_params)
        results.append(row)

        header_written = append_regressor_result(row, results_path, header_written)

        print(
            f"[{idx:02d}/{total_configs}] RMSE: {rmse_scores.mean():.3f}, "
            f"MAE: {mae_scores.mean():.3f}, R²: {cv_scores['test_r2'].mean():.3f}, "
            f"params: {clean_params}"
        )

    results_df = pd.DataFrame(results).sort_values("mean_cv_rmse").reset_index(drop=True)
    best_params = results_df.iloc[0]["params_prefixed"]

    # Train best regressor on full bin data
    best_pipeline = build_regressor_pipeline(numeric_features, categorical_features)
    best_pipeline.set_params(**best_params)
    best_pipeline.fit(X_train_bin, y_train_bin)

    # Display top models
    print(f"\nTop regressor models for bin {bin_idx} by cross-validated RMSE:")
    cols_to_show = ["model_filename", "mean_cv_rmse", "mean_cv_mae", "mean_cv_r2"] + CLEAN_REGRESSOR_PARAM_NAMES
    cols_to_show = [col for col in cols_to_show if col in results_df.columns]
    print(results_df.head(TOP_K)[cols_to_show].to_string(index=False))

    # Save results
    cols = [col for col in results_df.columns if col != "params_prefixed"]
    results_df[cols].to_csv(results_path, index=False)
    print(f"\nBin {bin_idx} regressor results saved to {results_path}")

    return results_df, best_params, best_pipeline


def evaluate_combined_model(
    classifier: Pipeline,
    regressors: list[Pipeline],
    bin_edges: np.ndarray,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    classifier_features: list[str],
    regressor_features: list[list[str]],
) -> dict:
    """
    Evaluate the combined multi-model on test data.
    """
    print("\n" + "=" * 70)
    print("EVALUATING COMBINED MULTI-MODEL ON TEST SET")
    print("=" * 70)

    # Select features for classifier and predict bins
    X_test_classifier = X_test[classifier_features]
    y_test_bins_pred = np.asarray(classifier.predict(X_test_classifier)).ravel()
    y_test_bins_true = np.asarray(assign_bins(y_test, bin_edges)).ravel()

    # Calculate classifier accuracy on test set
    bin_accuracy = accuracy_score(y_test_bins_true, y_test_bins_pred)
    print(f"Test set bin classification accuracy: {bin_accuracy:.3f}")

    # Predict TOTAL_LP using appropriate regressor for each bin
    predictions = np.zeros(len(X_test))
    for bin_idx in range(len(regressors)):
        mask = y_test_bins_pred == bin_idx
        if mask.sum() > 0:
            # Select features for this bin's regressor
            X_test_regressor = X_test.loc[mask, regressor_features[bin_idx]]
            predictions[mask] = regressors[bin_idx].predict(X_test_regressor)

    # Calculate regression metrics
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    mae = float(mean_absolute_error(y_test, predictions))
    r2 = float(r2_score(y_test, predictions))

    print(f"\nCombined model test set performance:")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  MAE:  {mae:.3f}")
    print(f"  R²:   {r2:.3f}")

    # Per-bin breakdown
    print("\nPer-bin test performance:")
    for bin_idx in range(len(regressors)):
        mask_true = y_test_bins_true == bin_idx
        mask_pred = y_test_bins_pred == bin_idx

        if mask_true.sum() > 0:
            bin_rmse = np.sqrt(mean_squared_error(y_test[mask_pred], predictions[mask_pred])) if mask_pred.sum() > 0 else np.nan
            bin_r2 = r2_score(y_test[mask_pred], predictions[mask_pred]) if mask_pred.sum() > 1 else np.nan
            print(
                f"  Bin {bin_idx}: {mask_true.sum()} true samples, "
                f"{mask_pred.sum()} predicted samples, "
                f"RMSE: {bin_rmse:.3f}, R²: {bin_r2:.3f}"
            )

    return {
        "bin_accuracy": bin_accuracy,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-Model (Classifier + Bin-specific Regressors) trainer."
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=5,
        help="Number of equal-width bins for TOTAL_LP.",
    )
    parser.add_argument(
        "--n-features-classifier",
        type=int,
        default=20,
        help="Number of top features to select for classifier. Set to 0 to use all features.",
    )
    parser.add_argument(
        "--n-features-regressor",
        type=int,
        default=20,
        help="Number of top features to select for each bin's regressor. Set to 0 to use all features.",
    )
    args = parser.parse_args()

    # Load data
    X, y = load_dataset()

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Create bins based on training data
    bin_edges, y_train_binned = create_bins(y_train, args.n_bins)
    y_test_binned = assign_bins(y_test, bin_edges)

    # Feature selection for classifier
    X_train_classifier = X_train.copy()
    X_test_classifier = X_test.copy()
    if args.n_features_classifier > 0:
        X_train_classifier = select_top_features_classifier(
            X_train, y_train_binned, args.n_features_classifier
        )
        X_test_classifier = X_test[X_train_classifier.columns]

    numeric_features_classifier, categorical_features_classifier = (
        determine_feature_types(X_train_classifier)
    )

    # Train classifier
    classifier_results, best_classifier_params, best_classifier = train_classifier(
        X_train_classifier,
        y_train_binned,
        numeric_features_classifier,
        categorical_features_classifier,
    )

    # Save best classifier
    best_models_dir = BASE_DIR / "best_models"
    best_models_dir.mkdir(parents=True, exist_ok=True)
    classifier_path = best_models_dir / "best_classifier.joblib"
    joblib.dump(best_classifier, classifier_path)
    print(f"\nSaved best classifier to {classifier_path}")

    # Train regressor for each bin
    best_regressors = []
    regressor_feature_lists = []  # Track which features each regressor uses
    
    for bin_idx in range(args.n_bins):
        # Get samples in this bin
        bin_mask = y_train_binned == bin_idx
        X_train_bin = X_train[bin_mask].copy()
        y_train_bin = y_train[bin_mask]

        if len(y_train_bin) < 10:
            print(f"\nWarning: Bin {bin_idx} has only {len(y_train_bin)} samples. Skipping hyperparameter search.")
            # Use default regressor
            numeric_features_bin, categorical_features_bin = determine_feature_types(X_train_bin)
            pipeline = build_regressor_pipeline(numeric_features_bin, categorical_features_bin)
            pipeline.fit(X_train_bin, y_train_bin)
            best_regressors.append(pipeline)
            regressor_feature_lists.append(X_train_bin.columns.tolist())
            continue

        # Feature selection for this bin's regressor
        if args.n_features_regressor > 0:
            X_train_bin = select_top_features_regressor(
                X_train_bin, y_train_bin, args.n_features_regressor, bin_idx
            )

        numeric_features_bin, categorical_features_bin = determine_feature_types(
            X_train_bin
        )

        # Train regressor for this bin
        regressor_results, best_regressor_params, best_regressor = (
            train_regressor_for_bin(
                X_train_bin,
                y_train_bin,
                numeric_features_bin,
                categorical_features_bin,
                bin_idx,
            )
        )

        best_regressors.append(best_regressor)
        regressor_feature_lists.append(X_train_bin.columns.tolist())

        # Save best regressor for this bin
        regressor_path = best_models_dir / f"best_regressor_bin_{bin_idx}.joblib"
        joblib.dump(best_regressor, regressor_path)
        print(f"Saved best regressor for bin {bin_idx} to {regressor_path}")

    # Create and save combined model
    combined_model = MultiModelPredictor(
        best_classifier, best_regressors, bin_edges,
        X_train_classifier.columns.tolist(), regressor_feature_lists
    )
    combined_path = best_models_dir / "best_combined_model.joblib"
    joblib.dump(combined_model, combined_path)
    print(f"\nSaved combined multi-model to {combined_path}")

    # Evaluate combined model on test set
    test_metrics = evaluate_combined_model(
        best_classifier, best_regressors, bin_edges, X_test, y_test,
        X_train_classifier.columns.tolist(), regressor_feature_lists
    )

    # Save test data with predictions
    test_data = X_test.copy()
    test_data[TARGET] = y_test
    test_data.to_csv(BASE_DIR / "test_data.csv", index=False)
    print(f"\nTest data saved to {BASE_DIR / 'test_data.csv'}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Classifier results: {BASE_DIR / 'multimodel_classifier_results.csv'}")
    for bin_idx in range(args.n_bins):
        print(f"Bin {bin_idx} regressor results: {BASE_DIR / f'multimodel_regressor_bin_{bin_idx}_results.csv'}")
    print(f"Combined model (for Check.py): {combined_path}")


if __name__ == "__main__":
    main()
