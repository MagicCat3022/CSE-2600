from pathlib import Path
import numpy as np

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.model_selection import KFold, ParameterGrid, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Updated_Data.csv"
RESULTS_PATH = BASE_DIR / "xgb_model_results.csv"

TARGET = "TOTAL_LP"
EXCLUDED_FEATURES = {"tier", "rank", "tier_int", "rank_int", "leaguepoints", "lp", "total_lp"}

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5
TOP_K = 10

PARAM_GRID = {
    "regressor__n_estimators": [4000, 5500],
    "regressor__max_depth": [6, 8],
    "regressor__learning_rate": [0.1],
    "regressor__subsample": [1.0],
    "regressor__colsample_bytree": [0.7],
    "regressor__gamma": [1],
    "regressor__reg_lambda": [1.5],
}
CLEAN_PARAM_NAMES = [key.replace("regressor__", "") for key in PARAM_GRID]


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(DATA_PATH, low_memory=False)
    if TARGET not in data.columns:
        raise ValueError(f"{TARGET} not found in {DATA_PATH.name}.")
    data = data.dropna(subset=[TARGET])

    feature_cols = [
        col for col in data.columns
        if col != TARGET and col.lower() not in EXCLUDED_FEATURES
    ]
    if not feature_cols:
        raise ValueError("No features remain after exclusions.")

    excluded_cols = [
        col for col in data.columns
        if col.lower() in EXCLUDED_FEATURES and col != TARGET
    ]
    print(f"Loaded {len(data)} rows.")
    if excluded_cols:
        print(f"Excluded LP/rank columns: {', '.join(excluded_cols)}")

    X = data[feature_cols].copy()
    y = data[TARGET].astype(float)
    return X, y


def determine_feature_types(feature_frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical_features = feature_frame.select_dtypes(include=["object"]).columns.tolist()
    numeric_features = [col for col in feature_frame.columns if col not in categorical_features]
    return numeric_features, categorical_features


def build_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    transformers = []
    if numeric_features:
        transformers.append(
            ("num", SimpleImputer(strategy="median"), numeric_features)
        )
    if categorical_features:
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", categorical_transformer, categorical_features))
    if not transformers:
        raise ValueError("No columns available for preprocessing.")
    preprocessor = ColumnTransformer(transformers)

    regressor = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("regressor", regressor)])


def rmse_score(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def append_partial_result(row: dict, filepath: Path, header_written: bool) -> bool:
    ordered_row = {}
    for col in ("mean_cv_rmse", "std_cv_rmse", "mean_cv_r2"):
        if col in row:
            ordered_row[col] = row[col]
    for col in CLEAN_PARAM_NAMES:
        if col in row:
            ordered_row[col] = row[col]
    pd.DataFrame([ordered_row]).to_csv(
        filepath,
        mode="a",
        header=not header_written,
        index=False,
    )
    return True


def run_hyperparameter_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    results_path: Path,
) -> tuple[pd.DataFrame, dict]:
    results_path.unlink(missing_ok=True)
    header_written = False

    param_grid = list(ParameterGrid(PARAM_GRID))
    total_configs = len(param_grid)
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rmse_scorer = make_scorer(rmse_score, greater_is_better=False)

    results = []
    print(f"Evaluating {total_configs} hyperparameter combinations...")
    for idx, params in enumerate(param_grid, start=1):
        pipeline = build_pipeline(numeric_features, categorical_features)
        pipeline.set_params(**params)
        cv_scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=kfold,
            scoring={"rmse": rmse_scorer, "r2": "r2"},
            n_jobs=-1,
        )

        rmse_scores = -cv_scores["test_rmse"]
        row = {
            "mean_cv_rmse": rmse_scores.mean(),
            "std_cv_rmse": rmse_scores.std(),
            "mean_cv_r2": cv_scores["test_r2"].mean(),
            "params_prefixed": params,
        }
        clean_params = {k.replace("regressor__", ""): v for k, v in params.items()}
        row.update(clean_params)
        results.append(row)

        header_written = append_partial_result(row, results_path, header_written)

        print(
            f"[{idx:02d}/{total_configs}] RMSE: {rmse_scores.mean():.3f}, "
            f"params: {clean_params}"
        )

    results_df = pd.DataFrame(results).sort_values("mean_cv_rmse").reset_index(drop=True)
    best_params = results_df.iloc[0]["params_prefixed"]
    return results_df, best_params


def display_top_models(results_df: pd.DataFrame, top_n: int = TOP_K) -> None:
    cols_to_show = ["mean_cv_rmse", "std_cv_rmse", "mean_cv_r2"] + CLEAN_PARAM_NAMES
    cols_to_show = [col for col in cols_to_show if col in results_df.columns]
    print("\nTop models by cross-validated RMSE:")
    print(results_df.head(top_n)[cols_to_show].to_string(index=False))


def save_results(results_df: pd.DataFrame) -> None:
    cols = [col for col in results_df.columns if col != "params_prefixed"]
    results_df[cols].to_csv(RESULTS_PATH, index=False)


def train_best_model(
    best_params: dict,
    numeric_features: list[str],
    categorical_features: list[str],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[Pipeline, float, float]:
    pipeline = build_pipeline(numeric_features, categorical_features)
    pipeline.set_params(**best_params)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = r2_score(y_test, predictions)
    return pipeline, rmse, r2


def main() -> None:
    X, y = load_dataset()
    numeric_features, categorical_features = determine_feature_types(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    results_df, best_params = run_hyperparameter_search(
        X_train, y_train, numeric_features, categorical_features, RESULTS_PATH
    )
    display_top_models(results_df)
    save_results(results_df)

    _, test_rmse, test_r2 = train_best_model(
        best_params,
        numeric_features,
        categorical_features,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    print(f"\nHold-out test RMSE: {test_rmse:.3f}")
    print(f"Hold-out test R^2: {test_r2:.3f}")
    print(f"All results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()