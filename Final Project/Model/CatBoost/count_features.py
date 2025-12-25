from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline


def unwrap_estimator(model):
    if isinstance(model, Pipeline):
        return model.steps[-1][1]
    return model


def infer_feature_count(model) -> int | None:
    attr_lengths = [
        ("n_features_in_", lambda m: getattr(m, "n_features_in_")),
        ("feature_names_in_", lambda m: len(getattr(m, "feature_names_in_", []))),
        ("feature_names_", lambda m: len(getattr(m, "feature_names_", []))),
        ("feature_importances_", lambda m: len(getattr(m, "feature_importances_", []))),
    ]
    for attr, getter in attr_lengths:
        if hasattr(model, attr):
            count = getter(model)
            if count:
                return int(count)
    return None


def main():
    parser = argparse.ArgumentParser(description="Show feature count for a saved CatBoost joblib model.")
    parser.add_argument("--model", required=True, help="Path to the .joblib model file.")
    args = parser.parse_args()

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found.")

    model = joblib.load(model_path)
    feature_count = infer_feature_count(model) or infer_feature_count(unwrap_estimator(model))
    if feature_count is None:
        raise RuntimeError("Unable to infer feature count from the provided model.")

    print(f"{model_path.name}: {feature_count} feature(s)")


if __name__ == "__main__":
    main()