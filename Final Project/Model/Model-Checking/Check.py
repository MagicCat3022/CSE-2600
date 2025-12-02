import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.pipeline import Pipeline

# 1. Point to the saved model file
# Adjust the path if your model is in a different folder
MODEL_PATH = Path(r"C:\Users\AHMET\Documents\GitHub\CSE-2600\Final Project\Model\CatBoost\all_models\catboost_reg_2.joblib")

# 2. Load the pipeline
pipeline: Pipeline = joblib.load(MODEL_PATH)

# 3. Extract the model from the pipeline
# In your regression code, the step was named 'regressor'
# In your classification code, the step was named 'model'
try:
    model = pipeline.named_steps['regressor']
except KeyError:
    model = pipeline.named_steps['model']

# 4. Retrieve feature names from the preprocessor
try:
    preprocessor = pipeline.named_steps['preprocessor']
except KeyError:
    preprocessor = pipeline.named_steps['preprocess']

# Get the list of feature names output by the preprocessor
feature_names = preprocessor.get_feature_names_out()

# 5. Get feature importance based on model type
if isinstance(model, (XGBClassifier, XGBRegressor)):
    # XGBoost feature importance
    importance = model.get_booster().get_score(importance_type='weight')
    
    # Map the "f0", "f1" keys to actual names
    mapped_importance = {}
    for fid, score in importance.items():
        try:
            # fid is string "f0", "f1", etc. -> convert to int index
            idx = int(fid[1:])
            name = feature_names[idx]
            
            # Clean up ColumnTransformer prefixes (e.g., "num__gold" -> "gold")
            if "__" in name:
                name = name.split("__", 1)[1]
                
            mapped_importance[name] = score
        except (ValueError, IndexError):
            # Fallback if parsing fails
            mapped_importance[fid] = score

elif isinstance(model, (CatBoostClassifier, CatBoostRegressor)):
    # CatBoost feature importance
    importance_values = model.get_feature_importance()
    
    # Map importance values to feature names
    mapped_importance = {}
    for idx, score in enumerate(importance_values):
        if idx < len(feature_names):
            name = feature_names[idx]
            
            # Clean up ColumnTransformer prefixes
            if "__" in name:
                name = name.split("__", 1)[1]
                
            mapped_importance[name] = score
else:
    raise ValueError(f"Unsupported model type: {type(model)}")

importance_df = pd.DataFrame({
    'Feature': list(mapped_importance.keys()),
    'Importance': list(mapped_importance.values())
}).sort_values(by='Importance', ascending=False)

top_n = 25
plt.figure(figsize=(10, 8))
plt.barh(
    importance_df['Feature'].head(top_n)[::-1],
    importance_df['Importance'].head(top_n)[::-1],
    color='skyblue'
)
plt.xlabel('Importance Score')
plt.title(f'Top {top_n} Feature Importance')
plt.tight_layout()
plt.show()