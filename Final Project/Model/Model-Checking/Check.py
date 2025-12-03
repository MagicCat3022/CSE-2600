import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# 1. Point to the saved model file
# Adjust the path if your model is in a different folder
MODEL_PATH = Path(r"C:\Users\AHMET\Documents\GitHub\CSE-2600\Final Project\Model\CatBoost\all_models\catboost_reg_2.joblib")

# 2. Load the pipeline
pipeline: Pipeline = joblib.load(MODEL_PATH)

# 3. Load test data (adjust path to your test data)
TEST_DATA_PATH = Path(r"C:\Users\AHMET\Documents\GitHub\CSE-2600\Final Project\Model\CatBoost\test_data.csv")
test_df = pd.read_csv(TEST_DATA_PATH)

# Separate features and target
# Adjust 'target_column_name' to your actual target column
TARGET_COL = 'TOTAL_LP'  # Change this to your actual target column name
X_test = test_df.drop(columns=[TARGET_COL])
y_test = test_df[TARGET_COL]

# 4. Make predictions
y_pred = pipeline.predict(X_test)

# 5. Calculate regression metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("=" * 60)
print("REGRESSION METRICS")
print("=" * 60)
print(f"Mean Absolute Error (MAE):     {mae:.4f}")
print(f"Mean Squared Error (MSE):      {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score:                      {r2:.4f}")
print("=" * 60)

# 6. Extract the model from the pipeline
# In your regression code, the step was named 'regressor'
# In your classification code, the step was named 'model'
try:
    model = pipeline.named_steps['regressor']
except KeyError:
    model = pipeline.named_steps['model']

# 7. Retrieve feature names from the preprocessor
try:
    preprocessor = pipeline.named_steps['preprocessor']
except KeyError:
    preprocessor = pipeline.named_steps['preprocess']

# Get the list of feature names output by the preprocessor
feature_names = preprocessor.get_feature_names_out()

# 8. Transform test data to get feature matrix
X_test_transformed = preprocessor.transform(X_test)

# 9. Calculate feature statistics and p-values
feature_stats = []

for idx, feature in enumerate(feature_names):
    # Clean up ColumnTransformer prefixes
    clean_name = feature.split("__", 1)[1] if "__" in feature else feature
    
    # Get feature values
    if hasattr(X_test_transformed, 'toarray'):
        feature_values = X_test_transformed.toarray()[:, idx]
    else:
        feature_values = X_test_transformed[:, idx]
    
    # Calculate correlation with target
    correlation = np.corrcoef(feature_values, y_test)[0, 1]
    
    # Calculate p-value using linear regression t-test
    slope, intercept, r_value, p_value, std_err = stats.linregress(feature_values, y_test)
    
    # Calculate feature statistics
    feature_mean = np.mean(feature_values)
    feature_std = np.std(feature_values)
    feature_min = np.min(feature_values)
    feature_max = np.max(feature_values)
    
    feature_stats.append({
        'Feature': clean_name,
        'P-Value': p_value,
        'Correlation': correlation,
        'R-Value': r_value,
        'Std_Error': std_err,
        'Mean': feature_mean,
        'Std_Dev': feature_std,
        'Min': feature_min,
        'Max': feature_max
    })

stats_df = pd.DataFrame(feature_stats).sort_values(by='P-Value')

# 10. Get feature importance
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

# 11. Merge importance with statistics
combined_df = importance_df.merge(stats_df, on='Feature', how='left')

# 12. Display feature analysis
print("\n" + "=" * 60)
print("FEATURE ANALYSIS (Top 25 by Importance)")
print("=" * 60)
print(combined_df.head(25).to_string(index=False))

# 13. Display most significant features by p-value
print("\n" + "=" * 60)
print("MOST SIGNIFICANT FEATURES (Top 25 by P-Value)")
print("=" * 60)
print(stats_df.head(25).to_string(index=False))

# 14. Save results to CSV
combined_df.to_csv(RESULTS_DIR / 'feature_analysis.csv', index=False)
print("\n" + "=" * 60)
print("Full feature analysis saved to 'feature_analysis.csv'")
print("=" * 60)

# 15. Visualization: Feature Importance
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
plt.savefig(RESULTS_DIR / 'feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# 16. Visualization: P-Values
plt.figure(figsize=(10, 8))
top_sig = stats_df.head(top_n).sort_values(by='P-Value', ascending=True)
plt.barh(
    top_sig['Feature'][::-1],
    -np.log10(top_sig['P-Value'][::-1]),
    color='coral'
)
plt.xlabel('-log10(P-Value)')
plt.title(f'Top {top_n} Most Significant Features')
plt.axvline(x=-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'feature_pvalues.png', dpi=300, bbox_inches='tight')
plt.show()

# 17. Visualization: Actual vs Predicted
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'Actual vs Predicted (R² = {r2:.4f})')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.show()

# 18. Visualization: Residuals
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5, edgecolors='k', linewidth=0.5)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'residuals.png', dpi=300, bbox_inches='tight')
plt.show()