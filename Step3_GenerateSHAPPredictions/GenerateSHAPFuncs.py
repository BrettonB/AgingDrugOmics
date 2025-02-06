import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from scipy import stats
import xgboost
import warnings
import os
import shap
import matplotlib.pyplot as plt
import math

# ============================
# 1) Best Params Dictionary
# ============================
best_params = {
    'FM':  {'subsample': 0.8, 'n_estimators': 500, 'max_depth': 3, 'learning_rate': 0.2, 'colsample_bytree': 0.6},
    'AM':  {'subsample': 0.8, 'n_estimators': 500, 'max_depth': 3, 'learning_rate': 0.2, 'colsample_bytree': 0.6},
    'AP':  {'subsample': 1.0, 'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.01, 'colsample_bytree': 1.0},
    'FP':  {'subsample': 1.0, 'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.01, 'colsample_bytree': 1.0},
    'FMP': {'subsample': 0.6, 'n_estimators': 500, 'max_depth': 3, 'learning_rate': 0.2, 'colsample_bytree': 0.6},
    'AMP': {'subsample': 0.8, 'n_estimators': 200, 'max_depth': 9, 'learning_rate': 0.05, 'colsample_bytree': 0.8}
}

def add_median_lifespan_increase(data):
    data = data.copy()
    data['median_lifespan_increase'] = np.nan  # Initialize with NaN

    # Define mapping from 'Grp_Sex' to 'median_lifespan_increase'
    mapping = {
        "M_Rapa": 23,
        "M_Cana": 14,
        "M_17aE2": 12,
        "M_CR": 32,
        "M_Aca": 22,
        "M_Cont_12": 0
    }

    # Apply mapping
    data['median_lifespan_increase'] = data['Grp_Sex'].map(mapping)
    return data

def train_and_compute_shap(
    dataset, 
    dataset_name, 
    selected_groups, 
    output_dir='./', 
    additional_exclude_cols=None
):
    """
    Train an XGBoost model with best hyperparameters (if available) 
    for a given dataset and compute SHAP values.
    """
    # ===================
    # 1) Prepare Dataset
    # ===================
    dataset = add_median_lifespan_increase(dataset)
    
    # Filter dataset for selected groups
    dataset = dataset[dataset['Grp_Sex'].isin(selected_groups)].reset_index(drop=True)
    
    # Default columns to exclude
    default_exclude_cols = [
        'median_lifespan_increase', 'Grp_Sex', 'Lifespan_Increased2', 'Grp', 'Mouse',
        'ID', 'group', 'Treatment', 'X18198', 'Condition', 'Sex', 'Mouse_ID', 'Unnamed: 0'
    ]
    
    # Combine default exclude columns with additional columns to exclude
    if additional_exclude_cols is not None:
        exclude_cols = default_exclude_cols + additional_exclude_cols
    else:
        exclude_cols = default_exclude_cols
    
    # Get predictor variables
    predictor_vars = [col for col in dataset.columns if col not in exclude_cols]
    
    # Convert predictor variables to numeric and remove any with NA/NaN/Inf
    dataset_numeric = dataset[predictor_vars].apply(pd.to_numeric, errors='coerce')
    valid_cols = dataset_numeric.columns[~dataset_numeric.isin([np.nan, np.inf, -np.inf]).any()]
    dataset_numeric = dataset_numeric[valid_cols]
    
    # Remove rows with missing values
    dataset_numeric = dataset_numeric.dropna()
    dataset = dataset.loc[dataset_numeric.index].reset_index(drop=True)
    dataset_numeric = dataset_numeric.reset_index(drop=True)
    
    print(f"Number of samples in {dataset_name}: {dataset.shape[0]}")
    print(f"Number of valid predictor variables: {dataset_numeric.shape[1]}")
    print(f"Groups in dataset: {', '.join(dataset['Grp_Sex'].unique())}")
    print("Unique values in median_lifespan_increase:")
    print(dataset['median_lifespan_increase'].unique())
    
    # Check if we have enough samples and variation in the response
    if dataset.shape[0] < 10 or dataset['median_lifespan_increase'].nunique() < 2:
        print(f"Warning: Not enough samples or variation in {dataset_name}")
        return None
    
    # =========================
    # 2) Select Model Params
    # =========================
    # We have a dictionary of best hyperparams for each dataset
    # We'll use them if present; otherwise fallback to a basic default
    default_params = {
        'n_estimators': 500,
        'random_state': 42,
        'n_jobs': -1  # Let XGBoost use all cores internally
    }
    
    # If the dataset name is in our best_params dict, override
    model_params = default_params.copy()
    if dataset_name in best_params:
        # Merge in the best hyperparams
        model_params.update(best_params[dataset_name])
    else:
        print(f"No specific hyperparams found for {dataset_name}; using default.")
    
    # =========================
    # 3) Fit XGBoost Model
    # =========================
    xgb_model = XGBRegressor(**model_params)
    xgb_model.fit(dataset_numeric, dataset['median_lifespan_increase'])
    
    # Make predictions
    predictions = xgb_model.predict(dataset_numeric)
    
    # Store predictions
    results = pd.DataFrame({
        "Dataset": dataset_name,
        "Grp_Sex": dataset['Grp_Sex'],
        "Prediction": predictions,
        "Actual": dataset['median_lifespan_increase']
    })
    
    # =========================
    # 4) Compute SHAP Values
    # =========================
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(dataset_numeric)
    
    # Compute mean SHAP values for each feature
    mean_shap_values = shap_values.mean(axis=0)
    shap_feature_importance = pd.DataFrame({
        'Feature': dataset_numeric.columns,
        'MeanSHAP': mean_shap_values
    }).sort_values(by='MeanSHAP', key=np.abs, ascending=False)
    
    # Save SHAP feature importance
    shap_feature_importance.to_csv(
        os.path.join(output_dir, f"SHAP_Feature_Importance_{dataset_name}.csv"),
        index=False
    )
    
    # Plot SHAP Summary
    plt.figure()
    shap.summary_plot(
        shap_values,
        dataset_numeric,
        feature_names=dataset_numeric.columns,
        show=False
    )
    plt.title(f"SHAP Summary Plot for {dataset_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"SHAP_Summary_{dataset_name}.png"))
    plt.close()
    
    # Save the predictions
    results.to_csv(os.path.join(output_dir, f"Predictions_{dataset_name}.csv"), index=False)
    
    return results
