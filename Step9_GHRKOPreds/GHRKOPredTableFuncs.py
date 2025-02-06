import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

# ---------------------
# Best Params Dictionary
# ---------------------
best_params = {
    'FM':  {'subsample': 0.8, 'n_estimators': 500, 'max_depth': 3, 'learning_rate': 0.2, 'colsample_bytree': 0.6},
    'AM':  {'subsample': 0.8, 'n_estimators': 500, 'max_depth': 3, 'learning_rate': 0.2, 'colsample_bytree': 0.6},
    'AP':  {'subsample': 1.0, 'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.01, 'colsample_bytree': 1.0},
    'FP':  {'subsample': 1.0, 'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.01, 'colsample_bytree': 1.0},
    'FMP': {'subsample': 0.6, 'n_estimators': 500, 'max_depth': 3, 'learning_rate': 0.2, 'colsample_bytree': 0.6},
    'AMP': {'subsample': 0.8, 'n_estimators': 200, 'max_depth': 9, 'learning_rate': 0.05, 'colsample_bytree': 0.8}
}

# ---------------------
# 1) Define helpers
# ---------------------
def sanitize_column_name(col_name):
    """
    Convert to string, then replace disallowed characters.
    XGBoost does not allow '[', ']', '<', '>' in feature names.
    We also replace parentheses, braces, spaces, or hyphens with underscores.
    """
    col_name = str(col_name)  # ensure string type
    bad_chars = ['[', ']', '<', '>', '(', ')', '{', '}', ' ', '-']
    for bad_char in bad_chars:
        col_name = col_name.replace(bad_char, '_')
    return col_name

def sanitize_columns(df):
    """
    Apply sanitize_column_name to every column in the DataFrame.
    """
    new_cols = {}
    for c in df.columns:
        new_cols[c] = sanitize_column_name(c)
    df.rename(columns=new_cols, inplace=True)
    return df

def add_median_lifespan_increase(data):
    """
    Map group labels to the median_lifespan_increase target column.
    """
    increase_map = {
        "M_Rapa": 23, 
        "M_Cana": 14, 
        "M_17aE2": 12, 
        "M_CR": 32,
        "M_Aca": 22, 
        "M_Cont_12": 0
    }
    data.loc[:, 'median_lifespan_increase'] = data['Grp_Sex'].map(increase_map)
    return data

def prepare_dataset(dataset, dataset_name):
    """
    Sanitize columns, log some info, and return the prepared DataFrame.
    """
    print(f"\nPreparing dataset: {dataset_name}")
    dataset = sanitize_columns(dataset)
    print(f"Unique Grp_Sex values: {dataset['Grp_Sex'].unique().tolist()}")
    print(f"Shape after preparation: {dataset.shape}")
    return dataset

# ---------------------
# 2) Main training/prediction function with multiple iterations
# ---------------------
def train_and_predict_on_full_data(
    dataset, 
    dataset_name, 
    train_groups, 
    predict_groups, 
    n_iterations=100
):
    """
    For each dataset:
      - Prepare and sanitize columns
      - Add the median_lifespan_increase (our target)
      - Create a unique Mouse ID
      - Split into train/test by groups
      - Label-encode string columns
      - Train and predict with XGB n_iterations times
      - Collect predictions for each iteration
      - Return a DataFrame with the *median* prediction per Mouse
    """
    
    # Prepare and add target
    dataset = prepare_dataset(dataset, dataset_name)
    dataset = add_median_lifespan_increase(dataset)
    
    # Create a unique "Mouse" ID if not present (or if ignoring any existing one)
    # We'll combine the dataset name and the row index to make it guaranteed unique
    dataset["Mouse"] = [f"{dataset_name}_{idx}" for idx in range(len(dataset))]
    
    # Split train/test
    train_data = dataset[dataset['Grp_Sex'].isin(train_groups)].copy()
    test_data  = dataset[dataset['Grp_Sex'].isin(predict_groups)].copy()
    
    # Drop rows in train_data that do not have the target
    train_data = train_data.dropna(subset=['median_lifespan_increase'])
    
    # Print row counts before cleaning
    for group in predict_groups:
        count_before_cleaning = len(test_data[test_data['Grp_Sex'] == group])
        print(f"Rows for {group} before cleaning: {count_before_cleaning}")
        
    # Define predictor variables: exclude columns you do NOT want as features
    # (We exclude "Mouse" so it is never used as a feature)
    excluded_cols = [
        'median_lifespan_increase', 'Grp_Sex', 'Sex', 'Grp',
        'Lifespan_Increased2', 'median_control_value', 'X18198', '18198',
        'Treatment','Condition','ID','group','sex',
        'Mouse'  # we created our own Mouse column, so exclude it
    ]
    predictor_vars = [col for col in dataset.columns if col not in excluded_cols]
    
    # Drop rows in test_data with NaN in predictor columns
    test_data_cleaned = test_data.dropna(subset=predictor_vars)
    
    # Print row counts after cleaning
    for group in predict_groups:
        count_after_cleaning = len(test_data_cleaned[test_data_cleaned['Grp_Sex'] == group])
        print(f"Rows for {group} after cleaning: {count_after_cleaning}")
    
    # Label-encode any object columns in both train and test
    for col in predictor_vars:
        if (train_data[col].dtype == 'object') or (test_data_cleaned[col].dtype == 'object'):
            le = LabelEncoder()
            combined_data = pd.concat([train_data[col], test_data_cleaned[col]]).astype(str)
            le.fit(combined_data)
            train_data[col]        = le.transform(train_data[col].astype(str))
            test_data_cleaned[col] = le.transform(test_data_cleaned[col].astype(str))

    # ---------------------
    # 3) Repeated training and prediction
    # ---------------------
    all_iterations_predictions = []

    for i in range(n_iterations):
        # Optionally vary the random seed per iteration for randomization
        # If you'd prefer consistent results, remove the "+ i"
        np.random.seed(123 + i)
        
        # Set up model parameters: start with default parameters
        params = {
            'n_jobs': -1,
            'random_state': 123 + i
        }
        default_model_params = {
            'n_estimators': 500,
            'max_depth': 3,
            'learning_rate': 0.2,
            'colsample_bytree': 0.6,
            'subsample': 0.8
        }
        params.update(default_model_params)
        
        # If specific best_params exist for this dataset, update the parameters
        if dataset_name in best_params:
            params.update(best_params[dataset_name])
        
        # Initialize the XGB model with the determined parameters
        xgb_model = XGBRegressor(**params)
        
        # Train
        xgb_model.fit(train_data[predictor_vars], train_data['median_lifespan_increase'])
        
        # Predict for each group in the test set
        for group in predict_groups:
            group_data = test_data_cleaned[test_data_cleaned['Grp_Sex'] == group]
            if not group_data.empty:
                preds = xgb_model.predict(group_data[predictor_vars])
                # Keep track of predictions along with the Mouse ID
                iteration_df = pd.DataFrame({
                    'Dataset': dataset_name,
                    'Whole_Training_Set': [', '.join(train_groups)] * len(group_data),
                    'Test_Group': [group] * len(group_data),
                    'Mouse': group_data['Mouse'].values,  # use our newly made Mouse IDs
                    'Prediction': preds,
                    'Iteration': i
                })
                all_iterations_predictions.append(iteration_df)
            else:
                print(f"Warning: No test data for '{group}' after cleaning.")
    
    # Combine all iterations
    if len(all_iterations_predictions) == 0:
        # No predictions
        return pd.DataFrame()
    else:
        predictions_df = pd.concat(all_iterations_predictions, ignore_index=True)
        
        # ---------------------
        # 4) Compute median per Mouse across all iterations
        # ---------------------
        # Group by the dataset, training set, test group, and mouse
        # and compute the median across the "Prediction" column.
        median_predictions = (
            predictions_df
            .groupby(["Dataset", "Whole_Training_Set", "Test_Group", "Mouse"], as_index=False)
            ["Prediction"]
            .median()
        )
        
        # This is your final single-row-per-mouse dataset
        return median_predictions
