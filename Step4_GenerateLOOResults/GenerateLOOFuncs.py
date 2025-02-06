import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

# ===========================
# 1) Example "Best Params" for XGBoost
# ===========================
best_params = {
    'FM':  {'subsample': 0.8, 'n_estimators': 500, 'max_depth': 3, 'learning_rate': 0.2, 'colsample_bytree': 0.6},
    'AM':  {'subsample': 0.8, 'n_estimators': 500, 'max_depth': 3, 'learning_rate': 0.2, 'colsample_bytree': 0.6},
    'AP':  {'subsample': 1.0, 'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.01, 'colsample_bytree': 1.0},
    'FP':  {'subsample': 1.0, 'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.01, 'colsample_bytree': 1.0},
    'FMP': {'subsample': 0.6, 'n_estimators': 500, 'max_depth': 3, 'learning_rate': 0.2, 'colsample_bytree': 0.6},
    'AMP': {'subsample': 0.8, 'n_estimators': 200, 'max_depth': 9, 'learning_rate': 0.05, 'colsample_bytree': 0.8}
}

def add_median_lifespan_increase(data):
    """
    Maps certain groups (M_Rapa, M_Cana, etc.) to fixed 'median_lifespan_increase' values.
    """
    increase_map = {
        "M_Rapa": 23, "M_Cana": 14, "M_17aE2": 12, "M_CR": 32,
        "M_Aca": 22, "M_Cont_12": 0
    }
    data.loc[:, 'median_lifespan_increase'] = data['Grp_Sex'].map(increase_map)
    return data

def prepare_dataset(dataset, dataset_name):
    """
    Basic dataset preparation:
      - Ensures 'Grp_Sex' exists, or tries 'Grp'
      - Filters data for male mice (Grp_Sex starts with 'M_')
    """
    print(f"\nPreparing dataset: {dataset_name}")
    all_columns = dataset.columns.tolist()
    print(f"First 10 columns: {all_columns[:10]}")
    print(f"Last 5 columns: {all_columns[-5:]}")
    print(f"Total number of columns: {len(all_columns)}")
    print(f"Original shape: {dataset.shape}")
    
    if 'Grp_Sex' not in dataset.columns:
        if 'Grp' in dataset.columns:
            dataset['Grp_Sex'] = dataset['Grp']
        else:
            raise ValueError(f"Cannot find 'Grp_Sex' or 'Grp' column in the dataset {dataset_name}")
    
    # Filter for male mice only (Grp_Sex starts with 'M_')
    dataset = dataset[dataset['Grp_Sex'].str.startswith('M_', na=False)].copy()
    
    print(f"Unique Grp_Sex values: {dataset['Grp_Sex'].unique().tolist()}")
    print(f"Shape after filtering: {dataset.shape}")
    
    return dataset

def perform_loo_regression_analysis(dataset, dataset_name, test_groups, control_values_dataset):
    """
    Perform a leave-one-group-out regression analysis using XGBoost, 
    incorporating custom best_params per dataset if available.
    - Also ensures each row has a unique Mouse_ID (like in your first code).
    """
    np.random.seed(123)
    
    # 1) Basic dataset prep
    dataset = prepare_dataset(dataset, dataset_name)
    dataset = add_median_lifespan_increase(dataset)

    # ---------------------
    #  CREATE A Mouse_ID (like in your first code snippet)
    # ---------------------
    if 'Mouse_ID' not in dataset.columns:
        dataset['Mouse_ID'] = [f"{dataset_name}_{idx}" for idx in range(len(dataset))]
    print(f"After adding Mouse_ID. Shape: {dataset.shape}")
    
    # 2) Possibly override M_Cont_12 with pre-calculated control median
    control_median = control_values_dataset[
        (control_values_dataset['Dataset'] == dataset_name) & 
        (control_values_dataset['Grp_Sex'] == "M_Cont_12")
    ]['Prediction'].median()
    
    if pd.notna(control_median):
        dataset.loc[dataset['Grp_Sex'] == "M_Cont_12", 'median_lifespan_increase'] = control_median
        print(f"Updated M_Cont_12 values for {dataset_name} to {control_median}")
    else:
        print(f"No pre-calculated control value found for {dataset_name}. Using original values.")
    
    # 3) Define predictors (exclude certain columns)
    predictor_vars = [
        col for col in dataset.columns 
        if col not in [
            'median_lifespan_increase', 'Grp_Sex', 'Mouse', 'Sex', 'Grp', 
            'Lifespan_Increased2', 'median_control_value', 'X18198', '18198', 'Unnamed: 0'
        ]
    ]
    
    # 4) Encode non-numeric columns
    le = LabelEncoder()
    for col in predictor_vars:
        if dataset[col].dtype == 'object':
            dataset[col] = le.fit_transform(dataset[col].astype(str))
    
    # 5) Keep only relevant test groups
    dataset = dataset[dataset['Grp_Sex'].isin(test_groups)]
    dataset = dataset.dropna(subset=predictor_vars + ['median_lifespan_increase'])
    
    print(f"\nDataset {dataset_name} after preparation:")
    print(f"Shape: {dataset.shape}")
    print(f"Columns: {dataset.columns.tolist()[:10]} ... {dataset.columns.tolist()[-5:]}")
    print(f"Unique Grp_Sex values: {dataset['Grp_Sex'].unique().tolist()}")
    print(f"First few rows of prepared dataset (first 10 columns):")
    print(dataset.iloc[:, :10].head())
    
    # 6) Prepare to store results
    results = []
    all_predictions = []
    
    # ======================
    # 7) XGBoost Param Logic
    # ======================
    default_xgb_params = {
        'n_estimators': 500,
        'random_state': 123,
        'n_jobs': -1
    }
    
    # Merge best_params if this dataset is found
    if dataset_name in best_params:
        model_params = default_xgb_params.copy()
        model_params.update(best_params[dataset_name])
    else:
        model_params = default_xgb_params
    
    # 8) Leave-One-Group-Out loop
    for test_group in test_groups:
        try:
            testing_data = dataset[dataset['Grp_Sex'] == test_group]
            training_data = dataset[dataset['Grp_Sex'] != test_group]
            
            if not testing_data.empty and not training_data.empty:
                xgb_model = XGBRegressor(**model_params)
                xgb_model.fit(training_data[predictor_vars], training_data['median_lifespan_increase'])
                
                predictions = xgb_model.predict(testing_data[predictor_vars])
                
                # Keep track of Mouse_ID from the testing_data
                all_predictions.append(pd.DataFrame({
                    'Dataset': dataset_name,
                    'Grp_Sex': test_group,
                    'Mouse_ID': testing_data['Mouse_ID'].values,
                    'Prediction': predictions
                }))
                
                aggregate_median_lifespan_extension = np.median(predictions)
                median_control_value = dataset.loc[dataset['Grp_Sex'] == "M_Cont_12", 'median_lifespan_increase'].median()
                tx_control = aggregate_median_lifespan_extension - median_control_value
                
                results.append({
                    'Dataset': dataset_name,
                    'Test_Group': test_group,
                    'Whole_Training_Set': ', '.join([g for g in test_groups if g != test_group]),
                    'Tx_Control': tx_control,
                    'Median_Prediction': aggregate_median_lifespan_extension,
                    'Median_Control_Value': median_control_value
                })
            else:
                print(f"Warning: Not enough data for test group {test_group} in {dataset_name}")
        except Exception as e:
            print(f"Error processing test group {test_group} in {dataset_name}: {str(e)}")
            print(f"Data types of first 10 predictor variables:")
            print(testing_data[predictor_vars[:10]].dtypes)
            print(f"First few rows of testing data (first 10 columns):")
            print(testing_data[predictor_vars[:10]].head())
    
    # 9) Return combined results
    results_df = pd.DataFrame(results)
    preds_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    return results_df, preds_df

def process_all_datasets(datasets, test_groups, control_values_dataset):
    """
    Loop over each dataset, call perform_loo_regression_analysis with XGBoost,
    and save the combined results to Excel.
    """
    all_results = []
    all_predictions = pd.DataFrame()
    
    for dataset_name, dataset in datasets.items():
        print(f"\n\nProcessing dataset: {dataset_name}")
        try:
            results, predictions = perform_loo_regression_analysis(
                dataset=dataset,
                dataset_name=dataset_name,
                test_groups=test_groups,
                control_values_dataset=control_values_dataset
            )
            if not results.empty:
                all_results.append(results)
            if not predictions.empty:
                all_predictions = pd.concat([all_predictions, predictions], ignore_index=True)
        except Exception as e:
            print(f"Error processing dataset: {dataset_name}")
            print(f"Error message: {str(e)}")
    
    # Combine all results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        
        if not combined_results.empty:
            combined_results.to_excel("Combined_Results_LOO_analysis.xlsx", index=False)
            print("LOO analysis results saved to 'Combined_Results_LOO_analysis.xlsx'")
        else:
            print("No results to save in the combined file.")
        
        if not all_predictions.empty:
            all_predictions.to_excel("all_loo_predictions_20_iters.xlsx", index=False)
            print("All LOO predictions saved to 'all_loo_predictions_20_iters.xlsx'")
        else:
            print("No predictions to save.")
        
        return combined_results
    else:
        print("No results to process.")
        return pd.DataFrame()


