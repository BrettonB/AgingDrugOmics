import os
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from scipy import stats
import re

def add_median_lifespan_increase(data):
    if 'Grp_Sex' not in data.columns:
        raise ValueError("Column 'Grp_Sex' not found in the dataset")
    
    lifespan_map = {
        'M_Rapa': 23,
        'M_Cana': 14,
        'M_17aE2': 12,
        'M_CR': 32,
        'M_Aca': 22,
        'M_Cont_12': 0
    }
    
    data['median_lifespan_increase'] = data['Grp_Sex'].map(lifespan_map)
    return data

def clean_column_name(name):
    # Remove any non-alphanumeric characters (except underscore) and ensure it starts with a letter
    cleaned = re.sub(r'\W+', '_', name)
    if cleaned and cleaned[0].isdigit():
        cleaned = 'f_' + cleaned
    return cleaned

def calculate_p_value(group, all_predictions, control_predictions):
    # Skip p-value calculation for control group itself
    if group == 'M_Cont_12':
        return np.nan
    group_predictions = all_predictions[all_predictions['Grp_Sex'] == group]['Prediction']
    _, p_value = stats.ttest_ind(group_predictions, control_predictions)
    return p_value if p_value >= 1e-300 else '< 1e-300'


def perform_xgboost_analysis_kfold(
    datasets, 
    selected_groups, 
    k=10, 
    output_prefix="XGBoost_Analysis_KFold", 
    cols_to_ignore=None,
    hyperparam_dict=None,
    iters=10,
    save_intermediate=False,      # <--- NEW
    intermediate_dir="IntermediatePreds"  # <--- NEW
):
    """
    Perform a repeated k-fold XGBoost analysis, optionally saving partial results
    to disk after each iteration to reduce memory usage.
    """
    print("Starting XGBoost analysis with k-fold cross-validation (XGBoost's own parallelism)")
    start_time = time.time()
    
    # Make sure intermediate directory exists if we are saving partial files
    if save_intermediate:
        os.makedirs(intermediate_dir, exist_ok=True)
    
    # Default columns to ignore if none provided
    if cols_to_ignore is None:
        cols_to_ignore = [
            'median_lifespan_increase', 'Grp_Sex', 'Lifespan_Increased2', 'Grp', 
            'Mouse', 'ID', 'group', 'Treatment', 'X18198', 'Condition', 'Sex', 
            'Mouse_ID'
        ]
    
    def perform_kfold_analysis(dataset, dataset_name, selected_groups, k, iters, hyperparam_dict):
        print(f"\nStarting analysis for dataset: {dataset_name}")
        dataset = add_median_lifespan_increase(dataset)
        print(f"After adding median_lifespan_increase. Shape: {dataset.shape}")
        
        # Ensure unique Mouse_ID if missing
        if 'Mouse_ID' not in dataset.columns:
            dataset['Mouse_ID'] = dataset['Grp_Sex'] + '_' + dataset.index.astype(str)
        print(f"After adding Mouse_ID. Shape: {dataset.shape}")
        
        # Filter by desired groups
        dataset = dataset[dataset['Grp_Sex'].isin(selected_groups)]
        print(f"After filtering for selected groups. Shape: {dataset.shape}")
        
        # Select numeric predictor columns only
        predictor_vars = [col for col in dataset.columns if col not in cols_to_ignore]
        dataset_numeric = dataset[predictor_vars].select_dtypes(include=[np.number])
        print(f"After selecting numeric predictors. Shape: {dataset_numeric.shape}")
        
        # Clean column names
        dataset_numeric.columns = [clean_column_name(col) for col in dataset_numeric.columns]
        print("Column names cleaned")
        
        # Remove columns with NaN or Inf
        valid_cols = dataset_numeric.columns[
            ~dataset_numeric.isna().any() & 
            ~dataset_numeric.isin([np.inf, -np.inf]).any()
        ]
        dataset_numeric = dataset_numeric[valid_cols]
        print(f"After removing columns with NaN or Inf. Shape: {dataset_numeric.shape}")
        
        # Check data sufficiency
        if len(dataset) < 10 or len(dataset['median_lifespan_increase'].unique()) < 2:
            print(f"Warning: Not enough samples or variation in {dataset_name}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        # Retrieve hyperparams for this dataset (if any)
        default_params = {
            'n_estimators': 500,
            'max_depth': 3,
            'learning_rate': 0.1,
            'colsample_bytree': 1.0,
            'subsample': 1.0,
            # Let XGBoost use all CPU cores
            'n_jobs': -1,
            'random_state': 42
        }
        if hyperparam_dict and dataset_name in hyperparam_dict:
            model_params = default_params.copy()
            model_params.update(hyperparam_dict[dataset_name])
        else:
            model_params = default_params
        
        # We'll aggregate predictions from each iteration/fold
        # BUT if save_intermediate is True, we won't keep them all in memory
        all_predictions_dfs = []
        
        for trialiter in range(iters):
            kf = KFold(n_splits=k, shuffle=True, random_state=trialiter)
            
            # We'll accumulate predictions for this iteration only
            iteration_predictions = []
            
            for i, (train_index, test_index) in enumerate(kf.split(dataset_numeric)):
                print(f"Processing fold {i+1} of {k} (Iteration {trialiter})")

                X_train, X_test = dataset_numeric.iloc[train_index], dataset_numeric.iloc[test_index]
                y_train = dataset.iloc[train_index]['median_lifespan_increase']
                y_test = dataset.iloc[test_index]['median_lifespan_increase']

                print(f"  Training XGBoost model for fold {i+1}, iteration {trialiter}")
                model = XGBRegressor(**model_params)
                model.fit(X_train, y_train)

                print(f"  Making predictions for fold {i+1}, iteration {trialiter}")
                predictions = model.predict(X_test)

                fold_results = pd.DataFrame({
                    'Dataset': dataset_name,
                    'Grp_Sex': dataset.iloc[test_index]['Grp_Sex'],
                    'Mouse_ID': dataset.iloc[test_index]['Mouse_ID'],
                    'Prediction': predictions,
                    'Actual': y_test.values,
                    'Fold': i+1,
                    'Iteration': trialiter
                })

                iteration_predictions.append(fold_results)
            
            # Combine this iteration's predictions
            iteration_predictions = pd.concat(iteration_predictions, ignore_index=True)
            
            if save_intermediate:
                # Save partial results for this dataset & iteration to disk
                # e.g., "IntermediatePreds/FM_iteration_3.csv"
                out_file = os.path.join(intermediate_dir, f"{dataset_name}_iter_{trialiter}.csv")
                iteration_predictions.to_csv(out_file, index=False)
                
                # Instead of storing in memory, we can discard these
                print(f"Saved intermediate predictions to {out_file}")
                iteration_predictions = None  # free memory
            else:
                # Keep in memory if not saving partially
                all_predictions_dfs.append(iteration_predictions)
        
        # If we saved partial data, we'll now re-read them at the end
        if save_intermediate:
            print(f"Re-reading partial predictions from {dataset_name} to combine everything.")
            combined_predictions = []
            for trialiter in range(iters):
                partial_file = os.path.join(intermediate_dir, f"{dataset_name}_iter_{trialiter}.csv")
                if os.path.exists(partial_file):
                    df_partial = pd.read_csv(partial_file)
                    combined_predictions.append(df_partial)
            if combined_predictions:
                all_predictions = pd.concat(combined_predictions, ignore_index=True)
            else:
                all_predictions = pd.DataFrame()
        else:
            # If we never wrote partial files, combine in-memory
            if all_predictions_dfs:
                all_predictions = pd.concat(all_predictions_dfs, ignore_index=True)
            else:
                all_predictions = pd.DataFrame()
        
        if all_predictions.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        print("Concatenating results from all folds/iterations for final stats.")
        
        # Summarize
        results = all_predictions.groupby(['Dataset', 'Grp_Sex', 'Iteration']).agg({
            'Prediction': ['median', 'std']
        }).reset_index()
        results.columns = ['Dataset', 'Grp_Sex', 'Iteration', 'Median_Prediction', 'Std_Dev_Prediction']
        
        # Calculate p-values vs control
        control_predictions = all_predictions[all_predictions['Grp_Sex'] == 'M_Cont_12']['Prediction']
        results['P_Value'] = results['Grp_Sex'].apply(lambda group: calculate_p_value(group, all_predictions, control_predictions))
        
        # Prepare control results
        control_results = all_predictions[all_predictions['Grp_Sex'] == 'M_Cont_12'].groupby(
            ['Dataset', 'Grp_Sex', 'Mouse_ID', 'Actual', 'Iteration']
        )['Prediction'].median().reset_index()
        
        print(f"Analysis completed for dataset: {dataset_name}")
        return results, control_results, all_predictions
    
    # --- main body ---
    all_results = []
    all_control_results = []
    all_mouse_predictions = []
    group_labels = {}
    
    for dataset_name, dataset in datasets.items():
        print(f"\nProcessing dataset: {dataset_name}")
        try:
            results, control_results, predictions = perform_kfold_analysis(
                dataset=dataset, 
                dataset_name=dataset_name, 
                selected_groups=selected_groups, 
                k=k, 
                iters=iters,
                hyperparam_dict=hyperparam_dict
            )
            
            if not results.empty:
                all_results.append(results)
                print(f"Results added for {dataset_name}")
            if not control_results.empty:
                all_control_results.append(control_results)
                print(f"Control results added for {dataset_name}")
            if not predictions.empty:
                all_mouse_predictions.append(predictions)
                print(f"All predictions added for {dataset_name}")
            
            # Optionally record group distribution
            group_labels[dataset_name] = dataset['Grp_Sex'].value_counts().to_dict()
            print(f"Group labels recorded for {dataset_name}")
        except Exception as e:
            print(f"Error processing dataset: {dataset_name}")
            print(f"Error message: {str(e)}")
    
    # Combine final results
    print("\nPreparing final consolidated results")
    if all_results:
        all_results = pd.concat(all_results, ignore_index=True)
        all_results.to_excel(f"{output_prefix}_All_Results.xlsx", index=False)
        print(f"All results saved to {output_prefix}_All_Results.xlsx")
    else:
        all_results = pd.DataFrame()
        print("No results to save for all_results.")
    
    if all_control_results:
        all_control_results = pd.concat(all_control_results, ignore_index=True)
        all_control_results.to_excel(f"{output_prefix}_Control_Results.xlsx", index=False)
        print(f"Control results saved to {output_prefix}_Control_Results.xlsx")
    else:
        all_control_results = pd.DataFrame()
        print("No results to save for control_results.")
    
    if all_mouse_predictions:
        all_mouse_predictions = pd.concat(all_mouse_predictions, ignore_index=True)
        all_mouse_predictions.to_excel(f"{output_prefix}_All_Predictions_All_Groups.xlsx", index=False)
        print(f"All mouse predictions saved to {output_prefix}_All_Predictions_All_Groups.xlsx")
    else:
        all_mouse_predictions = pd.DataFrame()
        print("No mouse predictions to save.")
    
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
    
    return {
        'all_results': all_results,
        'control_results': all_control_results,
        'group_labels': group_labels
    }
