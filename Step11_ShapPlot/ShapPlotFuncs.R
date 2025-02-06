# Function to add the 'median_lifespan_increase' column based on 'Grp_Sex'
add_median_lifespan_increase <- function(data) {
  data$median_lifespan_increase <- NA  # Initialize with NA
  
  # Apply conditions
  data$median_lifespan_increase[data$Grp_Sex == "M_Rapa"] <- 23
  data$median_lifespan_increase[data$Grp_Sex == "M_Cana"] <- 14
  data$median_lifespan_increase[data$Grp_Sex == "M_17aE2"] <- 12
  data$median_lifespan_increase[data$Grp_Sex == "M_CR"] <- 32
  data$median_lifespan_increase[data$Grp_Sex == "M_Aca"] <- 22
  data$median_lifespan_increase[data$Grp_Sex == "M_Cont_12"] <- 0
  
  return(data)
}

# Function to perform group analysis with Gini importance
# Load SHAP feature importance data and select top metabolites
perform_xgboost_shap_analysis <- function(dataset, dataset_name, selected_groups, shap_file, top_n = 20) {
  # Load SHAP importance values
  shap_importance <- read.csv(shap_file) %>%
    head(top_n) %>%  # Select top N metabolites
    rename(Metabolite = Feature) %>%
    mutate(Metabolite = make.names(Metabolite)) %>%  # Standardize names
    mutate(Rank = row_number())  # Rank metabolites
  
  # Ensure SHAP feature names match dataset column names
  dataset_features <- colnames(dataset)
  shap_importance <- shap_importance %>%
    filter(Metabolite %in% dataset_features)  # Only keep valid features
  
  print(paste("Loaded SHAP importance from:", shap_file))
  print("Top metabolites based on SHAP (after standardizing names):")
  print(shap_importance)
  
  return(shap_importance)
}




# Function to analyze metabolites
analyze_metabolites_shap <- function(dataset, shap_importance, 
                                     control_group = "M_Cont_12",
                                     treatment_groups = c("M_Cana", "M_17aE2", "M_Aca", "M_CR", "M_Rapa")) {
  
  # Undo log2 transformation
  dataset <- dataset %>%
    mutate(across(where(is.numeric), ~ 2^.))
  
  # Standardize SHAP metabolite names to match dataset column names
  shap_importance$Metabolite <- make.names(shap_importance$Metabolite)
  dataset_features <- colnames(dataset)
  
  # Filter only matching columns
  top_metabolites <- shap_importance %>%
    filter(Metabolite %in% dataset_features) %>%
    pull(Metabolite)
  
  # Filter dataset for control and treatment groups
  top_features <- dataset %>%
    filter(Grp_Sex %in% c(control_group, treatment_groups)) %>%
    select(Grp_Sex, all_of(top_metabolites))
  
  # Compute mean values for control group
  control_means <- top_features %>%
    filter(Grp_Sex == control_group) %>%
    summarise(across(all_of(top_metabolites), mean, na.rm = TRUE))
  
  # Compute fold changes relative to control
  fold_changes <- top_features %>%
    group_by(Grp_Sex) %>%
    summarise(across(all_of(top_metabolites), mean, na.rm = TRUE)) %>%
    ungroup() %>%
    mutate(across(all_of(top_metabolites), ~ . / control_means[[cur_column()]]))
  
  # Convert to long format
  fold_changes_long <- fold_changes %>%
    pivot_longer(cols = -Grp_Sex, names_to = "Metabolite", values_to = "FoldChange") %>%
    filter(Grp_Sex != control_group)
  
  # Ensure Grp_Sex is a factor with levels in the correct order
  fold_changes_long$Grp_Sex <- factor(fold_changes_long$Grp_Sex, levels = treatment_groups)
  
  print("Log2 Fold Change Calculated for SHAP metabolites:")
  print(colnames(fold_changes_long))
  
  return(list(fold_changes = fold_changes_long, top_metabolites = top_metabolites, control_group = control_group))
}
