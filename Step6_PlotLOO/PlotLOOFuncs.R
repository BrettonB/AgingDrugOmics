# --------------------------------------------------
# Load packages
# --------------------------------------------------
library(dplyr)
library(ggplot2)
library(openxlsx)
library(tidyr)
library(stats)


# --------------------------------------------------
# Corrected Function with Fixed y-axis for All Datasets
# --------------------------------------------------
process_and_visualize_loo_results <- function(all_loo_predictions_file,
                                              control_predictions_file,
                                              output_prefix = "LOO_Analysis") {
  
  # --------------------------------------------------
  # 1. Read in data
  # --------------------------------------------------
  # Load non-control LOO predictions 
  # (If your file only has the treatment groups, that's OK as long as the 'Dataset' names match the control file.)
  df_loo_raw <- read.xlsx(all_loo_predictions_file)
  
  # Load control data (M_Cont_12)
  df_controls_raw <- read.xlsx(control_predictions_file) %>%
    filter(Grp_Sex == "M_Cont_12")
  
  # --------------------------------------------------
  # 2. Summarize each file by Mouse_ID (each mouse is a single row = median of all iterations)
  # --------------------------------------------------
  df_loo <- df_loo_raw %>%
    group_by(Dataset, Grp_Sex, Mouse_ID) %>%
    summarise(Prediction = median(Prediction), .groups = "drop")
  
  df_controls <- df_controls_raw %>%
    group_by(Dataset, Grp_Sex, Mouse_ID) %>%
    summarise(Prediction = median(Prediction), .groups = "drop")
  
  # --------------------------------------------------
  # 3. Calculate t-tests (comparing each treatment group vs the controls, by dataset)
  # --------------------------------------------------
  calculate_t_tests <- function(treatment_data, control_data) {
    
    # Summarize controls by dataset
    control_stats <- control_data %>%
      group_by(Dataset) %>%
      summarise(
        control_mean = mean(Prediction),
        control_sd   = sd(Prediction),
        control_n    = n(),
        .groups = "drop"
      )
    
    # Summarize each treatment group and then calculate t-test statistics
    t_test_results <- treatment_data %>%
      group_by(Dataset, Grp_Sex) %>%
      summarise(
        mean_prediction = mean(Prediction),
        sd_prediction   = sd(Prediction),
        n               = n(),
        .groups         = "drop"
      ) %>%
      left_join(control_stats, by = "Dataset") %>%
      rowwise() %>%
      mutate(
        t_statistic = (mean_prediction - control_mean) / 
          sqrt((sd_prediction^2 / n) + (control_sd^2 / control_n)),
        df = ((sd_prediction^2 / n + control_sd^2 / control_n)^2) / 
          ((sd_prediction^2 / n)^2 / (n - 1) + (control_sd^2 / control_n)^2 / (control_n - 1)),
        p_value = 2 * pt(-abs(t_statistic), df)
      ) %>%
      ungroup()
    
    return(t_test_results)
  }
  
  # Compute t-test results
  t_test_results <- calculate_t_tests(df_loo, df_controls)
  
  # --------------------------------------------------
  # 4. Prepare plotting data
  # --------------------------------------------------
  # Attach p-values (and significance stars) back to the LOO data
  plot_data <- df_loo %>%
    left_join(
      t_test_results %>% 
        select(Dataset, Grp_Sex, p_value),
      by = c("Dataset", "Grp_Sex")
    ) %>%
    mutate(
      stars = case_when(
        p_value < 0.001 ~ "***",
        p_value < 0.01  ~ "**",
        p_value < 0.05  ~ "*",
        TRUE            ~ ""
      )
    )
  
  # Also include the control points in the same table, with empty stars for controls
  df_controls_for_plot <- df_controls %>% 
    mutate(stars = "")
  
  # --------------------------------------------------
  # 5. Calculate group medians (for the horizontal lines) and determine star positions
  # --------------------------------------------------
  overall_medians <- bind_rows(plot_data, df_controls_for_plot) %>%
    group_by(Dataset, Grp_Sex) %>%
    summarise(
      overall_median = median(Prediction),
      max_value      = max(Prediction),
      stars          = first(stars[stars != ""]),
      .groups        = "drop"
    ) %>%
    group_by(Dataset) %>%
    mutate(
      # Position the stars 10% above the highest median in each dataset
      star_y_position = max(overall_median, na.rm = TRUE) * 1.1
    ) %>%
    ungroup()
  
  # --------------------------------------------------
  # 6. Create the combined ggplot with a fixed y-axis across facets
  # --------------------------------------------------
  create_combined_graph <- function(plot_data, control_data, overall_medians) {
    
    # Define custom colors for each group
    custom_colors <- c(
      "M_Cont_12" = "black",
      "M_17aE2"   = "blue",
      "M_Cana"    = "green",
      "M_Aca"     = "red",
      "M_Rapa"    = "purple",
      "M_CR"      = "orange"
    )
    
    # Define the order of Grp_Sex levels
    group_order <- c("M_Cont_12", "M_17aE2", "M_Cana", "M_Aca", "M_Rapa", "M_CR")
    
    # Calculate common y-axis limits across all data (with a 10% boost to accommodate star labels)
    all_predictions <- c(plot_data$Prediction, control_data$Prediction)
    common_y_min <- min(all_predictions, na.rm = TRUE)
    common_y_max <- max(all_predictions, na.rm = TRUE) * 1.1
    
    ggplot() +
      # Plot points for treatment groups
      geom_jitter(
        data = plot_data %>% filter(Grp_Sex != "M_Cont_12"),
        aes(x = Grp_Sex, y = Prediction, color = Grp_Sex),
        width = 0.2, height = 0, shape = 17, size = 3, alpha = 0.7
      ) +
      # Plot points for the control group
      geom_jitter(
        data = control_data %>% filter(Grp_Sex == "M_Cont_12"),
        aes(x = Grp_Sex, y = Prediction, color = Grp_Sex),
        width = 0.2, height = 0, shape = 17, size = 3, alpha = 0.7
      ) +
      # Add horizontal lines indicating the median for each group
      geom_segment(
        data = overall_medians,
        aes(x = as.numeric(factor(Grp_Sex, levels = group_order)) - 0.4,
            xend = as.numeric(factor(Grp_Sex, levels = group_order)) + 0.4,
            y = overall_median,
            yend = overall_median,
            color = Grp_Sex),
        size = 1.5
      ) +
      # Add significance stars above the data points
      geom_text(
        data = overall_medians,
        aes(x = Grp_Sex, y = star_y_position, label = stars),
        vjust = -0.5,
        size = 6,
        fontface = "bold"
      ) +
      # Use facet_wrap with scales fixed so that each facet shares the same y-axis
      facet_wrap(~ Dataset, scales = "fixed") +
      # Set the common y-axis limits
      scale_y_continuous(limits = c(common_y_min, common_y_max)) +
      labs(
        title = "Predicted vs Actual % Lifespan Median Increase (LOO Analysis)",
        x     = "Treatment Group",
        y     = "Predicted Lifespan Increase"
      ) +
      scale_color_manual(values = custom_colors, name = "Treatment Group") +
      scale_x_discrete(limits = group_order) +
      theme_minimal() +
      theme(
        plot.title      = element_text(hjust = 0.5, face = "bold", size = 14),
        axis.title      = element_text(face = "bold", size = 12),
        axis.text.x     = element_text(angle = 45, hjust = 1),
        strip.text      = element_text(face = "bold", size = 14),
        legend.title    = element_text(face = "bold", size = 12),
        legend.text     = element_text(size = 10),
        legend.position = "bottom",
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border    = element_rect(color = "black", fill = NA, size = 0.5)
      )
  }
  
  combined_graph <- create_combined_graph(
    plot_data,
    df_controls,
    overall_medians
  )
  
  # --------------------------------------------------
  # 7. Save outputs
  # --------------------------------------------------
  ggsave(
    filename = paste0(output_prefix, "_Combined_LOO_Predicted_vs_Actual_Graph.pdf"),
    plot     = combined_graph, 
    width    = 15, 
    height   = 12, 
    dpi      = 1200
  )
  
  write.xlsx(t_test_results, paste0(output_prefix, "_t_test_results.xlsx"))
  
  # Return the main objects
  return(list(
    loo_predictions     = df_loo,
    control_predictions = df_controls,
    t_test_results      = t_test_results,
    combined_graph      = combined_graph
  ))
}
