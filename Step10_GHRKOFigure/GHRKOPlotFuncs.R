library(dplyr)
library(ggplot2)
library(openxlsx)
library(tidyr)
library(stats)

process_and_visualize_loo_results <- function(all_loo_predictions_file,
                                              output_prefix = "LOO_Analysis") {
  
  # Custom colors for each group, same as before
  custom_colors <- c(
    "M_GHWT"    = "black",   # control for GHRKO
    "M_GHRKO"   = "blue",
    "M_SnellWT" = "orange",  # control for DW
    "M_DW"      = "purple"
  )
  
  # Treatment-control pairing
  control_mapping <- list(
    "M_GHRKO" = "M_GHWT",
    "M_DW"    = "M_SnellWT"
  )
  
  # --------------------------------------------------------------------------------
  # (1) Calculate Welch's t-tests instead of Student's t-tests (unequal variances)
  # --------------------------------------------------------------------------------
  calculate_t_tests <- function(data) {
    # 1) Summarize control rows
    control_data <- data %>%
      filter(Test_Group %in% unlist(control_mapping)) %>%
      group_by(Dataset, Test_Group) %>%
      summarise(
        control_mean = mean(Prediction),
        control_sd   = sd(Prediction),
        control_n    = n(),
        .groups      = "drop"
      )
    
    # 2) Summarize treatment rows
    treatment_data <- data %>%
      filter(Test_Group %in% names(control_mapping)) %>%
      group_by(Dataset, Test_Group) %>%
      summarise(
        mean_prediction = mean(Prediction),
        sd_prediction   = sd(Prediction),
        n               = n(),
        .groups         = "drop"
      ) %>%
      mutate(control_group = sapply(Test_Group, function(x) control_mapping[[x]]))
    
    # 3) Join them and apply Welch’s formula
    t_test_results <- treatment_data %>%
      left_join(control_data, by = c("Dataset", "control_group" = "Test_Group")) %>%
      rowwise() %>%
      mutate(
        # Welch's t-statistic
        t_statistic = (mean_prediction - control_mean) /
          sqrt((sd_prediction^2 / n) + (control_sd^2 / control_n)),
        # Welch's degrees of freedom
        df = ((sd_prediction^2 / n + control_sd^2 / control_n)^2) /
          (((sd_prediction^2 / n)^2) / (n - 1) +
             ((control_sd^2 / control_n)^2) / (control_n - 1)),
        # Two-sided p-value
        p_value = 2 * pt(-abs(t_statistic), df)
      ) %>%
      ungroup()
    
    return(t_test_results)
  }
  
  # --------------------------------------------------------------------------------
  # (2) Plotting code stays the same (jitter points, medians, significance stars)
  # --------------------------------------------------------------------------------
  create_combined_graph <- function(plot_data, control_data, overall_medians) {
    group_order <- c("M_GHWT", "M_GHRKO", "M_SnellWT", "M_DW")
    
    overall_medians <- overall_medians %>%
      group_by(Dataset) %>%
      mutate(star_y_position = max(overall_median, na.rm = TRUE) * 1.1)
    
    ggplot() +
      geom_jitter(
        data = plot_data,
        aes(x = Test_Group, y = Prediction, color = Test_Group),
        width = 0.2, shape = 17, size = 3, alpha = 0.7
      ) +
      geom_segment(
        data = overall_medians,
        aes(x = as.numeric(factor(Test_Group, levels = group_order)) - 0.4,
            xend = as.numeric(factor(Test_Group, levels = group_order)) + 0.4,
            y = overall_median,
            yend = overall_median,
            color = Test_Group),
        linewidth = 1.5
      ) +
      geom_text(
        data = overall_medians %>% filter(!is.na(stars) & stars != ""),
        aes(x = Test_Group, y = star_y_position, label = stars),
        vjust = -0.5, size = 6, fontface = "bold"
      ) +
      facet_wrap(~ Dataset, scales = "fixed") +
      labs(title = "Predicted Lifespan Increase for GHRKO / DW vs Controls",
           x = "Group",
           y = "Predicted Lifespan Increase") +
      scale_color_manual(values = custom_colors, name = "Treatment Group") +
      scale_x_discrete(limits = group_order) +
      theme_minimal() +
      theme(
        plot.title        = element_text(hjust = 0.5, face = "bold", size = 14),
        axis.title        = element_text(face = "bold", size = 12),
        axis.text.x       = element_text(angle = 45, hjust = 1),
        strip.text        = element_text(face = "bold", size = 14),
        legend.title      = element_text(face = "bold", size = 12),
        legend.text       = element_text(size = 10),
        legend.position   = "bottom",
        panel.grid.major  = element_blank(),
        panel.grid.minor  = element_blank(),
        panel.border      = element_rect(color = "black", fill = NA, linewidth = 0.5)
      )
  }
  
  # --------------------------------------------------------------------------------
  # (3) Main script logic: read data, compute stats, plot, save
  # --------------------------------------------------------------------------------
  df_loo <- read.xlsx(all_loo_predictions_file) %>%
    filter(Test_Group %in% c("M_GHRKO", "M_GHWT", "M_DW", "M_SnellWT"))
  
  # Welch’s t-test
  t_test_results <- calculate_t_tests(df_loo)
  
  # Control data for optional usage in the plot
  control_data <- df_loo %>%
    filter(Test_Group %in% unlist(control_mapping)) %>%
    group_by(Dataset, Test_Group) %>%
    summarise(
      Prediction = median(Prediction),
      .groups = "drop"
    )
  
  # Add p-values (from t-tests) for star annotation
  plot_data <- df_loo %>%
    left_join(
      t_test_results %>% select(Dataset, Test_Group, p_value),
      by = c("Dataset", "Test_Group")
    ) %>%
    mutate(
      stars = case_when(
        p_value < 0.001 ~ "***",
        p_value < 0.01  ~ "**",
        p_value < 0.05  ~ "*",
        TRUE            ~ ""
      )
    )
  
  # Write the combined data to an Excel file
  write.xlsx(plot_data, "plot_data_new.xlsx")
  
  # Compute overall medians
  overall_medians <- plot_data %>%
    group_by(Dataset, Test_Group) %>%
    summarise(
      overall_median = median(Prediction),
      max_value      = max(Prediction),
      stars          = first(stars[stars != ""]), 
      .groups        = "drop"
    )
  
  # Generate the final plot
  combined_graph <- create_combined_graph(plot_data, control_data, overall_medians)
  
  # Save the plot to a PDF
  ggsave(paste0(output_prefix, "_Combined_GHRKO_DW_vs_Controls.pdf"),
         combined_graph, width = 10, height = 8, dpi = 1200)
  
  # Save t-test results
  write.xlsx(t_test_results, paste0(output_prefix, "_t_test_results.xlsx"))
  
  # Return key objects
  return(list(
    loo_predictions     = df_loo,
    control_predictions = control_data,
    t_test_results      = t_test_results,
    combined_graph      = combined_graph
  ))
}
