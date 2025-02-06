process_and_visualize_loo_results <- function(all_loo_predictions_file, control_predictions_file, output_prefix = "LOO_Analysis") {
  
  # Define custom colors for each group
  custom_colors <- c(
    "M_Cont_12" = "black",
    "M_17aE2" = "blue",
    "M_Cana"   = "green",
    "M_Aca"    = "red",
    "M_Rapa"   = "purple",
    "M_CR"     = "orange"
  )
  
  # Function to calculate t-tests with unique Mouse_IDs
  calculate_t_tests <- function(data, control_data) {
    # Summarise control data by Mouse_ID to calculate medians
    control_stats <- control_data %>%
      group_by(Dataset, Mouse_ID) %>%
      summarise(control_median = median(Prediction), .groups = "drop") %>%
      group_by(Dataset) %>%
      summarise(
        control_mean = mean(control_median),
        control_sd = sd(control_median),
        control_n = n_distinct(Mouse_ID)  # Use unique Mouse_ID count
      )
    
    # Summarise non-control data by Mouse_ID and calculate t-statistics
    t_test_results <- data %>%
      group_by(Dataset, Grp_Sex, Mouse_ID) %>%
      summarise(mouse_median = median(Prediction), .groups = "drop") %>%
      group_by(Dataset, Grp_Sex) %>%
      summarise(
        mean_prediction = mean(mouse_median),
        sd_prediction = sd(mouse_median),
        n = n_distinct(Mouse_ID),  # Use unique Mouse_ID count
        .groups = "drop"
      ) %>%
      left_join(control_stats, by = "Dataset") %>%
      rowwise() %>%
      mutate(
        t_statistic = (mean_prediction - control_mean) /
          sqrt((sd_prediction^2 / n) + (control_sd^2 / control_n)),
        df = ((sd_prediction^2 / n + control_sd^2 / control_n)^2) /
          (((sd_prediction^2 / n)^2 / (n - 1)) + ((control_sd^2 / control_n)^2 / (control_n - 1))),
        p_value = 2 * pt(-abs(t_statistic), df)
      ) %>%
      ungroup()
    
    return(t_test_results)
  }
  
  # Function to create the combined graph
  create_combined_graph <- function(plot_data, control_data, overall_medians) {
    group_order <- c("M_Cont_12", "M_17aE2", "M_Cana", "M_Aca", "M_Rapa", "M_CR")
    
    overall_medians <- overall_medians %>%
      group_by(Dataset) %>%
      mutate(star_y_position = max(overall_median, na.rm = TRUE) * 0.8)
    
    ggplot() +
      # Plot non-control points (median per Mouse_ID)
      geom_jitter(data = plot_data %>% filter(Grp_Sex != "M_Cont_12"),
                  aes(x = Grp_Sex, y = Prediction, color = Grp_Sex),
                  width = 0.2, height = 0, shape = 17, size = 3, alpha = 0.7) +
      # Plot control points (already summarized to median per Mouse_ID)
      geom_jitter(data = control_data %>% filter(Grp_Sex == "M_Cont_12"),
                  aes(x = Grp_Sex, y = Prediction, color = Grp_Sex),
                  shape = 17, size = 3, alpha = 0.7) +
      # Plot overall median line for each group
      geom_segment(data = overall_medians,
                   aes(x = as.numeric(factor(Grp_Sex, levels = group_order)) - 0.4,
                       xend = as.numeric(factor(Grp_Sex, levels = group_order)) + 0.4,
                       y = overall_median,
                       yend = overall_median,
                       color = Grp_Sex),
                   size = 1.5) +
      # Add significance stars
      geom_text(data = overall_medians,
                aes(x = Grp_Sex, y = star_y_position, label = stars),
                vjust = -0.5, size = 6, fontface = "bold") +
      facet_wrap(~ Dataset, scales = "free_y") +
      labs(title = "Predicted vs Actual % Lifespan Median Increase for All Datasets",
           x = "Treatment Group",
           y = "Predicted Lifespan Increase") +
      scale_color_manual(values = custom_colors, name = "Treatment Group") +
      scale_x_discrete(limits = group_order) +
      theme_minimal() +
      theme(
        plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
        axis.title = element_text(face = "bold", size = 12),
        axis.text.x = element_text(angle = 45, hjust = 1),
        strip.text = element_text(face = "bold", size = 14),
        legend.title = element_text(face = "bold", size = 12),
        legend.text = element_text(size = 10),
        legend.position = "bottom",
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_rect(color = "black", fill = NA, size = 0.5)
      )
    
  }
  
  # Load LOO predictions and control data
  df_loo <- read.xlsx(all_loo_predictions_file)
  df_controls <- read.xlsx(control_predictions_file) %>%
    filter(Grp_Sex == "M_Cont_12")
  
  # Calculate t-tests using unique Mouse_ID counts
  t_test_results <- calculate_t_tests(df_loo, df_controls)
  
  # Prepare control data: summarise per Mouse_ID and add empty stars
  df_controls_cleaned <- df_controls %>%
    group_by(Dataset, Grp_Sex, Mouse_ID) %>%
    summarise(Prediction = median(Prediction), .groups = "drop") %>%
    mutate(stars = "")
  
  # Add t-test results and stars for significance to all LOO data
  plot_data_full <- df_loo %>%
    left_join(t_test_results %>% select(Dataset, Grp_Sex, p_value), by = c("Dataset", "Grp_Sex")) %>%
    mutate(
      stars = case_when(
        p_value < 0.001 ~ "***",
        p_value < 0.01 ~ "**",
        p_value < 0.05 ~ "*",
        TRUE ~ ""
      )
    )
  
  # Calculate overall medians for each group (including control)
  overall_medians <- bind_rows(
    plot_data_full %>% 
      group_by(Dataset, Grp_Sex, Mouse_ID) %>% 
      summarise(Prediction = median(Prediction), stars = first(stars), .groups = "drop"),
    df_controls_cleaned
  ) %>%
    group_by(Dataset, Grp_Sex) %>%
    summarise(
      overall_median = median(Prediction),
      max_value = max(Prediction),
      stars = first(stars[stars != ""]),
      .groups = "drop"
    )
  
  # *** NEW ADJUSTMENT ***
  # Summarize the LOO predictions by Mouse_ID (so that each Mouse_ID per Dataset appears only once with its median)
  plot_data_median <- plot_data_full %>%
    group_by(Dataset, Grp_Sex, Mouse_ID) %>%
    summarise(Prediction = median(Prediction),
              stars = first(stars),
              .groups = "drop")
  
  # Create and save the combined graph using the median-per-Mouse data for plotting
  combined_graph <- create_combined_graph(plot_data_median, df_controls_cleaned, overall_medians)
  ggsave(paste0(output_prefix, "_Predicted_Median_Lifespan_Extension.pdf"),
         combined_graph, width = 15, height = 12, dpi = 1200)
  
  # Save t-test results to Excel
  write.xlsx(t_test_results, paste0(output_prefix, "_t_test_results.xlsx"))
  
  return(list(
    loo_predictions = df_loo,
    control_predictions = df_controls,
    t_test_results = t_test_results,
    combined_graph = combined_graph,
    plot_data = plot_data_median
  ))
}
