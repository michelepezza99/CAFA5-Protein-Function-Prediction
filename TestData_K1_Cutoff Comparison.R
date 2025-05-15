#Script to produce side by side comparisons of 1NN GO-Aware models with different vote cutoffs
library(ggplot2)
library(patchwork)

Cutoffs <- c(0, 0.02, 0.04, 0.06, 0.08, 0.10)

F1avg_k1 <- c(0.4504, 0.3413, 0.2841, 0.2460, 0.2164, 0.1887)

Precision_k1 <- c(0.5219, 0.6191, 0.6528, 0.6752, 0.6937, 0.7099)
  
WeightedF1_k1 <- c(0.4456, 0.2062, 0.1495, 0.1199, 0.0998, 0.0825)
  
ExactRate_k1 <- c(.1946, 0.0353, 0.0293, 0.0209, 0.0170, 0.0070)

cutoff_data <- data.frame(cutoffs, F1avg_k1, Precision_k1, WeightedF1_k1, ExactRate_k1)

plots <- list() 

y_columns <- c("F1avg_k1", "Precision_k1", "WeightedF1_k1", "ExactRate_k1")

for (col in y_columns) {
  p <- ggplot(cutoff_data, aes(x = cutoffs, y = .data[[col]])) +
    geom_point(color = "orange", size = 3) +
    geom_line(color = "darkgreen", linewidth = 1) +
    labs(title = paste(col, "vs. Cutoff"), x = "Cutoff",y = col) +
    theme_minimal() +
    scale_x_continuous(
      breaks = seq(0, 0.10, by = 0.02),  
      limits = c(0, 0.10))
  
  plots[[col]] <- p  # Add plot to list
}

combined_plot <- plots[[1]] + plots[[2]] + plots[[3]] + plots[[4]] +
  plot_layout(nrow = 2, ncol = 2)  # 2x2 grid

print(combined_plot)

# Doing the same within refined cutoff range 0 to 0.02

Cutoffs <- c(0, 0.002, 0.004, 0.006, 0.008, 0.010, 012, 0.014, 0.016, 0.018, 0.020)

F1avg_k1 <- c(0.4504, 0.4331, 0.4180, 0.4048, 0.3938, 0.3835, 0.3739, 0.3652,0.3570, 0.3489, 0.3413)

Precision_k1 <- c(0.5219, 0.5482, 0.5631, 0.5737, 0.5832, 0.5908, 0.5974, 0.6041, 0.6092, 0.6144, 0.6191)

WeightedF1_k1 <- c(0.4456, 0.3648, 0.3253, 0.2990, 0.2791, 0.2627, 0.2485, 0.2364, 0.2255, 0.2152, 0.2062)

ExactRate_k1 <- c(.1946, 0.0882, 0.0714, 0.0595, 0.0520, 0.0463, 0.0396, 0.0385, 0.0378, 0.0360, 0.0353)

cutoff_data <- data.frame(cutoffs, F1avg_k1, Precision_k1, WeightedF1_k1, ExactRate_k1)

plots <- list() 

y_columns <- c("F1avg_k1", "Precision_k1", "WeightedF1_k1", "ExactRate_k1")

for (col in y_columns) {
  p <- ggplot(cutoff_data, aes(x = cutoffs, y = .data[[col]])) +
    geom_point(color = "orange", size = 3) +
    geom_line(color = "darkgreen", linewidth = 1) +
    labs(title = paste(col, "vs. Cutoff"), x = "Cutoff",y = col) +
    theme_minimal() +
    scale_x_continuous(
      breaks = seq(0, 0.10, by = 0.02),  
      limits = c(0, 0.10))
  
  plots[[col]] <- p  # Add plot to list
}

combined_plot <- plots[[1]] + plots[[2]] + plots[[3]] + plots[[4]] +
  plot_layout(nrow = 2, ncol = 2)  # 2x2 grid

print(combined_plot)

