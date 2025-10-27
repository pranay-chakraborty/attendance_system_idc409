# Face Recognition Attendance System - EDA
library(pixmap)
library(imager)
library(ggplot2)

# Create output directory for plots
output_dir <- "EDA_plots"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

## 1. Load Dataset - for 40 subjects with 10 image each

load_face_dataset <- function(base_path = 'face-dataset/') {
  sub_count <- 40
  img_persub <- 10
  img_ht <- 112
  img_wd <- 92
  tot_imgs <- sub_count * img_persub
  
  all_img <- vector('list', tot_imgs)
  y_labs <- character(tot_imgs)
  img_count <- 1
  
  
  for (sub_id in 1:sub_count) {
    sub_folder <- paste0('s', sub_id)
    
    for (img_num in 1:img_persub) {
      file_path <- file.path(base_path, sub_folder, paste0(img_num, '.pgm'))
      pgm_img <- pixmap::read.pnm(file_path)
      all_img[[img_count]] <- as.vector(pgm_img@grey)
      y_labs[img_count] <- sub_folder
      img_count <- img_count + 1
    }
  }
  
  X <- do.call(rbind, all_img)
  y <- as.factor(y_labs)
  
  return(list(X = X, y = y, img_ht = img_ht, img_wd = img_wd, img_persub = img_persub))
}

## 2. Mean Face
compute_mean_face <- function(X, img_ht, img_wd) {
  mean_face_vec <- colMeans(X)
  mean_face_mat <- matrix(mean_face_vec, nrow = img_ht, ncol = img_wd)
  
  png(file.path(output_dir, "01_mean_face.png"), width = 800, height = 800, res = 120)
  plot(as.cimg(mean_face_mat), main = "Mean Face (Average of 400 Images)")
  dev.off()
  
  return(mean_face_vec)
}

## 3. Variance Analysis
analyze_variance <- function(X, y, img_ht, img_wd, img_persub) {
  sub_5_indices <- which(y == 's5')
  within_var <- apply(X[sub_5_indices, ], 2, var)
  
  first_imgs <- seq(from = 1, by = img_persub, length.out = 10)
  between_var <- apply(X[first_imgs, ], 2, var)
  
  mean_within <- mean(within_var)
  mean_between <- mean(between_var)
  ratio <- mean_between / mean_within
  
  df <- data.frame(
    Type = c(rep("Within", length(within_var)), rep("Between", length(between_var))),
    Variance = c(within_var, between_var)
  )
  # Plotting Variance Comparison
  p <- ggplot(df, aes(x = Type, y = Variance, fill = Type)) +
    geom_boxplot() +
    scale_fill_manual(values = c("Within" = "coral", "Between" = "lightgreen")) +
    labs(title = "Variance Comparison", y = "Pixel Variance") +
    theme_minimal() +
    theme(legend.position = "none")
  
  ggsave(file.path(output_dir, "02_variance_comparison.png"), plot = p, 
         width = 8, height = 6, dpi = 300)
  
  return(list(ratio = ratio, mean_within = mean_within, mean_between = mean_between))
}

## 4. Pixel Variance Heatmap
vis
ualize_variance_map <- function(X, img_ht, img_wd) {
  pixel_var <- apply(X, 2, var)
  variance_map <- matrix(pixel_var, nrow = img_ht, ncol = img_wd)
  
  png(file.path(output_dir, "03_pixel_variance_heatmap.png"), 
      width = 800, height = 1000, res = 120)
  image(t(variance_map[nrow(variance_map):1, ]),
        col = heat.colors(100, rev = TRUE),
        main = "Where Do Faces Vary Most?\n(Bright = High Variance)",
        axes = FALSE)
  dev.off()
  
  return(variance_map)
}

## 5.Computing Eigen Values and Eigen Vectors

compute_pca <- function(X, mean_face) {
  # Step 1: Center the data
  X_centered <- sweep(X, 2, mean_face, "-")
  
  # Step 2: Compute covariance matrix using eigenfaces trick
  n <- nrow(X_centered)
  L <- (X_centered %*% t(X_centered)) / n
  
  # Step 3: Eigenvalue decomposition
  eigen_result <- eigen(L)
  eigenvalues <- eigen_result$values
  eigenvectors_L <- eigen_result$vectors
  
  # Step 4: Convert to eigenfaces (project back to image space)
  eigenfaces <- t(X_centered) %*% eigenvectors_L
  
  # Normalize eigenfaces
  for (i in 1:ncol(eigenfaces)) {
    eigenfaces[, i] <- eigenfaces[, i] / sqrt(sum(eigenfaces[, i]^2))
  }
  
  return(list(eigenvalues = eigenvalues, 
              eigenfaces = eigenfaces,
              X_centered = X_centered))

## 6. Scree Plot
plot_scree <- function(eigenvalues) {
  total_variance <- sum(eigenvalues)
  explained_variance <- eigenvalues / total_variance
  cumulative_variance <- cumsum(explained_variance)
  
  n_95 <- which(cumulative_variance >= 0.95)[1]
  
  n_show <- min(100, length(eigenvalues))
  df <- data.frame(
    component = 1:n_show,
    variance = explained_variance[1:n_show]
  )
  
  p <- ggplot(df, aes(x = component, y = variance)) +
    geom_line(color = "blue", size = 1) +
    geom_point(color = "red", size = 2) +
    labs(title = "Scree Plot (Variance Explained by Each Component)",
         x = "Eigen-Component",
         y = "Proportion of Variance") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"))
  
  ggsave(file.path(output_dir, "04_scree_plot.png"), plot = p, 
         width = 10, height = 6, dpi = 300)
  
  return(list(n_95 = n_95))
}

## 7. Visualising Eigenfaces

visualize_eigenfaces <- function(eigenfaces, img_ht, img_wd, n_show = 16) {
  png(file.path(output_dir, "05_eigenfaces.png"), 
      width = 1600, height = 1600, res = 150)
  par(mfrow = c(4, 4), mar = c(1, 1, 2, 1))
  
  for (i in 1:n_show) {
    eigenface_vec <- eigenfaces[, i]
    eigenface_mat <- matrix(eigenface_vec, nrow = img_ht, ncol = img_wd)
    
    # Normalize for visualization
    eigenface_mat <- (eigenface_mat - min(eigenface_mat)) / 
      (max(eigenface_mat) - min(eigenface_mat))
    
    # Flip for correct orientation
    eigenface_cimg <- imager::mirror(as.cimg(eigenface_mat), 'y')
    plot(eigenface_cimg, main = paste("Eigenface", i), axes = FALSE)
  }
  dev.off()
  par(mfrow = c(1, 1)) # Reset graphics device
  
  return(invisible(NULL))
}

## 8. Project Data onto Eigen Facespace

project_faces <- function(X_centered, eigenfaces, n_components = 100) {
  eigenfaces_subset <- eigenfaces[, 1:n_components]
  projected_data <- X_centered %*% eigenfaces_subset
  
  return(projected_data)
}

### Explorting outputs

cat("Running PCA-EDA for Face Recognition...\n")

# Run analysis
dataset <- load_face_dataset('face-dataset/')
X <- dataset$X
y <- dataset$y

mean_face <- compute_mean_face(X, dataset$img_ht, 
                               dataset$img_wd)
var_analysis <- analyze_variance(X, y, dataset$img_ht,
                                 dataset$img_wd,
                                 dataset$img_persub)
variance_map <- visualize_variance_map(X, dataset$img_ht,
                                       dataset$img_wd)

# PCA computation
pca_result <- compute_pca(X, mean_face)
scree_result <- plot_scree(pca_result$eigenvalues)
visualize_eigenfaces(pca_result$eigenfaces,
                     dataset$img_ht,
                     dataset$img_wd, n_show = 16)
projected_data <- project_faces(pca_result$X_centered,
                                pca_result$eigenfaces,
                                n_components = scree_result$n_95)

cat("EDA complete. Plots saved to:", output_dir, "\n")
