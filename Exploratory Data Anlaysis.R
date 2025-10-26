## Visual EDA for our dataset

library(pixmap)
library(imager)

base <- ('/home/aswin/ClassAttendance/face-dataset')

# Defining subjects based on the dataset's structure
sub_count <- 40
img_persub <- 10
img_ht <- 112
img_wd <- 92

tot_imgs <- sub_count*img_persub
vector_len <- img_ht*img_wd

# Loading the data
cat(' loading data ')
all_img <- vector('list', tot_imgs)
# Define constants based on the dataset's structure

y_labs <- character(tot_imgs)
img_count <- 1

# looping through folders(s1,s2..)
for (sub_id in 1:sub_count){
  sub_folder <- paste0('s', sub_id)
  for (img_num in 1:img_persub){
    file_name <- paste0(img_num, '.pgm')
    file_path <- file.path(base, sub_folder, file_name)
    
    #reading PGM using pixmap
    pgm_img <- pixmap::read.pnm(file_path)
    
    # flattening the matrix colwise into vector. 
    img_vector <- as.vector(pgm_img@grey)
    
    #storing the vector and label
    all_img[[img_count]] <- img_vector
    y_labs[img_count] <- sub_folder
    
    #increment the counter
    img_count <- img_count+1
  }
}

# Combine the list of vectors into a single matrix
x <- do.call(rbind, all_img)
# y as labels
y <- as.factor(y_labs)

# Verification
cat("Data loading complete.\n")
cat("Dimensions of the data matrix X:", dim(x), "\n")
cat("Length of the label vector y:", length(y), "\n\n")

## Checking the data balance ##
cat('Checking the data balance\n\n')
tab <- data.frame(X=x,Y=y)
View(tab)
summary(tab)


## PLotting the mean face

cat('2. Plotting the mean face')

mean_face_vec <- colMeans(x)
mean_face_mat <- matrix(data = mean_face_vec,
                        nrow = img_ht,
                        ncol = img_wd)

# imager to make it plottable
mean_face_cimg <- as.cimg(mean_face_mat)

#plot
mean_face <- plot(mean_face_cimg, main = "The mean face (avg of 400 imgs)")
cat("The 'mean face' plot is now displayed. It shows the average features across everyone.\n")

## Plot within subject variance  ##

cat('3. plotting the within subject variance')

plot_sub <- 's5'
cat(paste('Displaying all 10 imgs for subject', plot_sub, '\n'))
# getting indices for the subject and extract the 10 img vectors for this subject
sub_indices <- which(y == plot_sub)
sub_imgmatrix <- x[sub_indices,]
# plot area 2x5 grid
par(mfrow = c(2,5), mar = c(1,1,2,1))


# Loop through and plot each of the 10 images
for (i in 1:nrow(sub_img_matrix)){
  img_vector <- sub_img_matrix[i,]
  img_matrix <- matrix(img_vector, nrow = img_ht, ncol=img_wd)
  #Plot using imager
  plot(as.cimg(img_matrix), main = paste('Image', i))
  
}

# Reset the plotting device to a single plot
par(mfrow = c(1,1))
cat('The grid shows variance to ignore - lighting, smiling vs neutral, head tilt \n\n')

## 4. Plotting between subject variances

cat('5. Plotting between subject variance \n')

# Taking the first image of the first 10 subjects. since the data is ordered
# The rows will be then, 1,11,21,31,...
indices_to_plot <- seq(from=1, by=img_persub, length.out=10)

# Extract the corresponding img vectors and labels
between_sub_imgs <- x[indices_to_plot, ]
between_sub_labs <- y[indices_to_plot]

# Set up the plotting area to be a 2x5 grid
par(mfrow = c(2,5), mar = c(1,1,2,1))

# Loop through and plot each of these 10 imgs
for (i in 1:nrow(between_sub_imgs)) {
  # Get the single image vector
  img_vector <- between_sub_imgs[i, ]
  
  # Reshape it back to a matrix
  img_matrix <- matrix(img_vector, nrow = img_ht, ncol = img_wd)
  
  # Convert to a cimg object to use imager's functions
  img_cimg <- as.cimg(img_matrix)
  
  # Flip the image vertically (along the 'y' axis)
  img_cimg_flipped <- imager::mirror(img_cimg, 'y')
  
  # Plot the FLIPPED image, using the subject ID as the title
  plot(img_cimg_flipped, main = as.character(between_sub_labs[i]))
}
par(mfrow = c(1,1))
cat('The grid plot shows variance to capture: different identities, headshapes etc. \n')
cat('Visually, the between subject variance appears to be greater than the within-subject variance, which is good for a face recognition model \n')



