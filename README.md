## Face Recognition Attendance System Using PCA and OpenCV
This project is a command-line tool for face recognition(can be used for taking attendance) from individual photos or a group photograph. It uses a two stage  process: first, it detects all faces in an image using Haar Cascades, and second, it identifies known individuals using a pre-trained Eigenfaces(PCA) model. Recognised faces are logged to a CSV as 'present', while logging the unrecognised ones as unknown.
___

### Installation
1. Clone the repository: `git clone https://github.com/pranay-chakraborty/attendance_system_idc409.git`
2. Install the required libraries: `pip install -r requirements.txt`
3. Placing your training images in the `dataset/` folder
4. Run `python -m src.face_recognizer` to create the `models/eigenface_model.yml` file. (We already have it for a known dataset, so for sampling you can skip this step)
5. Running the recognition: `python group_photo_recognizer.py --image path/to/your_group_photo.jpg`
### How It Works: The Recognition Pipeline
#### Dataset Used 
The face dataset used in this project is downloaded from the **ORL Database of Faces** https://cam-orl.co.uk/facedatabase.html. There are 40 different subject as samples with 10 images with varying lighting, facial angles and expressions. All the images were already grayscaled and were converted to `.pgm`.
We have mdae a separated module for grayscaling an input image, which could be used for training the model for new faces separately and should be moved to the `/orl_faces/` for usage. This project mainly uses Eigenfaces and OpenCV algorithm to perform the recognition.
___

### Approach & Insights from Exploratory Data Analysis
#### 1. Mean Face
- This is the average of all 400 faces in our dataset. It captures the general common features like head shape, average position of eyes/nose etc and averages out all the features unique to a person.

<img width="300" height="300" alt="01_mean_face" src="https://github.com/user-attachments/assets/f57b2c48-eaa6-4b0b-8be7-5d19dd5df48a" />

- This average face is subtracted from every single image to ensure the PCA algorithm only analyses the differences between faces, reducing the data size to be compared.

#### 2. Variance Comparison

<img width="450" height="400" alt="02_variance_comparison" src="https://github.com/user-attachments/assets/23225ed4-9d7e-4ba9-8975-e6dfb98abba9" />

- This box plot compares the average `variance_between_different_people` to the `variance_within_multiple_pictures_of_same_person`
- This proves that the differences between subjects are significantly larger than the differences within subject photos caused by different lighting or varying expressions. This analysis shows that the recognition is feasible with our dataset.

#### 3. Pixel Variance as Heat Map

<img width="300" alt="03_pixel_variance_heatmap" src="https://github.com/user-attachments/assets/6e24eeb2-1d2d-4ac7-86ff-f57c3bcefc2a" />

-  This heatmap shows which pixels vary the most across our entire dataset.
-  The bright areas like eyes, nose, mouth, hair etc. are showing high variance, meaning that these areas are the most different. While the darker areas have low variance.

<img width="542" height="478" alt="1761508182_grim" src="https://github.com/user-attachments/assets/bde0dae5-fc9b-452a-9758-cbd150ecdc36" />

xi = data value of x
yi = data value of y
x̄ = mean of x
Ȳ = mean of y
N = number of data values
- During the covariance matrix calculation, these high-variance pixels will have the largest values in the matrix and will therefore dominate the principal components.

#### 4. Scree Plot

<img width="600" height="860" alt="04_scree_plot" src="https://github.com/user-attachments/assets/52dbd85d-adfb-4ea5-8c22-e6348c8d1e34" />

- This plot shows the eigenvalues for each component. The y-axis shows the amount of variance 'explained' by each eigenvector.
- **This shows that only a small proportion of eigenvectors hold a significant amount of variance**. 
- As we can see, the first component alone explains 17% of variance. The 20 to 50 components captures vast majority of variances. Thus proving that we can reduce the dimensionality from 10,304 features to just ~100 without losing the necessary information.

#### 4. Eigen Faces

<img width="600" height="600" alt="05_eigenfaces" src="https://github.com/user-attachments/assets/bd68b7dd-563d-4cf9-9bb1-106950821062" />

- Each of these ghostly looking faces is an eigenvector(a principal component). 
- These eigenvectors forms our new basis set for our face space. 
- The `eigenface 1` corresponds to the first, highest-variance capturing component (the highest variance red dot seen in scree plot). Later ones capture finer and more complex details.
___

# Model Algorithm

#### 1. Load Model
Load a pre-trained `eigenface_model.yml` file which was created by `src.face_recognizer` script and contains:
- **Mean Face**
- **The Eigenfaces**
- **The Projections** - is the set of weights for each of our trained subjects, mapping their face into the 'face space' defined by the Eigenfaces.
```# Create the recognizer object
face_recognizer = cv2.face.EigenFaceRecognizer_create()
# Load the pre-calculated Mean Face, Eigenfaces, and Projections
face_recognizer.read(MODEL_PATH)
```

#### 2.Face Detection (Haar Cascade)
The `detect_faces` function scans the input image to find the location of all faces.
  -  This is done by **Haar Cascades Classifier** - a fast object detection algorithm that slides a window across the image, looking for features common to all faces (like a dark eye region above a bright cheek region). And returns a list of bounding boxes:`(x,y,w,h)`
   - `(x, y)` is the x and y coordinates of the **top-left corner** of the rectangle surrounding the face, `w` is the **width** of the rectangle around the face and `h`: The **height** of the rectangle.

#### 3.Process Each Face:
For every face found.
  - Crop the face from the main image.
  - Pre-process it (convert to grayscale, resize).
  - Feed it to the recognizer.
The code iterates through each bounding box found in the previous step:
- The face is **cropped** from the grayscale image using the `(x, y, w, h)` coordinates.
- The cropped image is **resized** to the exact dimensions required by the Eigenface model (e.g., `92x112`). This is critical, as PCA requires all inputs to have the same dimensions.

#### 4. Face Recognition and Validation
The pre-processed face is now ready for identification.

- `label, confidence = face_recognizer.predict(resized_face)` This line performs the core recognition. It takes the resized face and **projects it into the "Face Space"** (using the loaded Mean Face and Eigenfaces).
- It then **calculates the Euclidean distance** (returned as `confidence`) between this new face's projection and the stored projections of all known subjects.

- It returns the `label` (e.g., "Subject 1") corresponding to the projection with the **smallest distance**.
- `if confidence < RECOGNITION_THRESHOLD:` This is the validation step. The `confidence` (distance) is compared to a threshold. If the distance is small enough, the match is accepted. If it's too large, the face is "Unknown" because it's too far from any face in our training data.

#### 5. Log and Annotate
If the match is 'Unknown', it's logged as such. If it's a known `label`, the attendance is logged in a CSV file. A rectangle and the corresponding `label` (or Unknown) are drawn on the original image. This annotated image will then be displayed on the screen and also saved to disk.
<img width="600" height="600" alt="05_eigenfaces" src="https://github.com/user-attachments/assets/ef2aa43e-c70d-42dd-89d9-7daa956667cf" />

- Example of unrecognised face detection
## Model Validation
#### 1. The model acheived a high and consistent average accuracy of 96.5% during 5-fold Cross-Validation.
- Fold 1 = 96.25%
- Fold 2 = 98.75%
- Fold 3 = 97.50%
- Fold 4 = 97.50%
- Fold 5 = 92.50%
#### 2. Confusion matrix: 

<img width="600" alt="face_1" src="[[https://github.com/user-attachments/assets/00324644-3d43-4425-9ee4-b476d8564328](https://github.com/user-attachments/assets/0bdd2184-0a88-4fe0-b31e-a7482dbfde33)](https://github.com/user-attachments/assets/2affc573-c4b7-4ba7-a03a-49826b61cebb)" />

- Overall accuracy on a standard 80/20 split, the model achieved 96% accuracy.
-  Excellent Performance: The model correctly identified the majority of subjects with perfect precision and recall.
-  Specific Confusion: The model struggled with certain
subjects, highlighting areas for future improvement.
- Subject 10: Consistently misidentified (0% recall).
- Subject 23: Occasionally predicted when the true subject
was someone else (50% precision).
- Subject 38: Correctly identified only half the time (50%
recall).

#### 3. Model performance via ROC curve analysis

<img width="600" alt="face_1" src="[https://github.com/user-attachments/assets/00324644-3d43-4425-9ee4-b476d8564328](https://github.com/user-attachments/assets/0bdd2184-0a88-4fe0-b31e-a7482dbfde33)" />

This is a multi class one vs Rest ROC analysis, to show the ability of the model to distinguish a person from the rest.
- We got a nearly perfect score of AUC = 0.98, meaning that the model is reliable and is an effective classifier.
- The curve's position in the top-left corner visually confirms the model's **high True Positive Rate and Low False Positive Rate**.
___

## Additional Modules

## Input Dataprep Module
This script detects faces in images and extracts them as grayscale PGM files with expanded bounding boxes (passport size) for cleaner input for the face recognition module's training.
#### How it works
1. **Loads image** from user-specified path 
2. **Converts to grayscale** for face detection 
3. **Detects faces** using Haar Cascade classifier 
4. **Expands bounding box** by 25% (12.5% on each side) 
5. **Extracts and saves** each face as PGM format 
6. **Shows visual feedback** with color-coded rectangles
- **Green** = Original face detection 
- **Blue** = Expanded region (saved)
#### Usage
1. Install dependencies: `pip install -r requirements_conv.txt`
2. Run `python Input_photo_to_pgm_converter.py`
3. You will be prompted for the `path/to/image.jpg`
	**Tip:** You can use: 
- Full path: `/home/user/photos/group.jpg` 
- Relative path: `images/photo.jpg` 
- Home shortcut: `~/Pictures/photo.jpg`

4. It will detect the face and enralges it to give a full face pgm converted output which can be used in the actual face recognition module.
5. Here the face 
- Displays the image with green boxes around detected faces
- The resizing is done in accordance with the optimum input preference for the model (app.py)

**Detected Face**

<img width="205" height="245" alt="Detected Faces (Green=original, Blue=expanded)_screenshot_26 10 2025" src="https://github.com/user-attachments/assets/f6dc3800-098a-42a3-8720-9ac56d1f72a4" />

**Processed Face**

<img width="200" height="200" alt="face_1" src="https://github.com/user-attachments/assets/00324644-3d43-4425-9ee4-b476d8564328" />

___

## Acknowledgements and References:

_**Eigenfaces (PCA for Recognition):**  Turk, M., & Pentland, A. (1991). Eigenfaces for recognition. *Journal of Cognitive Neuroscience, 3*(1), 71-86._

_**Haar Cascades (Face Detection):** Viola, P., & Jones, M. (2001). Rapid object detection using a boosted cascade of simple features. *In Proceedings of the 2001 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR)*._

_**The ORL Database of Faces:** AT&T Laboratories Cambridge. (1994). The ORL Database of Faces. (Available at: `https://cam-orl.co.uk/facedatabase.html`)_

_**OpenCV (Open Source Computer Vision Library):** Bradski, G. (2000). The OpenCV Library. *Dr. Dobb's Journal of Software Tools*. (Official Site: `https://opencv.org`)_

_**NumPy (Numerical Operations):** Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). Array programming with NumPy. *Nature, 585*, 357–362._

