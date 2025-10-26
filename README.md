## Face Recognition Attendance System Using PCA and OpenCV
This project is a command-line tool for taking attendance from a group photograph. It uses a two stage  process: first, it detects all faces in an image using Haar Cascades, and second, it identifies known individuals using a pre-trained Eigenfaces(PCA) model. Recognised faces are logged to a CSV as 'present', while logging the unrecognised ones as unknown.
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

#### 1. Load Model
Load a pre-trained `eigenface_model.yml` file which was created by `src.face_recognizer` script and contains:
- **Mean Face** - A single average face computed from all faces in our training dataset
- **The Eigenfaces** - A set of 'ghost-like' basis images. These represents the principal components of variation found across all training faces. 
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


