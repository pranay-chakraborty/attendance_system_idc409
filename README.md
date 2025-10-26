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
