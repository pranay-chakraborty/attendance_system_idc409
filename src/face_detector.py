import cv2
import numpy as np
from src.data_loader import load_face_data

CASCADE_PATH = 'models/haarcascade_frontalface_default.xml'

def detect_faces(image):
    """
    Detects faces in a grayscale image using a Haar Cascade classifier.

    Args:
        image (numpy.ndarray): The input grayscale image.

    Returns:
        list: A list of tuples (x, y, w, h) for each detected face's bounding box.
    """
    # We load the pre-trained Haar Cascade classifier
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        print(f"Error: Could not load Haar Cascade model from {CASCADE_PATH}")
        return []

    faces_rects = face_cascade.detectMultiScale(
        image, 
        scaleFactor=1.2, 
        minNeighbors=5
    )
    
    return faces_rects

# --- Test block ---
if __name__ == '__main__':
    faces, _ = load_face_data()
    
    if faces:
        sample_image = faces[np.random.randint(0, len(faces))].copy() # radom image from the dataset

        detected_rects = detect_faces(sample_image)
        
        print("--- Face Detection Test ---")
        print(f"Detected {len(detected_rects)} face(s) in the sample image.")

        # We draw a white rectangle around each detected face for visualization
        for (x, y, w, h) in detected_rects:
            cv2.rectangle(sample_image, (x, y), (x+w, y+h), (255, 255, 255), 2)

        cv2.imshow("Detected Face", sample_image)
        print("Displaying image with detected face. Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()