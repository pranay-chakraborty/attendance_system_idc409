import cv2
import datetime
import os
import argparse
from src.face_detector import detect_faces
from src.csv_logger import AttendanceLogger

MODEL_PATH = 'models/eigenface_model.yml'

RECOGNITION_THRESHOLD = 3500.0
RESIZE_DIM = (92, 112)

def process_group_photo(image_path):
    """
    This function orchestrates the entire recognition process on a single image.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Trained model not found at {MODEL_PATH}.")
        print("Please run 'python -m src.face_recognizer' to train the model first.")
        return
        
    face_recognizer = cv2.face.EigenFaceRecognizer_create()
    face_recognizer.read(MODEL_PATH)
    print("Face recognizer model loaded.")

    logger = AttendanceLogger()

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image file at {image_path}")
        return
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    faces = detect_faces(gray_image)
    print(f"Found {len(faces)} face(s). Now attempting to recognize them...")

    for (x, y, w, h) in faces:
        face_roi = gray_image[y:y+h, x:x+w]
        
        resized_face = cv2.resize(face_roi, RESIZE_DIM, interpolation=cv2.INTER_LANCZOS4)
        
        label, confidence = face_recognizer.predict(resized_face)
        
        if confidence < RECOGNITION_THRESHOLD:
            display_text = f"Subject: {label}"
            color = (0, 255, 0)  # green
            logger.log(label)    
        else:
            display_text = "Unknown"
            color = (100, 100, 255)  # red
            unknown_id = f"Unknown_{datetime.datetime.now().strftime('%H%M%S_%f')}"
            logger.log(unknown_id)

        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        cv2.putText(image, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    base, ext = os.path.splitext(image_path)
    output_path = f"{base}_annotated{ext}"
    cv2.imwrite(output_path, image)
    print(f"Process complete. Annotated image saved to: {output_path}")

    cv2.imshow('Attendance Results', image)
    print("Press any key to close the window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process a group photo for attendance.")
    parser.add_argument("-i", "--image", required=True, help="Path to the group photo image file.")
    args = parser.parse_args()
    
    process_group_photo(args.image)
