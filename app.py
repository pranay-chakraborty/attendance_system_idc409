import cv2
import os
import argparse
import datetime
from src.face_detector import detect_faces
from src.csv_logger import AttendanceLogger
from src.face_recognizer import train_model, evaluate_model
from src.analyze_performance import perform_kfold_validation, analyze_single_split_performance
from src.generate_roc_curves import generate_multiclass_roc

MODEL_PATH = 'models/eigenface_model.yml'
RECOGNITION_THRESHOLD = 3500.0
RESIZE_DIM = (92, 112)

def process_group_photo(image_path: str) -> None:
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Trained model not found at {MODEL_PATH}.")
        print("Please run 'python app.py --train' to train the model first.")
        return
        
    face_recognizer = cv2.face.EigenFaceRecognizer_create()
    face_recognizer.read(MODEL_PATH)
    print("Face recognizer model loaded.")

    image_dir = os.path.dirname(os.path.abspath(image_path))
    csv_path = os.path.join(image_dir, 'attendance.csv')
    logger = AttendanceLogger(filename=csv_path)
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
            color = (0, 255, 0)
            logger.log(label)
        else:
            display_text = "Unknown"
            color = (100, 100, 255)
            unknown_id = f"Unknown_{datetime.datetime.now().strftime('%H%M%S_%f')}"
            logger.log(unknown_id)

        cv2.rectangle(image, (x, y), (x+w, y+h), color, 3)
        cv2.putText(image, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    base, ext = os.path.splitext(image_path)
    output_path = f"{base}_annotated{ext}"
    cv2.imwrite(output_path, image)
    print(f"Process complete. Annotated image saved to: {output_path}")

    cv2.namedWindow('Attendance Results', cv2.WINDOW_NORMAL)
    cv2.imshow('Attendance Results', image)
    print("Press any key to close the window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Face Recognition Attendance System CLI")
    
    group = parser.add_mutually_exclusive_group()
    
    group.add_argument(
        "--train", 
        action="store_true", 
        help="Train the face recognition model and save it."
    )
    group.add_argument(
        "--validate", 
        action="store_true", 
        help="Perform K-Fold validation and generate a detailed performance analysis."
    )
    group.add_argument(
        "--roc", 
        action="store_true", 
        help="Generate and display the ROC curve for the model."
    )
    group.add_argument(
        "-i", "--image", 
        help="Path to a group photo to process for attendance."
    )

    args = parser.parse_args()

    if args.train:
        print("--- Starting Model Training ---")
        train_model()
        print("\n--- Evaluating Trained Model ---")
        evaluate_model()
    elif args.validate:
        from src.data_loader import load_face_data
        import numpy as np
        faces, labels = load_face_data()
        if faces:
            labels = labels.astype(np.int32)
            perform_kfold_validation(faces, labels)
            analyze_single_split_performance()
    elif args.roc:
        generate_multiclass_roc()
    elif args.image:
        process_group_photo(args.image)
    else:
        parser.print_help()