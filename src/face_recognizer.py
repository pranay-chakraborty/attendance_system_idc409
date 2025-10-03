import cv2
import os
import numpy as np
from src.prepare_data import prepare_data

# The path to save the trained model
MODEL_PATH = 'models/eigenface_model.yml'

def train_model():
    """
    Trains the Eigenface recognizer on the dataset and saves the model.
    """
    print("Preparing data for training...")
    train_faces, test_faces, train_labels, test_labels = prepare_data()
    
    if train_faces is None:
        print("Could not prepare data. Aborting training.")
        return

    print("Data prepared successfully.")
    print(f"Training on {len(train_faces)} images...")

    face_recognizer = cv2.face.EigenFaceRecognizer_create()

    face_recognizer.train(train_faces, train_labels)

    face_recognizer.save(MODEL_PATH)

    print(f"Training complete. Model saved to {MODEL_PATH}")

def evaluate_model():
    """
    Loads the trained model and evaluates its accuracy on the test set.
    """
    print("\nEvaluating model performance...")
    _, test_faces, _, test_labels = prepare_data()

    if test_faces is None:
        print("Could not prepare data for evaluation.")
        return

    face_recognizer = cv2.face.EigenFaceRecognizer_create()
    face_recognizer.read(MODEL_PATH)

    correct_predictions = 0
    for i, test_image in enumerate(test_faces):
        predicted_label, confidence = face_recognizer.predict(test_image)
        
        actual_label = test_labels[i]

        if predicted_label == actual_label:
            correct_predictions += 1
        

    accuracy = (correct_predictions / len(test_faces)) * 100
    print("--- Evaluation Complete ---")
    print(f"Correct predictions: {correct_predictions} / {len(test_faces)}")
    print(f"Model Accuracy: {accuracy:.2f}%")

if __name__ == '__main__':
    if not os.path.exists('models'):
        os.makedirs('models')

    train_model()
    evaluate_model()