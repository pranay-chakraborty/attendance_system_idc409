
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from src.data_loader import load_face_data
from src.prepare_data import prepare_data

MODEL_PATH = 'models/eigenface_model.yml'


def perform_kfold_validation(faces, labels, n_splits=5):
    """
    Performs K-Fold Cross-Validation on the Eigenface model.
    """
    print("--- 1. Starting K-Fold Cross-Validation ---")
    
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_accuracies = []
    fold_number = 1

    for train_indices, test_indices in kfold.split(faces, labels):
        print(f"\n--- FOLD {fold_number}/{n_splits} ---")

        train_faces = [faces[i] for i in train_indices]
        train_labels = labels[train_indices]
        
        test_faces = [faces[i] for i in test_indices]
        test_labels = labels[test_indices]
        
        print(f"Training on {len(train_faces)} images, testing on {len(test_faces)} images.")

        face_recognizer = cv2.face.EigenFaceRecognizer_create()
        face_recognizer.train(train_faces, train_labels)

        correct_predictions = 0
        for i, test_image in enumerate(test_faces):
            predicted_label, _ = face_recognizer.predict(test_image)
            if predicted_label == test_labels[i]:
                correct_predictions += 1
        
        accuracy = (correct_predictions / len(test_faces)) * 100
        fold_accuracies.append(accuracy)
        print(f"Accuracy for Fold {fold_number}: {accuracy:.2f}%")
        
        fold_number += 1

    mean_accuracy = np.mean(fold_accuracies)
    std_deviation = np.std(fold_accuracies)

    print("\n\n--- K-Fold Cross-Validation Summary ---")
    print(f"Number of folds (K): {n_splits}")
    print(f"Individual Fold Accuracies: {[f'{acc:.2f}%' for acc in fold_accuracies]}")
    print(f"Average Accuracy: {mean_accuracy:.2f}%")
    print(f"Standard Deviation: +/- {std_deviation:.2f}%")
    print("---------------------------------------")


def analyze_single_split_performance():
    """
    Analyzes one specific train-test split with a classification report and confusion matrix.
    """
    print("\n\n--- 2. Starting Detailed Performance Analysis (Single Split) ---")

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Final trained model not found at {MODEL_PATH}.")
        print("Please run 'python -m src.face_recognizer' to train the final model first.")
        return

    _, test_faces, _, y_true = prepare_data()
    if test_faces is None:
        print("Failed to load test data.")
        return

    face_recognizer = cv2.face.EigenFaceRecognizer_create()
    face_recognizer.read(MODEL_PATH)
    print("Final trained model loaded successfully.")

    print("Making predictions on the test set...")
    y_pred = []
    for test_image in test_faces:
        predicted_label, _ = face_recognizer.predict(test_image)
        y_pred.append(predicted_label)
    
    class_labels = np.unique(y_true)
    class_names = [f"s{label}" for label in class_labels]

    print("\n\n--- Classification Report ---")
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print(report)

    print("\n--- Generating Confusion Matrix ---")
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=False, cmap='viridis', xticklabels=class_names, yticklabels=class_names)
    
    plt.title('Confusion Matrix (Single Split)', fontsize=20)
    plt.ylabel('True Subject ID', fontsize=16)
    plt.xlabel('Predicted Subject ID', fontsize=16)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    
    print("Displaying confusion matrix plot. Close the plot window to exit.")
    plt.show()


if __name__ == '__main__':
    faces, labels = load_face_data()
    
    if faces is not None:
        labels = labels.astype(np.int32)
        
        perform_kfold_validation(faces, labels)
        
        analyze_single_split_performance()
        