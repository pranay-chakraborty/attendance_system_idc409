import numpy as np
from sklearn.model_selection import train_test_split
from src.data_loader import load_face_data # Assuming you run from root

def prepare_data(dataset_path='orl_faces', test_size=0.2, random_state=42):
    """
    Loads, prepares, and splits the face data into training and testing sets.
    """
    faces, labels = load_face_data(dataset_path)
    if faces is None:
        return None, None, None, None

    labels = labels.astype(np.int32)
    
    
    X_train, X_test, y_train, y_test = train_test_split(
        faces, labels, test_size=test_size, stratify=labels, random_state=random_state
    )

    return X_train, X_test, y_train, y_test

# --- Test block ---
if __name__ == '__main__':
    train_faces, test_faces, train_labels, test_labels = prepare_data()

    if train_faces is not None:
        print("--- Data Preparation and Splitting Test Successful ---")
        print(f"Total training images: {len(train_faces)}")
        print(f"Total testing images:  {len(test_faces)}")
        print(f"Unique subjects in training set: {len(np.unique(train_labels))}")
        print(f"Unique subjects in testing set:  {len(np.unique(test_labels))}")