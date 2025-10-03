import os
import cv2
import numpy as np

def load_face_data(dataset_path='orl_faces'):
    """
    Loads face images and their corresponding labels from the AT&T dataset directory.

    Args:
        dataset_path (str): The path to the root directory of the dataset.

    Returns:
        tuple: A tuple containing two items:
               - faces (list): A list of images, where each image is a NumPy array.
               - labels (list): A list of integer labels corresponding to each image.
    """
    faces = []
    labels = []
    
    abs_dataset_path = os.path.abspath(dataset_path)

    if not os.path.isdir(abs_dataset_path):
        print(f"Error: Dataset path not found. Please ensure '{dataset_path}' exists.")
        return None, None

    for subject_dir in sorted(os.listdir(abs_dataset_path)):
        subject_path = os.path.join(abs_dataset_path, subject_dir)

        if os.path.isdir(subject_path):
            try:
                label = int(subject_dir.replace('s', ''))
            except ValueError:
                continue

            for image_name in sorted(os.listdir(subject_path)):
                if image_name.endswith('.pgm'):
                    image_path = os.path.join(subject_path, image_name)
                    
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    
                    if image is not None:
                        faces.append(image)
                        labels.append(label)

    if not faces:
        print("Warning: No images were loaded. Check the directory structure and file permissions.")
        return None, None
    
    return faces, np.array(labels)

if __name__ == '__main__':
    faces, labels = load_face_data()
    
    if faces is not None:
        print(f"\n--- Data Loading Test Successful ---")
        print(f"Total images loaded: {len(faces)}")
        print(f"Total labels loaded: {len(labels)}")
        print(f"Number of unique subjects: {len(np.unique(labels))}")
        print(f"Shape of the first image array: {faces[0].shape}")

        cv2.imshow("Test: First Face Loaded", faces[0])
        print("\nDisplaying first image. Press any key in the image window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()