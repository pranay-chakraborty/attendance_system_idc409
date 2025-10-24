import os
import cv2
import numpy as np

TARGET_SIZE = (92, 112)

def load_face_data(dataset_path='orl_faces'):
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
                        image_resized = cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_AREA)
                        faces.append(image_resized)
                        labels.append(label)
    if not faces:
        print("Warning: No images were loaded. Check the directory structure and file permissions.")
        return None, None
    return faces, np.array(labels)

if __name__ == '__main__':
    faces, labels = load_face_data()
    random_number = np.random.randint(0, len(faces)) if faces is not None else 0
    if faces is not None:
        print(f"\n--- Data Loading Test Successful ---")
        print(f"Total images loaded: {len(faces)}")
        print(f"Total labels loaded: {len(labels)}")
        print(f"Number of unique subjects: {len(np.unique(labels))}")
        print(f"Shape of the first image array: {faces[0].shape}")
        cv2.imshow("Test: Random Face Loaded", faces[random_number])
        print("\nDisplaying first image. Press any key in the image window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
