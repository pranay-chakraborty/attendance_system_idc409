import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import cv2
    import numpy as np
    import os

    # Request the directory path from user
    image_path = input("Enter the path to your image file: ").strip()

    # Expand user home directory if path starts with ~
    image_path = os.path.expanduser(image_path)

    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: File not found at '{image_path}'")
        print("Please check the file path and try again.")
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Loading classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Load image from user-provided path
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not read image at '{image_path}'")
        print("Please check the file path and try again.")
        raise ValueError(f"Failed to load image: {image_path}")

    print(f"✓ Image loaded successfully: {image.shape}")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)
    print(f"Found {len(faces)} face(s)")

    # Get image dimensions
    img_height, img_width = gray.shape

    # Process each detected face
    for i, (x, y, w, h) in enumerate(faces):
        # Calculate 1/4th expansion (12.5% on each side = 25% total per dimension)
        expansion = 0.125  # 12.5% expansion on each side
    
        # Calculate new coordinates
        new_w = int(w * (1 + 2 * expansion))
        new_h = int(h * (1 + 2 * expansion))
        new_x = int(x - w * expansion)
        new_y = int(y - h * expansion)
    
        # Ensure coordinates stay within image bounds
        new_x = max(0, new_x)
        new_y = max(0, new_y)
        new_w = min(new_w, img_width - new_x)
        new_h = min(new_h, img_height - new_y)
    
        # Draw original detection rectangle (green)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
        # Draw expanded rectangle (blue)
        cv2.rectangle(image, (new_x, new_y), (new_x+new_w, new_y+new_h), (255, 0, 0), 2)
    
        # Extract expanded face region from grayscale image
        face_roi = gray[new_y:new_y+new_h, new_x:new_x+new_w]
    
        # Save as PGM format (grayscale)
        output_filename = f'face_{i+1}.pgm'
        cv2.imwrite(output_filename, face_roi)
        print(f"  Saved: {output_filename} (size: {face_roi.shape[1]}x{face_roi.shape[0]})")
    
        # Show individual face in a window
        cv2.imshow(f'Face {i+1}', face_roi)

    # Display the result with rectangles
    # Green = original detection, Blue = expanded (saved) region
    cv2.imshow('Detected Faces (Green=original, Blue=expanded)', image)
    print("Press any key to close all windows.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


if __name__ == "__main__":
    app.run()
