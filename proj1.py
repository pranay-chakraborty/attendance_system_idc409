import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import cv2
    import numpy as np

    # Loading classifier and image
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    image = cv2.imread('/home/aswin/ClassAttendance/face/b2.jpg')
    if image is None:
        print("err, no img")
        exit()

    # Preprocessing

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)

    print(f"found {len(faces)} faces")

    # Display the result

    cv2.imshow('Detected Faces', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return cv2, faces, gray, image


@app.cell
def _(cv2, faces, gray, image):
    for (x,y,w,h) in faces:
        cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)
        face_roi = gray[y:y+h, x:x+w]
        processed_face = cv2.resize(face_roi, (92, 112))
        cv2.imwrite(f'processed_face_{x}_{y}.jpg', processed_face)

    cv2.imshow('Detected Faces', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


if __name__ == "__main__":
    app.run()
