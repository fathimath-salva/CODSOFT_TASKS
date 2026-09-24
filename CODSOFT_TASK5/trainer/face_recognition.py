import cv2
import json
import os

# Paths
model_path = "trainer/trainer.yml"
labels_path = "trainer/labels.json"

# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_path)

# Load names
with open(labels_path, "r") as file:
    names = json.load(file)

# Load face detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Start webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("===================================")
print("     FACE RECOGNITION SYSTEM")
print("===================================")
print("Camera started.")
print("Press Q to exit.")

while True:

    ret, frame = camera.read()

    if not ret:
        print("Could not read camera frame.")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(100, 100)
    )

    for (x, y, w, h) in faces:

        # Recognize face
        face_id, confidence = recognizer.predict(
            gray[y:y + h, x:x + w]
        )

        # Convert confidence to approximate accuracy
        accuracy = round(100 - confidence)

        # Check if confidence is good enough
        if accuracy >= 50 and str(face_id) in names:
            person_name = names[str(face_id)]
        else:
            person_name = "Unknown"

        # Draw face rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Display name
        cv2.putText(
            frame,
            person_name,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        # Display accuracy
        cv2.putText(
            frame,
            f"Confidence: {accuracy}%",
            (x, y + h + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Display application title
    cv2.putText(
        frame,
        "Face Detection & Recognition",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow("Face Recognition", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print("Face recognition system closed.")