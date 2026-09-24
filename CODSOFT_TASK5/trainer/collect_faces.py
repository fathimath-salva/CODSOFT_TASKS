import cv2
import os
import time

os.makedirs("dataset", exist_ok=True)

person_name = input("Enter your name: ").strip()

if not person_name:
    print("Name cannot be empty.")
    exit()

person_folder = os.path.join("dataset", person_name)
os.makedirs(person_folder, exist_ok=True)

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not open webcam.")
    exit()

count = 0
last_capture_time = 0
capture_interval = 0.15
max_images = 150

print("\nCamera started.")
print("Move your face slowly.")
print("Change distance and angles.")
print("Press Q to stop.")
print(f"Collecting {max_images} images...\n")

while True:

    ret, frame = camera.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    current_time = time.time()

    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        if current_time - last_capture_time >= capture_interval:

            count += 1
            last_capture_time = current_time

            face_image = gray[y:y+h, x:x+w]

            # Resize every face to the same size
            face_image = cv2.resize(face_image, (200, 200))

            file_path = os.path.join(
                person_folder,
                f"{count}.jpg"
            )

            cv2.imwrite(file_path, face_image)

        cv2.putText(
            frame,
            f"Images: {count}/{max_images}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Dataset Collection", frame)

    if count >= max_images:
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print("\nDataset collection completed!")
print(f"Images saved in: {person_folder}")
print(f"Total images: {count}")