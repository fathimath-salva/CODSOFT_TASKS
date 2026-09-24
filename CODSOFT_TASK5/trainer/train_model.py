import cv2
import os
import json
import numpy as np

# Paths
dataset_path = "dataset"
model_path = "trainer/trainer.yml"
labels_path = "trainer/labels.json"

print("\n========================================")
print("     FACE RECOGNITION MODEL TRAINING")
print("========================================\n")

faces = []
ids = []
labels = {}

current_id = 0

# Check dataset
if not os.path.exists(dataset_path):
    print("ERROR: Dataset folder not found.")
    exit()

# Process each person's folder
for person_name in os.listdir(dataset_path):

    person_folder = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_folder):
        continue

    current_id += 1
    labels[current_id] = person_name

    print(f"Processing person: {person_name}")

    image_count = 0

    for filename in os.listdir(person_folder):

        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(person_folder, filename)

        # Read the already-cropped face image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Could not read: {filename}")
            continue

        # Make every training image the same size
        image = cv2.resize(image, (200, 200))

        faces.append(image)
        ids.append(current_id)

        image_count += 1

    print(f"Images loaded: {image_count}\n")

# Check if images were loaded
if len(faces) == 0:
    print("----------------------------------------")
    print("ERROR: No usable face images found.")
    print("Please collect the face dataset again.")
    print("----------------------------------------")
    exit()

print("----------------------------------------")
print(f"Total face samples: {len(faces)}")
print("----------------------------------------")

# Convert IDs to NumPy array
ids = np.array(ids)

# Create LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

print("\nTraining model...")

# Train model
recognizer.train(faces, ids)

# Save model
recognizer.write(model_path)

# Save labels
with open(labels_path, "w") as file:
    json.dump(labels, file, indent=4)

print("\n========================================")
print("   TRAINING COMPLETED SUCCESSFULLY")
print("========================================")
print(f"Model: {model_path}")
print(f"Labels: {labels_path}")
print(f"Total samples used: {len(faces)}")
print("========================================\n")