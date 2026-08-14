import os
import random
import shutil

# Original dataset
dataset_path = "/Users/amirali/Documents/vs/FedProject/dataset"

# New folder for test images
test_folder = "/Users/amirali/Documents/vs/FedProject/test_images"

# Number of random images you want
number_of_images = 20

# Create test folder
os.makedirs(test_folder, exist_ok=True)

# Find all images recursively
image_files = []

valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(valid_extensions):
            image_files.append(os.path.join(root, file))

print(f"Found {len(image_files)} images.")

# Randomly select images
selected_images = random.sample(
    image_files,
    min(number_of_images, len(image_files))
)

# Copy selected images
for i, source_path in enumerate(selected_images):

    extension = os.path.splitext(source_path)[1]

    destination_path = os.path.join(
        test_folder,
        f"test_image_{i+1:03d}{extension}"
    )

    shutil.copy2(source_path, destination_path)

print(f"Copied {len(selected_images)} random images.")
print(f"Test folder: {test_folder}")