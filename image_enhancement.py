import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

# ---------------------------
# Step 1: Copy original images into input_image folder
# ---------------------------
local_images = [
    r"C:\Users\HP\OneDrive\Desktop\images_project\sunrise.jpg",
    r"C:\Users\HP\OneDrive\Desktop\images_project\polarBear.jpg",
    r"C:\Users\HP\OneDrive\Desktop\images_project\mountain.jpg"
]

input_folder = "input_image"
if not os.path.exists(input_folder):
    os.makedirs(input_folder)

for img_path in local_images:
    file_name = os.path.basename(img_path)
    dest_path = os.path.join(input_folder, file_name)
    shutil.copy(img_path, dest_path)
    print(f"Copied {file_name} to {input_folder}")

# ---------------------------
# Gamma Transformation
# ---------------------------
def gamma_transformation(image, gamma=1.0):
    normalized = image / 255.0
    gamma_corrected = np.power(normalized, gamma)
    return np.uint8(gamma_corrected * 255)

# ---------------------------
# Enhance Image
# ---------------------------
def enhance_image(image_path, output_folder, median_size=3, gamma_pow=1.0):
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not read {image_path}. Check the file path.")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 1. Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Median Filter
    denoised = median_filter(gray, size=median_size)

    # 3. Gamma Correction
    gamma_corrected = gamma_transformation(denoised, gamma=gamma_pow)

    # Save enhanced image
    filename = os.path.basename(image_path)
    save_path = os.path.join(output_folder, f"enhanced_{median_size}_{gamma_pow}_{filename}")
    cv2.imwrite(save_path, gamma_corrected)
    print(f"✅ Saved: {save_path}")

    # Show results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 4, 1), plt.imshow(image_rgb), plt.title("Original"), plt.axis("off")
    plt.subplot(1, 4, 2), plt.imshow(gray, cmap="gray"), plt.title("Grayscale"), plt.axis("off")
    plt.subplot(1, 4, 3), plt.imshow(denoised, cmap="gray"), plt.title(f"Median {median_size}"), plt.axis("off")
    plt.subplot(1, 4, 4), plt.imshow(gamma_corrected, cmap="gray"), plt.title(f"Gamma {gamma_pow}"), plt.axis("off")
    plt.show()

# ---------------------------
# Process Folder
# ---------------------------
def process_folder(input_folder, output_folder, median_size=3, gamma_pow=1.0):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in os.listdir(input_folder):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, file)
            print(f"Processing: {file}")
            enhance_image(image_path, output_folder, median_size, gamma_pow)

# ---------------------------
# Runs
# ---------------------------
if __name__ == "__main__":
    output_folder = "output_image"  # inside project folder

    # Run with different parameters
    process_folder(input_folder, output_folder, median_size=3, gamma_pow=0.5)
    process_folder(input_folder, output_folder, median_size=5, gamma_pow=1.0)
    process_folder(input_folder, output_folder, median_size=7, gamma_pow=2.0)
