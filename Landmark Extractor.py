"""
This script uses Google MediaPipe Face Landmarker to process a collection of images,
extract 478 face landmarks, and generate 2D Gaussian point heatmaps.
The results are saved as PyTorch (.pt) tensors, typically used for training 
generative AI models like ControlNet or T2I-Adapters.
"""

import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Update these paths to your local dataset and model locations
INPUT_FOLDER = "path/to/your/input_images"
OUTPUT_FOLDER = "path/to/your/output_results"
MODEL_PATH = "path/to/models/face_landmarker.task"

IMG_SIZE = 512          # Target resolution for the heatmap
SIGMA_POINT = 3.0       # Standard deviation for Gaussian kernel
NUM_FACES = 1           # Maximum number of faces to detect per image

heatmap_dir = os.path.join(OUTPUT_FOLDER, "point_heatmap")
os.makedirs(heatmap_dir, exist_ok=True)

# Generate heatmap
def draw_points(points, img_size=512, sigma=2.5):
    """
    Creates a black canvas and draws Gaussian blobs at each landmark coordinate.
    """
    canvas = np.zeros((img_size, img_size), dtype=np.float32)
    for x, y in points:
        # Scale normalized coordinates [0, 1] to pixel coordinates
        ix = int(round(x * (img_size - 1)))
        iy = int(round(y * (img_size - 1)))
        
        if 0 <= ix < img_size and 0 <= iy < img_size:
            canvas[iy, ix] = 1.0
            
    # Apply Gaussian blur to create the 'heatmap' effect
    canvas = gaussian_filter(canvas, sigma=sigma)
    
    # Normalize heatmap intensity to [0, 1]
    if canvas.max() > 0:
        canvas /= canvas.max()
    return canvas


def process_images():
    """
    Recursively scans the input folder, detects landmarks, 
    and saves heatmaps as .pt files.
    """
    # Initialize MediaPipe FaceLandmarker task
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=NUM_FACES
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    # Collect all image file paths recursively
    all_images = []
    for root, _, files in os.walk(INPUT_FOLDER):
        for filename in files:
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                # Use the immediate parent directory name as an identity label
                identity = os.path.basename(root)
                all_images.append((identity, os.path.join(root, filename)))

    print(f"Total {len(all_images)} images found.")

    # Iterate over images with a progress bar
    for identity, img_path in tqdm(all_images, desc="Generating heatmaps"):
        try:
            # Recreate the identity-based folder structure in the output directory
            out_dir = os.path.join(heatmap_dir, identity)
            os.makedirs(out_dir, exist_ok=True)

            base_name = os.path.splitext(os.path.basename(img_path))[0]
            heatmap_path = os.path.join(out_dir, f"{base_name}_heatmap.pt")

            # Skip processing if the file already exists (supports resume)
            if os.path.exists(heatmap_path):
                continue

            # Load and convert image to RGB (MediaPipe requirement)
            image_bgr = cv2.imread(img_path)
            if image_bgr is None:
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

            # Perform landmark detection
            detection_result = landmarker.detect(mp_image)
            if not detection_result.face_landmarks:
                continue

            # Extract (x, y) coordinates for the first face detected
            face_landmarks = detection_result.face_landmarks[0]
            points = [(lm.x, lm.y) for lm in face_landmarks]

            # Convert points to Gaussian heatmap
            heatmap = draw_points(points, IMG_SIZE, SIGMA_POINT)

            # Save the heatmap as a PyTorch tensor
            torch.save(torch.tensor(heatmap, dtype=torch.float32), heatmap_path)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    print("All landmark heatmaps generated successfully.")

if __name__ == "__main__":
    process_images()