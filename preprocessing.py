import cv2
import numpy as np


def preprocess_bike_image(image_path, target_size=(500, 500)):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unreadable.")

    # Resize to target size
    resized = cv2.resize(image, target_size)

    # Convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Apply Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Denoising with bilateral filter
    #denoised = cv2.bilateralFilter((enhanced * 255).astype(np.uint8), d=9, sigmaColor=75, sigmaSpace=75)

    # Denoising with Non-local Means Denoising
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10, templateWindowSize=7, searchWindowSize=21)

    # Morphological operations to enhance bike structure
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(denoised, cv2.MORPH_GRADIENT, kernel)

    # Edge detection with automatic threshold calculation and using Otsu's method to find optimal thresholds
    high_thresh = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    low_thresh = 0.5 * high_thresh
    edges = cv2.Canny(morph, low_thresh, high_thresh)

    return morph, edges
