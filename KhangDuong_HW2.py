import os

import cv2
import numpy as np

SEED = 42
cv2.setRNGSeed(SEED)

# Remove all existing images in the directory that start with 'HW1_IMG_CS898BA' and end with '.png'
# Except for the original image
for f in os.listdir('.'):
    if f.startswith('HW1_IMG_CS898BA') and f.endswith('.png') and f != 'HW1_IMG_CS898BA.png':
        os.remove(f)


# Part 2: Image Preprocessing & Multi-Channel Normalization
image = cv2.imread('HW1_IMG_CS898BA.png')

blue, green, red = cv2.split(image)

normalized_blue = cv2.equalizeHist(blue)
normalized_green = cv2.equalizeHist(green)
normalized_red = cv2.equalizeHist(red)
normalized_image = cv2.merge((normalized_blue, normalized_green, normalized_red))

cv2.imwrite('HW1_IMG_CS898BA_normalized.png', normalized_image)


# Part 3: Threshold-Based Segmentation
gray_image = cv2.cvtColor(normalized_image, cv2.COLOR_BGR2GRAY)

# 1. Otsu's Global Thresholding
ret, otsu_thresholded = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imwrite('HW1_IMG_CS898BA_otsu_thresholded.png', otsu_thresholded)
otsu_foreground = cv2.bitwise_and(normalized_image, normalized_image, mask=otsu_thresholded)
cv2.imwrite('HW1_IMG_CS898BA_otsu_foreground.png', otsu_foreground)

# 2. Adaptive Thresholding
adaptive_thresholded = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imwrite('HW1_IMG_CS898BA_adaptive_thresholded.png', adaptive_thresholded)
adaptive_foreground = cv2.bitwise_and(normalized_image, normalized_image, mask=adaptive_thresholded)
cv2.imwrite('HW1_IMG_CS898BA_adaptive_foreground.png', adaptive_foreground)

# Part 4: Classical & Optimization-Based Segmentation
# K-means Clustering Segmentation
hsv_image = cv2.cvtColor(normalized_image, cv2.COLOR_BGR2HSV)
height, width = hsv_image.shape[:2]

# Reshape the image into a 2D array of pixels
# Each row represents a pixel, and each column represents a color channel (H, S, V)
pixel_values = hsv_image.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# Define criteria and apply k-means clustering for K = 3, 4, 5
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
attempts = 10

for k in range(3, 6):
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)

    # Sort clusters by centroid V (brightness) so cluster 0 is darkest and cluster k-1 is brightest.
    # This makes the cluster indexing stable for inspection across runs.
    order = np.argsort(centers[:, 2])
    remap = np.zeros(k, dtype=np.int32)
    remap[order] = np.arange(k)
    labels = remap[labels.flatten()]
    centers = centers[order]

    label_map = labels.reshape(height, width)

    # Save the quantized preview (all clusters colored by their centroid) for visual K comparison
    centers_u8 = np.uint8(centers)
    quantized_hsv = centers_u8[labels].reshape(hsv_image.shape)
    quantized_bgr = cv2.cvtColor(quantized_hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(f'HW1_IMG_CS898BA_kmeans_{k}_quantized.png', quantized_bgr)

    # Save a binary mask + foreground extraction for each individual cluster so the
    # figure-bearing cluster can be identified by visual inspection.
    for cluster_idx in range(k):
        cluster_mask = np.where(label_map == cluster_idx, 255, 0).astype(np.uint8)
        cv2.imwrite(f'HW1_IMG_CS898BA_kmeans_{k}_cluster{cluster_idx}_mask.png', cluster_mask)
        cluster_foreground = cv2.bitwise_and(normalized_image, normalized_image, mask=cluster_mask)
        cv2.imwrite(f'HW1_IMG_CS898BA_kmeans_{k}_cluster{cluster_idx}_foreground.png', cluster_foreground)

# Select the optimal K and the cluster that captures the "unknown figure".
# Inspect the *_quantized.png and *_clusterN_mask.png outputs above, then set these.
OPTIMAL_K = 5
FIGURE_CLUSTER = 1

_, labels, centers = cv2.kmeans(pixel_values, OPTIMAL_K, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)
order = np.argsort(centers[:, 2])
remap = np.zeros(OPTIMAL_K, dtype=np.int32)
remap[order] = np.arange(OPTIMAL_K)
labels = remap[labels.flatten()]
label_map = labels.reshape(height, width)

kmeans_mask = np.where(label_map == FIGURE_CLUSTER, 255, 0).astype(np.uint8)
cv2.imwrite('HW1_IMG_CS898BA_kmeans_mask.png', kmeans_mask)
kmeans_foreground = cv2.bitwise_and(normalized_image, normalized_image, mask=kmeans_mask)
cv2.imwrite('HW1_IMG_CS898BA_kmeans_foreground.png', kmeans_foreground)
