import os
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np

SEED = 42
cv2.setRNGSeed(SEED)

# Remove all existing images in the directory that start with 'HW1_IMG_CS898BA' and end with '.png'
# Except for the original image
for f in os.listdir('.'):
    if f.startswith('HW1_IMG_CS898BA') and f.endswith('.png') and f !='HW1_IMG_CS898BA.png':
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
cv2.imwrite('HW1_IMG_CS898BA_otsu_mask.png', otsu_thresholded)
otsu_foreground = cv2.bitwise_and(normalized_image, normalized_image, mask=otsu_thresholded)
cv2.imwrite('HW1_IMG_CS898BA_otsu_foreground.png', otsu_foreground)

# 2. Adaptive Thresholding
adaptive_thresholded = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imwrite('HW1_IMG_CS898BA_adaptive_mask.png', adaptive_thresholded)
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

# Part 5: Evaluation and Analysis
# Re-create the HW1 binary mask (adaptive Gaussian threshold on the RAW grayscale image,
# without the per-channel histogram equalization from Part 2) so we have a like-for-like
# baseline to compare HW2's adaptive output against.
raw_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hw1_binary_image = cv2.adaptiveThreshold(
    src=raw_gray,
    maxValue=255,
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    thresholdType=cv2.THRESH_BINARY,
    blockSize=11,
    C=2
)
cv2.imwrite('HW1_IMG_CS898BA_binary.png', hw1_binary_image)

# 1. Qualitative Analysis
# None of the three methods captures the figure as a single connected region - it is split into disconnected segments in every output and all three masks pull in background pixels.

# Otsu:
# pros: Smoothest mask of the three with a little noise, so the large shapes of the figure stay readable.
# cons: Global cutoff classifies the ground, houses, and sky as white too, and the figure fragments wherever its brightness crosses the threshold.

# Adaptive:
# pros: Sensitive to local intensity changes, so it traces the figure's silhouette and tolerates uneven lighting.
# cons: Output is almost pure salt-and-pepper - grass, houses, cars, trees, and other details all become speckle, burying the figure in noise.

# K-means:
# pros: Clustering in HSV groups pixels by color, so the figure is the dominant content of the chosen cluster - the best of the three.
# cons: A bit background houses, trees, and foliage with a similar color cast still land in the figure cluster, the figure is still fragmented, and edges are blocky.

# Compare to HW1:
# HW1's adaptive binary used the raw grayscale; HW2's uses the equalized grayscale. Per-channel histogram equalization redistributes intensities so all dark or low-variation regions get stretched into a usable range, which gives the local Gaussian window more consistent statistics. HW2's adaptive mask therefore looks cleaner than HW1's: fewer arbitrary speckles in shadowed background and a more coherent silhouette around the figure. The same stretch sharpens the global intensity histogram, which makes Otsu's automatic cutoff land on a meaningful valley instead of collapsing the image into a near-uniform mask, and it helps K-means by widening color separation between the figure and the background.

# 2. Quantitative Analysis
CVAT_mask = cv2.imread('CVAT Segmentation/SegmentationClass/HW1_IMG_CS898BA.png', cv2.IMREAD_GRAYSCALE)
if CVAT_mask is None:
    raise FileNotFoundError(
        "Ground-truth mask not found at 'CVAT Segmentation/SegmentationClass/HW1_IMG_CS898BA.png'."
    )
# Convert the CVAT mask to binary (0 and 1) for IoU and Dice calculations
# Keep figure as white, background as black
CVAT_mask_binary = np.where(CVAT_mask > 0, 255, 0).astype(np.uint8)
cv2.imwrite("HW1_IMG_CS898BA_ground_truth.png", CVAT_mask_binary)

def convert_to_binary(mask):
    return (mask > 0).astype(np.uint8)

# IoU and Dice Coefficient calculations for each method
def calculate_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask)
    union = np.logical_or(pred_mask, gt_mask)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def calculate_dice(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask)
    dice = (2 * np.sum(intersection)) / (np.sum(pred_mask) + np.sum(gt_mask))
    return dice

if CVAT_mask_binary.shape != otsu_thresholded.shape:
    raise ValueError(
        f"Ground-truth shape {CVAT_mask_binary.shape} does not match prediction shape {otsu_thresholded.shape}."
    )

ground_truth_mask_binary = convert_to_binary(CVAT_mask_binary)
otsu_mask_binary = convert_to_binary(otsu_thresholded)
adaptive_mask_binary = convert_to_binary(adaptive_thresholded)
kmeans_mask_binary = convert_to_binary(kmeans_mask)

metrics = {
    'Otsu': {
        'IoU': calculate_iou(otsu_mask_binary, ground_truth_mask_binary),
        'Dice': calculate_dice(otsu_mask_binary, ground_truth_mask_binary)
    },
    'Adaptive': {
        'IoU': calculate_iou(adaptive_mask_binary, ground_truth_mask_binary),
        'Dice': calculate_dice(adaptive_mask_binary, ground_truth_mask_binary)
    },
    'K-Means': {
        'IoU': calculate_iou(kmeans_mask_binary, ground_truth_mask_binary),
        'Dice': calculate_dice(kmeans_mask_binary, ground_truth_mask_binary)
    }
}

print('Quantitative Comparison against Ground Truth')
for method_name, result in metrics.items():
    print(f"{method_name}: IoU={result['IoU']:.4f}, Dice={result['Dice']:.4f}")

# 3. Visualization
# Build side-by-side comparison for README.
# First row: Original, Normalized, Ground Truth
# Second row: Otsu Mask, Adaptive Mask, K-Means Mask
comparison_images = [
    ('Original', cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
    ('Normalized', cv2.cvtColor(normalized_image, cv2.COLOR_BGR2RGB)),
    ('Ground Truth', CVAT_mask_binary),
    ('Otsu Mask', otsu_thresholded),
    ('Adaptive Mask', adaptive_thresholded),
    ('K-Means Mask', kmeans_mask),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
BG_COLOR = '#363636'
TEXT_COLOR = 'white'

fig.patch.set_facecolor(BG_COLOR)

for ax, (title, img) in zip(axes.flatten(), comparison_images):
    if img.ndim == 2:
        ax.imshow(img, cmap='gray')
    else:
        ax.imshow(img)
    ax.set_title(title, color=TEXT_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.axis('off')

plt.tight_layout()
plt.savefig('HW1_IMG_CS898BA_segmentation_comparison.png', dpi=200, bbox_inches='tight')
plt.close(fig)

# Update the README to display the selected plots under "# HW 2 Segmentation Comparison",
# preserving any sections (e.g. "# Discussions") that follow.
README_PATH = 'README.md'
README_MARKER = '# HW 2 Segmentation Comparison'

if os.path.exists(README_PATH):
    with open(README_PATH, 'r', encoding='utf-8') as f:
        readme_content = f.read()
    if README_MARKER in readme_content:
        head, _, tail = readme_content.partition(README_MARKER)
        next_section = re.search(r'^---\s*$', tail[1:], re.MULTILINE)
        rest = tail[1 + next_section.start():] if next_section else ''
        image_md = '\n\n' + "![HW1_IMG_CS898BA_segmentation_comparison.png](HW1_IMG_CS898BA_segmentation_comparison.png)" + '\n\n'
        new_content = head + README_MARKER + image_md + rest
        with open(README_PATH, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f'Updated {README_PATH} with the segmentation comparison plot.')
    else:
        print(f'Marker "{README_MARKER}" not found in {README_PATH}; README not updated.')
else:
    print(f'{README_PATH} not found; skipping README update.')