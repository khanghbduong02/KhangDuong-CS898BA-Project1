import os

import cv2
import numpy as np

rng = np.random.default_rng(seed=42)

image = cv2.imread('HW1_IMG_CS898BA.png')
print(image.shape)

blue, green, red = cv2.split(image)

# Part 2:
# 1. Find and print basic image statistics of the original image for each individual channel (min, max, average, median, mode, skew, range, standard deviation, variance)
print("Blue channel:")
print("    min:", blue.min())
print("    max:", blue.max())
print("    average:", blue.mean())
print("    median:", np.median(blue))
print("    mode:", np.bincount(blue.flatten()).argmax())
print("    skew:", (blue.mean() - np.median(blue)) / blue.std())
print("    range:", blue.max() - blue.min())
print("    std:", blue.std())
print("    variance:", blue.var())

print("Green channel:")
print("    min:", green.min())
print("    max:", green.max())
print("    average:", green.mean())
print("    median:", np.median(green))
print("    mode:", np.bincount(green.flatten()).argmax())
print("    skew:", (green.mean() - np.median(green)) / green.std())
print("    range:", green.max() - green.min())
print("    std:", green.std())
print("    variance:", green.var())

print("Red channel:")
print("    min:", red.min())
print("    max:", red.max())
print("    average:", red.mean())
print("    median:", np.median(red))
print("    mode:", np.bincount(red.flatten()).argmax())
print("    skew:", (red.mean() - np.median(red)) / red.std())
print("    range:", red.max() - red.min())
print("    std:", red.std())
print("    variance:", red.var())

# 2. Convert and save the image to greyscale, binary, and different color spaces (HSV, CIELAB, and HLS).
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('HW1_IMG_CS898BA_grayscale.png', grayscale_image)

# Binary image using Adaptive Gaussian Thresholding
binary_image = cv2.adaptiveThreshold(
    src=grayscale_image,
    maxValue=255,
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    thresholdType=cv2.THRESH_BINARY,
    blockSize=11,
    C=2
)
cv2.imwrite('HW1_IMG_CS898BA_binary.png', binary_image)

HSV_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imwrite('HW1_IMG_CS898BA_HSV.png', HSV_image)

CIELAB_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
cv2.imwrite('HW1_IMG_CS898BA_CIELAB.png', CIELAB_image)

HLS_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
cv2.imwrite('HW1_IMG_CS898BA_HLS.png', HLS_image)

# 3. On the HSV converted image, normalize the lighting by performing histogram equalization across the V (value) channel.
hue, saturation, value = cv2.split(HSV_image)
equalized_value = cv2.equalizeHist(value)
equalized_HSV = cv2.merge((hue, saturation, equalized_value))

# 4. Convert the normalized image back to RGB and save it.
equalized_image = cv2.cvtColor(equalized_HSV, cv2.COLOR_HSV2BGR)
cv2.imwrite('HW1_IMG_CS898BA_equalized.png', equalized_image)

# 5. You should now have 7 images.
print("Total images in directory:", len([f for f in os.listdir('.') if f.startswith('HW1_IMG_CS898BA') and f.endswith('.png')]))

# 6. Perform random affine transformations on each image (you should perform 14 total transformations - 2 for each image). Affine transformations can be translation, rotation, scaling, or shear as long as each is unique in either transformation type or transformation value (rotate 90 degrees vs rotate 186 degrees). No two images should be transformed in the exact same way. Save each of those images to new files.

def rotate_image(image, angle):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (width, height))

def shear_image(image, shear_matrix):
    height, width = image.shape[:2]
    pts1 = np.float32([[0, 0], [width, 0], [0, height]])
    pts2 = np.float32(shear_matrix) * np.float32([width, height])
    matrix = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(image, matrix, (width, height))

def scale_image(image, scale_factor):
    height, width = image.shape[:2]
    cx, cy = width / 2, height / 2
    matrix = np.float32([
        [scale_factor, 0, cx * (1 - scale_factor)],
        [0, scale_factor, cy * (1 - scale_factor)],
    ])
    interp = cv2.INTER_CUBIC if scale_factor >= 1.0 else cv2.INTER_AREA
    return cv2.warpAffine(image, matrix, (width, height), flags=interp)

def translate_image(image, translation):
    height, width = image.shape[:2]
    matrix = np.float32([[1, 0, translation[0]], [0, 1, translation[1]]])
    return cv2.warpAffine(image, matrix, (width, height))

transformations_dict = {
    '': [ # Original image
        [('rotate', 20.2), ('scale', 1.3), ('translate', (50, 30))],
        [('rotate', -15.6), ('shear', [[0, 0], [1, 0], [0.15, 1]])]
    ],
    'grayscale': [
        [('scale', 1.45), ('rotate', -22.3), ('translate', (15, -10)), ('shear', [[0, 0], [1, 0], [0.08, 1]])],
        [('translate', (25, -18)), ('rotate', 11.7)]
    ],
    'binary': [
        [('translate', (45, 35)), ('scale', 0.82)],
        [('rotate', 47.5), ('shear', [[0, 0], [1, -0.12], [0, 1]]), ('scale', 1.22), ('translate', (-20, 18))]
    ],
    'HSV': [
        [('shear', [[0, 0], [1, -0.27], [0, 1]]), ('translate', (-40, 22)), ('rotate', 8.9)],
        [('rotate', 63.4), ('scale', 1.18)]
    ],
    'CIELAB': [
        [('rotate', 90.0), ('translate', (32, -52)), ('scale', 1.05), ('shear', [[0, 0], [1, 0.06], [0, 1]])],
        [('scale', 0.73), ('shear', [[0, 0], [1, 0.21], [0, 1]])]
    ],
    'HLS': [
        [('scale', 0.78), ('shear', [[0, 0], [1, 0], [0.18, 1]])],
        [('rotate', -38.7), ('translate', (-28, 41)), ('scale', 0.91), ('shear', [[0, 0], [1, -0.05], [0.03, 1]])]
    ],
    'equalized': [
        [('rotate', 74.9), ('scale', 1.07), ('translate', (8, 14)), ('shear', [[0, 0], [1, 0.09], [0, 1]]), ('rotate', -5.2)],
        [('translate', (12, -58)), ('shear', [[0, 0], [1, 0], [0.24, 1]])]
    ],
}

for f in os.listdir('.'):
    if f.startswith('HW1_IMG_CS898BA') and f.endswith('.png'):
        image_type = f.split('.')[0].split('_')[-1]
        if image_type == 'CS898BA':  # Original image
            image_type = ''
        if image_type in transformations_dict:
            img = cv2.imread(f)
            for i, transformations in enumerate(transformations_dict[image_type]):
                transformed_img = img.copy()
                for (transformation, value) in transformations:
                    if transformation == 'rotate':
                        transformed_img = rotate_image(transformed_img, value)
                    elif transformation == 'shear':
                        transformed_img = shear_image(transformed_img, value)
                    elif transformation == 'scale':
                        transformed_img = scale_image(transformed_img, value)
                    elif transformation == 'translate':
                        transformed_img = translate_image(transformed_img, value)
                new_filename = f.replace('.png', f'_transformed_{i+1}.png')
                cv2.imwrite(new_filename, transformed_img)

# 7. You should now have 21 images.
print("Total images in directory after transformations:", len([f for f in os.listdir('.') if f.startswith('HW1_IMG_CS898BA') and f.endswith('.png')]))

# 8. Apply a Gaussian blur to each image using the levels of sigma: 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5. Discuss how the level of sigma changes the image. Save each of those images to new files.
sigma_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
for f in os.listdir('.'):
    if f.startswith('HW1_IMG_CS898BA') and f.endswith('.png') and 'blurred' not in f:
        img = cv2.imread(f)
        for sigma in sigma_values:
            blurred_img = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
            new_filename = f.replace('.png', f'_blurred_sigma{sigma}.png')
            cv2.imwrite(new_filename, blurred_img)

# Low sigma values like 0.5 and 1.0 helps in smoothing the images while preserving most of the details.
# As the sigma value increases, the images lose more and more details like edges and textures, which makes it harder to detect edges later on.

# 9. You should now have 168 images.
print("Total images in directory after blurring:", len([f for f in os.listdir('.') if f.startswith('HW1_IMG_CS898BA') and f.endswith('.png')]))

# Part 3:
# 1. Randomly create 4 equally sized subsets of the images from part 2.
all_images = [f for f in os.listdir('.') if f.startswith('HW1_IMG_CS898BA') and f.endswith('.png')]
rng.shuffle(all_images)
subset_size = len(all_images) // 4
subsets = [all_images[i:i + subset_size] for i in range(0, len(all_images), subset_size)]
for i, subset in enumerate(subsets):
    print(f"Subset {i+1}: {len(subset)} images")

