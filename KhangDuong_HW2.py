import os

import cv2

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
