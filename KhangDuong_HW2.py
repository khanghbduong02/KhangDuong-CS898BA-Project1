import cv2

# Part 2:
# 1. Multi-Channel Color Normalization
image = cv2.imread('HW1_IMG_CS898BA.png')

blue, green, red = cv2.split(image)

equalized_blue = cv2.equalizeHist(blue)
equalized_green = cv2.equalizeHist(green)
equalized_red = cv2.equalizeHist(red)
equalized_image = cv2.merge((equalized_blue, equalized_green, equalized_red))

cv2.imwrite('HW1_IMG_CS898BA_normalized.png', equalized_image)