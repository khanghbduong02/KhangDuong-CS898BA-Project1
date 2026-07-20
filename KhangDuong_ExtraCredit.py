from pathlib import Path

import cv2


PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_IMAGE_PATH = PROJECT_DIR / "HW1_IMG_CS898BA.png"
OUTPUT_DIR = PROJECT_DIR / "advanced_segmentation_outputs"
CHANNEL_A_PATH = OUTPUT_DIR / "HW1_IMG_CS898BA_channel_a_original_rgb.png"
CHANNEL_B_PATH = OUTPUT_DIR / "HW1_IMG_CS898BA_channel_b_hsv_value_normalized_rgb.png"
CHANNEL_C_PATH = OUTPUT_DIR / "HW1_IMG_CS898BA_channel_c_rgb_channel_normalized.png"

# Part 2: Experimental Requirements & Pipelines
original_bgr = cv2.imread(str(SOURCE_IMAGE_PATH), cv2.IMREAD_COLOR)
if original_bgr is None:
	raise FileNotFoundError(f"Unable to load target image: {SOURCE_IMAGE_PATH}")

# Channel A: original, unmodified RGB image.
channel_a_original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

# Channel B: preserve hue and saturation while histogram-equalizing V in HSV.
hsv_image = cv2.cvtColor(channel_a_original_rgb, cv2.COLOR_RGB2HSV)
hue, saturation, value = cv2.split(hsv_image)
normalized_value = cv2.equalizeHist(value)
channel_b_hsv_value_normalized_rgb = cv2.cvtColor(
	cv2.merge((hue, saturation, normalized_value)),
	cv2.COLOR_HSV2RGB,
)

# Channel C: histogram-equalize red, green, and blue independently.
red, green, blue = cv2.split(channel_a_original_rgb)
channel_c_rgb_channel_normalized = cv2.merge(
	(
		cv2.equalizeHist(red),
		cv2.equalizeHist(green),
		cv2.equalizeHist(blue),
	)
)

# Save BGR copies because OpenCV's imwrite expects BGR channel order.
OUTPUT_DIR.mkdir(exist_ok=True)
if not cv2.imwrite(str(CHANNEL_A_PATH), cv2.cvtColor(channel_a_original_rgb, cv2.COLOR_RGB2BGR)):
	raise OSError(f"Unable to save prepared image: {CHANNEL_A_PATH}")
if not cv2.imwrite(
	str(CHANNEL_B_PATH),
	cv2.cvtColor(channel_b_hsv_value_normalized_rgb, cv2.COLOR_RGB2BGR),
):
	raise OSError(f"Unable to save prepared image: {CHANNEL_B_PATH}")
if not cv2.imwrite(
	str(CHANNEL_C_PATH),
	cv2.cvtColor(channel_c_rgb_channel_normalized, cv2.COLOR_RGB2BGR),
):
	raise OSError(f"Unable to save prepared image: {CHANNEL_C_PATH}")

print("Part 2 input preparation complete:")
print(f"  Channel A: {CHANNEL_A_PATH.relative_to(PROJECT_DIR)}")
print(f"  Channel B: {CHANNEL_B_PATH.relative_to(PROJECT_DIR)}")
print(f"  Channel C: {CHANNEL_C_PATH.relative_to(PROJECT_DIR)}")
