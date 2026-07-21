from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_IMAGE_PATH = PROJECT_DIR / "HW1_IMG_CS898BA.png"
OUTPUT_DIR = PROJECT_DIR / "advanced_segmentation_outputs"
PLOTS_DIR = PROJECT_DIR / "plots"
CHANNEL_A_PATH = OUTPUT_DIR / "HW1_IMG_CS898BA_channel_a_original_rgb.png"
CHANNEL_B_PATH = OUTPUT_DIR / "HW1_IMG_CS898BA_channel_b_hsv_value_normalized_rgb.png"
CHANNEL_C_PATH = OUTPUT_DIR / "HW1_IMG_CS898BA_channel_c_rgb_channel_normalized.png"
SEMANTIC_MAP_PATH = OUTPUT_DIR / "HW1_IMG_CS898BA_channel_a_segformer_semantic_map.png"
SEMANTIC_OVERLAY_PATH = OUTPUT_DIR / "HW1_IMG_CS898BA_channel_a_segformer_semantic_overlay.png"
SEMANTIC_LEGEND_PATH = OUTPUT_DIR / "HW1_IMG_CS898BA_channel_a_segformer_semantic_legend.png"
COMPARISON_PLOT_PATH = PLOTS_DIR / "EC_advanced_segmentation_comparison.png"
GROUND_TRUTH_PATH = (
	PROJECT_DIR
	/ "CVAT segmentation"
	/ "SegmentationClass"
	/ "HW1_IMG_CS898BA.png"
)
MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
MODEL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set this to a label from the diagnostic legend after visual inspection.
# None saves the Channel A diagnostic image and legend without calculating IoU/Dice.
TARGET_LABEL_NAME = "person"

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


# Part 3: Evaluation Outputs
ground_truth_mask = cv2.imread(str(GROUND_TRUTH_PATH), cv2.IMREAD_GRAYSCALE)
if ground_truth_mask is None:
	raise FileNotFoundError(f"Unable to load HW2 ground-truth mask: {GROUND_TRUTH_PATH}")

ground_truth_binary = ground_truth_mask > 0
image_height, image_width = channel_a_original_rgb.shape[:2]
if ground_truth_binary.shape != (image_height, image_width):
	raise ValueError(
		"Ground-truth mask dimensions do not match the target image: "
		f"{ground_truth_binary.shape} != {(image_height, image_width)}"
	)

image_processor_config, _ = SegformerImageProcessor.get_image_processor_dict(MODEL_NAME)
image_processor_config.pop("feature_extractor_type", None)
image_processor_config["do_reduce_labels"] = image_processor_config.pop("reduce_labels", False)
image_processor = SegformerImageProcessor.from_dict(image_processor_config)
segmentation_model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME).to(MODEL_DEVICE)
segmentation_model.eval()

semantic_palette = np.zeros((segmentation_model.config.num_labels, 3), dtype=np.uint8)
for label_id in range(segmentation_model.config.num_labels):
	semantic_palette[label_id] = (
		64 + (47 * label_id) % 192,
		64 + (89 * label_id) % 192,
		64 + (131 * label_id) % 192,
	)

target_label_id = None
if TARGET_LABEL_NAME is not None:
	target_label_id = next(
		(
			int(label_id)
			for label_id, label_name in segmentation_model.config.id2label.items()
			if label_name.lower() == TARGET_LABEL_NAME.lower()
		),
		None,
	)
	if target_label_id is None:
		raise ValueError(f'"{TARGET_LABEL_NAME}" is not a class in {MODEL_NAME}.')

input_variants = {
	"channel_a": channel_a_original_rgb,
	"channel_b": channel_b_hsv_value_normalized_rgb,
	"channel_c": channel_c_rgb_channel_normalized,
}
predicted_masks = {}
segmentation_overlays = {}
segmentation_metrics = {}

for channel_name, rgb_image in input_variants.items():
	model_inputs = image_processor(images=rgb_image, return_tensors="pt")
	model_inputs = {
		input_name: input_tensor.to(MODEL_DEVICE)
		for input_name, input_tensor in model_inputs.items()
	}

	with torch.inference_mode():
		model_outputs = segmentation_model(**model_inputs)

	resized_logits = functional.interpolate(
		model_outputs.logits,
		size=(image_height, image_width),
		mode="bilinear",
		align_corners=False,
	)
	semantic_labels = resized_logits.argmax(dim=1)[0].cpu().numpy()

	if channel_name == "channel_a":
		detected_label_ids, pixel_counts = np.unique(semantic_labels, return_counts=True)
		semantic_map_rgb = semantic_palette[semantic_labels]
		semantic_overlay_rgb = cv2.addWeighted(rgb_image, 0.45, semantic_map_rgb, 0.55, 0)

		if not cv2.imwrite(
			str(SEMANTIC_MAP_PATH),
			cv2.cvtColor(semantic_map_rgb, cv2.COLOR_RGB2BGR),
		):
			raise OSError(f"Unable to save semantic map: {SEMANTIC_MAP_PATH}")
		if not cv2.imwrite(
			str(SEMANTIC_OVERLAY_PATH),
			cv2.cvtColor(semantic_overlay_rgb, cv2.COLOR_RGB2BGR),
		):
			raise OSError(f"Unable to save semantic overlay: {SEMANTIC_OVERLAY_PATH}")

		legend_row_height = 34
		legend_bgr = np.full(
			(20 + legend_row_height * len(detected_label_ids), 720, 3),
			255,
			dtype=np.uint8,
		)
		for row_index, (label_id, pixel_count) in enumerate(
			sorted(zip(detected_label_ids, pixel_counts), key=lambda item: int(item[0]))
		):
			label_id = int(label_id)
			y_position = 10 + row_index * legend_row_height
			label_color_bgr = tuple(int(color) for color in semantic_palette[label_id][::-1])
			cv2.rectangle(
				legend_bgr,
				(12, y_position),
				(38, y_position + 26),
				label_color_bgr,
				thickness=-1,
			)
			cv2.putText(
				legend_bgr,
				f"{label_id}: {segmentation_model.config.id2label[label_id]} ({pixel_count:,} pixels)",
				(50, y_position + 20),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.5,
				(0, 0, 0),
				1,
				cv2.LINE_AA,
			)
		if not cv2.imwrite(str(SEMANTIC_LEGEND_PATH), legend_bgr):
			raise OSError(f"Unable to save semantic legend: {SEMANTIC_LEGEND_PATH}")

		print("\nChannel A full semantic diagnostic saved:")
		print(f"  Map: {SEMANTIC_MAP_PATH.relative_to(PROJECT_DIR)}")
		print(f"  Overlay: {SEMANTIC_OVERLAY_PATH.relative_to(PROJECT_DIR)}")
		print(f"  Legend: {SEMANTIC_LEGEND_PATH.relative_to(PROJECT_DIR)}")
		print("Inspect the overlay and legend before changing TARGET_LABEL_NAME.")

		if target_label_id is None:
			break

	predicted_mask = semantic_labels == target_label_id
	predicted_masks[channel_name] = predicted_mask

	predicted_mask_uint8 = (predicted_mask.astype(np.uint8) * 255)
	mask_path = OUTPUT_DIR / f"HW1_IMG_CS898BA_{channel_name}_segformer_person_mask.png"
	if not cv2.imwrite(str(mask_path), predicted_mask_uint8):
		raise OSError(f"Unable to save predicted mask: {mask_path}")

	# Dim non-target pixels, fill the selected figure region orange, and draw a cyan boundary.
	overlay = (rgb_image.astype(np.float32) * 0.45).astype(np.uint8)
	overlay[predicted_mask] = (
		0.45 * rgb_image[predicted_mask] + 0.55 * np.array((255, 128, 0))
	).astype(np.uint8)
	boundary = cv2.morphologyEx(
		predicted_mask_uint8,
		cv2.MORPH_GRADIENT,
		np.ones((5, 5), dtype=np.uint8),
	) > 0
	overlay[boundary] = (0, 255, 255)
	segmentation_overlays[channel_name] = overlay

	overlay_path = OUTPUT_DIR / f"HW1_IMG_CS898BA_{channel_name}_segformer_overlay.png"
	if not cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)):
		raise OSError(f"Unable to save segmentation overlay: {overlay_path}")

	intersection = np.logical_and(predicted_mask, ground_truth_binary).sum()
	union = np.logical_or(predicted_mask, ground_truth_binary).sum()
	mask_total = predicted_mask.sum() + ground_truth_binary.sum()
	segmentation_metrics[channel_name] = {
		"IoU": intersection / union if union else 0.0,
		"Dice": (2 * intersection) / mask_total if mask_total else 0.0,
	}
if target_label_id is None:
	print("\nSet TARGET_LABEL_NAME to a label shown in the Channel A legend and run again.")
else:
	print(f"\nPart 3 SegFormer results using the '{TARGET_LABEL_NAME}' class:")
	for channel_name, metrics in segmentation_metrics.items():
		print(
			f"  {channel_name}: "
			f"IoU={metrics['IoU']:.4f}, Dice={metrics['Dice']:.4f}"
		)

	# Part 4: Evaluation and Analysis
	comparison_images = (
		("Original RGB", channel_a_original_rgb),
		("HSV V-Normalized", channel_b_hsv_value_normalized_rgb),
		("RGB Channel-Normalized", channel_c_rgb_channel_normalized),
		("HW2 Ground Truth", ground_truth_binary),
		("SegFormer Channel A Mask", predicted_masks["channel_a"]),
		("SegFormer Channel B Mask", predicted_masks["channel_b"]),
		("SegFormer Channel C Mask", predicted_masks["channel_c"]),
	)

	figure, axes = plt.subplots(2, 4, figsize=(20, 10))
	for axis, (title, image) in zip(axes.flat, comparison_images):
		if image.ndim == 2:
			axis.imshow(image, cmap="gray")
		else:
			axis.imshow(image)
		axis.set_title(title)
		axis.axis("off")

	axes.flat[-1].axis("off")
	figure.suptitle("Extra Credit: SegFormer Advanced Segmentation", fontsize=16)
	figure.tight_layout()
	PLOTS_DIR.mkdir(exist_ok=True)
	figure.savefig(COMPARISON_PLOT_PATH, dpi=150, bbox_inches="tight")
	plt.close(figure)

	print(f"Saved Part 4 comparison plot: {COMPARISON_PLOT_PATH.relative_to(PROJECT_DIR)}")
