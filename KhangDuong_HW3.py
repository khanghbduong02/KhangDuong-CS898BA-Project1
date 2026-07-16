import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

from sklearn.model_selection import train_test_split


SEED = 42
DATASET_DIR = Path(__file__).resolve().parent / "Fish" / "Fish"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
TARGET_SIZE = (128, 128)
BATCH_SIZE = 32


# Part 2: Data Preprocessing & Augmentation
images = []
labels = []

for class_dir in sorted(path for path in DATASET_DIR.iterdir() if path.is_dir()):
	for image_path in sorted(class_dir.iterdir()):
		if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
			image = cv2.imread(str(image_path))
			if image is None:
				raise ValueError(f"Unable to load image: {image_path}")
			image = cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_AREA)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			images.append(image.astype(np.float32) / 255.0)
			labels.append(class_dir.name)

class_names = sorted(set(labels))
label_to_index = {class_name: index for index, class_name in enumerate(class_names)}
images = np.asarray(images, dtype=np.float32)
labels = np.asarray([label_to_index[label] for label in labels], dtype=np.int32)

train_images, val_test_images, train_labels, val_test_labels = train_test_split(
	images,
	labels,
	test_size=0.30,
	stratify=labels,
	random_state=SEED,
)
val_images, test_images, val_labels, test_labels = train_test_split(
	val_test_images,
	val_test_labels,
	test_size=0.50,
	stratify=val_test_labels,
	random_state=SEED,
)

data_augmentation = tf.keras.Sequential(
	[
		tf.keras.layers.RandomFlip("horizontal", seed=SEED),
		tf.keras.layers.RandomRotation(0.03, fill_mode="reflect", seed=SEED),
		tf.keras.layers.RandomBrightness(0.15, value_range=(0, 1), seed=SEED),
	],
	name="data_augmentation",
)

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = train_dataset.shuffle(len(train_images), seed=SEED, reshuffle_each_iteration=True)
train_dataset = train_dataset.batch(BATCH_SIZE).map(
	lambda image_batch, label_batch: (data_augmentation(image_batch, training=True), label_batch),
	num_parallel_calls=tf.data.AUTOTUNE,
).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(BATCH_SIZE)
val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(BATCH_SIZE)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

print(f"Training images: {len(train_images)}")
print(f"Validation images: {len(val_images)}")
print(f"Test images: {len(test_images)}")

