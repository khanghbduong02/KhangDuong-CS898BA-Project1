import cv2
import numpy as np
import torch
from pathlib import Path

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


SEED = 42
DATASET_DIR = Path(__file__).resolve().parent / "Fish" / "Fish"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
TARGET_SIZE = (128, 128)
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = DEVICE.type == "cuda"

torch.manual_seed(SEED)
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(SEED)


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

class FishDataset(Dataset):
	def __init__(self, images, labels, transform=None):
		self.images = torch.from_numpy(images).permute(0, 3, 1, 2)
		self.labels = torch.from_numpy(labels).long()
		self.transform = transform

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		image = self.images[index]
		if self.transform:
			image = self.transform(image)
		return image, self.labels[index]


train_transform = transforms.Compose(
	[
		transforms.RandomHorizontalFlip(),
		transforms.RandomRotation(10, interpolation=InterpolationMode.BILINEAR),
		transforms.ColorJitter(brightness=0.15),
	]
)

train_dataset = FishDataset(train_images, train_labels, transform=train_transform)
val_dataset = FishDataset(val_images, val_labels)
test_dataset = FishDataset(test_images, test_labels)

data_loader_generator = torch.Generator().manual_seed(SEED)
train_loader = DataLoader(
	train_dataset,
	batch_size=BATCH_SIZE,
	shuffle=True,
	generator=data_loader_generator,
	num_workers=0,
	pin_memory=PIN_MEMORY,
)
val_loader = DataLoader(
	val_dataset,
	batch_size=BATCH_SIZE,
	shuffle=False,
	num_workers=0,
	pin_memory=PIN_MEMORY,
)
test_loader = DataLoader(
	test_dataset,
	batch_size=BATCH_SIZE,
	shuffle=False,
	num_workers=0,
	pin_memory=PIN_MEMORY,
)

print(f"Training device: {DEVICE}")
print(f"Training images: {len(train_images)}")
print(f"Validation images: {len(val_images)}")
print(f"Test images: {len(test_images)}")

