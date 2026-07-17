import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path

from networks import BaselineCNN
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


SEED = 42
DATASET_DIR = Path(__file__).resolve().parent / "Fish" / "Fish"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
TARGET_SIZE = (128, 128)
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = DEVICE.type == "cuda"
OUTPUT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = OUTPUT_DIR / "plots"
BASELINE_WEIGHTS_PATH = OUTPUT_DIR / "baseline_cnn_weights.pth"
BASELINE_PLOT_PATH = PLOTS_DIR / "HW3_baseline_training_curves.png"

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


# Part 3: Baseline CNN Architecture
baseline_model = BaselineCNN(
	num_classes=len(class_names),
	image_size=train_images.shape[1:3],
).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(baseline_model.parameters(), lr=LEARNING_RATE)


def run_epoch(data_loader, training=False):
	if training:
		baseline_model.train()
	else:
		baseline_model.eval()

	total_loss = 0.0
	total_correct = 0
	total_images = 0

	with torch.set_grad_enabled(training):
		for image_batch, label_batch in data_loader:
			image_batch = image_batch.to(DEVICE, non_blocking=PIN_MEMORY)
			label_batch = label_batch.to(DEVICE, non_blocking=PIN_MEMORY)

			if training:
				optimizer.zero_grad(set_to_none=True)

			logits = baseline_model(image_batch)
			loss = criterion(logits, label_batch)

			if training:
				loss.backward()
				optimizer.step()

			total_loss += loss.item() * label_batch.size(0)
			total_correct += (logits.argmax(dim=1) == label_batch).sum().item()
			total_images += label_batch.size(0)

	return total_loss / total_images, total_correct / total_images


PLOTS_DIR.mkdir(exist_ok=True)
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
best_validation_loss = float("inf")

for epoch in range(1, NUM_EPOCHS + 1):
	train_loss, train_accuracy = run_epoch(train_loader, training=True)
	val_loss, val_accuracy = run_epoch(val_loader)

	train_losses.append(train_loss)
	train_accuracies.append(train_accuracy)
	val_losses.append(val_loss)
	val_accuracies.append(val_accuracy)

	if val_loss < best_validation_loss:
		best_validation_loss = val_loss
		torch.save(
			{
				"model_state_dict": baseline_model.state_dict(),
				"class_names": class_names,
				"target_size": TARGET_SIZE,
				"epoch": epoch,
				"validation_loss": val_loss,
			},
			BASELINE_WEIGHTS_PATH,
		)

	print(
		f"Epoch {epoch:02d}/{NUM_EPOCHS} - "
		f"train loss: {train_loss:.4f}, train accuracy: {train_accuracy:.2%} - "
		f"validation loss: {val_loss:.4f}, validation accuracy: {val_accuracy:.2%}"
	)

baseline_checkpoint = torch.load(BASELINE_WEIGHTS_PATH, map_location=DEVICE, weights_only=True)
baseline_model.load_state_dict(baseline_checkpoint["model_state_dict"])
baseline_model.eval()

epochs = range(1, NUM_EPOCHS + 1)
figure, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(epochs, train_losses, label="Training")
axes[0].plot(epochs, val_losses, label="Validation")
axes[0].set(title="Baseline CNN Loss", xlabel="Epoch", ylabel="Cross-Entropy Loss")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(epochs, train_accuracies, label="Training")
axes[1].plot(epochs, val_accuracies, label="Validation")
axes[1].set(title="Baseline CNN Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
axes[1].legend()
axes[1].grid(alpha=0.3)

figure.tight_layout()
figure.savefig(BASELINE_PLOT_PATH, dpi=150)
plt.close(figure)

print(f"Saved baseline weights: {BASELINE_WEIGHTS_PATH.name}")
print(f"Best baseline epoch: {baseline_checkpoint['epoch']}")
print(f"Saved training curves: {BASELINE_PLOT_PATH.relative_to(OUTPUT_DIR)}")

