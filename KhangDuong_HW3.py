import logging
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import ray
import shutil
import tempfile
from pathlib import Path

from networks import BaselineCNN
from ray import tune
from ray.tune import Callback
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
TUNING_LEARNING_RATES = (0.01, 0.001, 0.0001)
TUNING_BATCH_SIZES = (32, 64)
TUNING_DROPOUT_RATES = (0.3, 0.5)
TUNING_TRIAL_COUNT = (
	len(TUNING_LEARNING_RATES) * len(TUNING_BATCH_SIZES) * len(TUNING_DROPOUT_RATES)
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = DEVICE.type == "cuda"
OUTPUT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = OUTPUT_DIR / "plots"
BASELINE_WEIGHTS_PATH = OUTPUT_DIR / "baseline_cnn_weights.pth"
BASELINE_PLOT_PATH = PLOTS_DIR / "HW3_baseline_training_curves.png"
OPTIMIZED_WEIGHTS_PATH = OUTPUT_DIR / "optimized_cnn_weights.pth"
RAY_RESULTS_DIR = OUTPUT_DIR / "ray_tune_results"
RAY_EXPERIMENT_NAME = "fish_cnn_tuning"

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


def run_epoch(data_loader, model, loss_function, training=False, model_optimizer=None, device=DEVICE):
	if training and model_optimizer is None:
		raise ValueError("Training requires an optimizer.")

	if training:
		model.train()
	else:
		model.eval()

	total_loss = 0.0
	total_correct = 0
	total_images = 0

	with torch.set_grad_enabled(training):
		for image_batch, label_batch in data_loader:
			image_batch = image_batch.to(device, non_blocking=device.type == "cuda")
			label_batch = label_batch.to(device, non_blocking=device.type == "cuda")

			if training:
				model_optimizer.zero_grad(set_to_none=True)

			logits = model(image_batch)
			loss = loss_function(logits, label_batch)

			if training:
				loss.backward()
				model_optimizer.step()

			total_loss += loss.item() * label_batch.size(0)
			total_correct += (logits.argmax(dim=1) == label_batch).sum().item()
			total_images += label_batch.size(0)

	return total_loss / total_images, total_correct / total_images


def copy_model_state(model):
	return {
		name: tensor.detach().cpu().clone()
		for name, tensor in model.state_dict().items()
	}


PLOTS_DIR.mkdir(exist_ok=True)
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
best_baseline_validation_loss = float("inf")
best_baseline_validation_accuracy = 0.0
best_baseline_epoch = 0
best_baseline_model_state = None

for epoch in range(1, NUM_EPOCHS + 1):
	train_loss, train_accuracy = run_epoch(
		train_loader,
		baseline_model,
		criterion,
		training=True,
		model_optimizer=optimizer,
	)
	val_loss, val_accuracy = run_epoch(val_loader, baseline_model, criterion)

	train_losses.append(train_loss)
	train_accuracies.append(train_accuracy)
	val_losses.append(val_loss)
	val_accuracies.append(val_accuracy)

	if val_loss < best_baseline_validation_loss:
		best_baseline_validation_loss = val_loss
		best_baseline_validation_accuracy = val_accuracy
		best_baseline_epoch = epoch
		best_baseline_model_state = copy_model_state(baseline_model)

	print(
		f"Epoch {epoch:02d}/{NUM_EPOCHS} - "
		f"train loss: {train_loss:.4f}, train accuracy: {train_accuracy:.2%} - "
		f"validation loss: {val_loss:.4f}, validation accuracy: {val_accuracy:.2%}"
	)

if best_baseline_model_state is None:
	raise RuntimeError("Baseline training did not produce a validation checkpoint.")

torch.save(
	{
		"model_state_dict": best_baseline_model_state,
		"class_names": class_names,
		"target_size": TARGET_SIZE,
		"training_epochs": NUM_EPOCHS,
		"best_epoch": best_baseline_epoch,
		"validation_loss": best_baseline_validation_loss,
		"validation_accuracy": best_baseline_validation_accuracy,
		"learning_rate": LEARNING_RATE,
		"batch_size": BATCH_SIZE,
		"dropout_rate": baseline_model.dropout_rate,
	},
	BASELINE_WEIGHTS_PATH,
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
print(
	f"Best baseline validation loss: {baseline_checkpoint['validation_loss']:.4f} "
	f"at epoch {baseline_checkpoint['best_epoch']}"
)
print(f"Saved training curves: {BASELINE_PLOT_PATH.relative_to(OUTPUT_DIR)}")

# Part 4: Hyperparameter Optimization
class TuneProgressLogger(Callback):
	COLUMN_WIDTHS = (7, 13, 10, 7, 10, 10, 13)
	HEADERS = (
		"Trial",
		"Learning Rate",
		"Batch Size",
		"Dropout",
		"Best Epoch",
		"Val. Loss",
		"Val. Accuracy",
	)

	def __init__(self):
		self.completed_trials = 0
		self.header_printed = False
		self.footer_printed = False

	def format_row(self, values):
		return "|" + "|".join(
			f" {str(value):>{width}} "
			for value, width in zip(values, self.COLUMN_WIDTHS)
		) + "|"

	def table_border(self):
		return "+" + "+".join("-" * (width + 2) for width in self.COLUMN_WIDTHS) + "+"

	def print_footer(self):
		if self.header_printed and not self.footer_printed:
			print(self.table_border(), flush=True)
			self.footer_printed = True

	def on_trial_complete(self, iteration, trials, trial, **info):
		result = trial.last_result
		config = trial.config
		self.completed_trials += 1

		if not self.header_printed:
			print("Ray Tune trial results:")
			print(self.table_border())
			print(self.format_row(self.HEADERS))
			print(self.table_border())
			self.header_printed = True

		print(
			self.format_row(
				(
					f"{self.completed_trials}/{TUNING_TRIAL_COUNT}",
					f"{config['learning_rate']:.4g}",
					config["batch_size"],
					f"{config['dropout_rate']:.1f}",
					result["best_epoch"],
					f"{result['validation_loss']:.4f}",
					f"{result['validation_accuracy']:.2%}",
				)
			),
			flush=True,
		)

		if self.completed_trials >= TUNING_TRIAL_COUNT:
			self.print_footer()


def tune_baseline_cnn(config, train_data, validation_data, class_names, image_size):
	torch.manual_seed(SEED)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(SEED)

	trial_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	trial_generator = torch.Generator().manual_seed(SEED)
	trial_train_loader = DataLoader(
		train_data,
		batch_size=config["batch_size"],
		shuffle=True,
		generator=trial_generator,
		num_workers=0,
		pin_memory=trial_device.type == "cuda",
	)
	trial_val_loader = DataLoader(
		validation_data,
		batch_size=config["batch_size"],
		shuffle=False,
		num_workers=0,
		pin_memory=trial_device.type == "cuda",
	)

	tuning_model = BaselineCNN(
		num_classes=len(class_names),
		image_size=image_size,
		dropout_rate=config["dropout_rate"],
	).to(trial_device)
	tuning_criterion = nn.CrossEntropyLoss()
	tuning_optimizer = optim.Adam(tuning_model.parameters(), lr=config["learning_rate"])
	trial_history = {
		"train_losses": [],
		"train_accuracies": [],
		"val_losses": [],
		"val_accuracies": [],
	}
	best_validation_loss = float("inf")
	best_validation_accuracy = 0.0
	best_epoch = 0
	best_model_state = None

	for epoch in range(1, NUM_EPOCHS + 1):
		train_loss, train_accuracy = run_epoch(
			trial_train_loader,
			tuning_model,
			tuning_criterion,
			training=True,
			model_optimizer=tuning_optimizer,
			device=trial_device,
		)
		val_loss, val_accuracy = run_epoch(
			trial_val_loader,
			tuning_model,
			tuning_criterion,
			device=trial_device,
		)

		trial_history["train_losses"].append(train_loss)
		trial_history["train_accuracies"].append(train_accuracy)
		trial_history["val_losses"].append(val_loss)
		trial_history["val_accuracies"].append(val_accuracy)

		if val_loss < best_validation_loss:
			best_validation_loss = val_loss
			best_validation_accuracy = val_accuracy
			best_epoch = epoch
			best_model_state = copy_model_state(tuning_model)

	if best_model_state is None:
		raise RuntimeError("Tuning trial did not produce a validation checkpoint.")

	checkpoint_data = {
		"model_state_dict": best_model_state,
		"class_names": class_names,
		"target_size": TARGET_SIZE,
		"training_epochs": NUM_EPOCHS,
		"best_epoch": best_epoch,
		"validation_loss": best_validation_loss,
		"validation_accuracy": best_validation_accuracy,
		"learning_rate": config["learning_rate"],
		"batch_size": config["batch_size"],
		"dropout_rate": config["dropout_rate"],
		"history": trial_history,
	}

	with tempfile.TemporaryDirectory() as checkpoint_directory:
		torch.save(checkpoint_data, Path(checkpoint_directory) / "model.pth")
		checkpoint = tune.Checkpoint.from_directory(checkpoint_directory)
		tune.report(
			{
				"validation_loss": best_validation_loss,
				"validation_accuracy": best_validation_accuracy,
				"training_epochs": NUM_EPOCHS,
				"best_epoch": best_epoch,
			},
			checkpoint=checkpoint,
		)


shutil.rmtree(RAY_RESULTS_DIR / RAY_EXPERIMENT_NAME, ignore_errors=True)
ray.init(
	include_dashboard=False,
	ignore_reinit_error=True,
	num_gpus=1 if DEVICE.type == "cuda" else 0,
	log_to_driver=False,
	logging_level=logging.ERROR,
)

tuning_resources = {"cpu": 1}
if DEVICE.type == "cuda":
	tuning_resources["gpu"] = 1

ray_tune_trainable = tune.with_parameters(
	tune_baseline_cnn,
	train_data=train_dataset,
	validation_data=val_dataset,
	class_names=class_names,
	image_size=tuple(train_images.shape[1:3]),
)

print(f"Ray Tune: evaluating {TUNING_TRIAL_COUNT} fixed-{NUM_EPOCHS}-epoch configurations.")
progress_logger = TuneProgressLogger()
tuner = tune.Tuner(
	tune.with_resources(ray_tune_trainable, tuning_resources),
	param_space={
		"learning_rate": tune.grid_search(TUNING_LEARNING_RATES),
		"batch_size": tune.grid_search(TUNING_BATCH_SIZES),
		"dropout_rate": tune.grid_search(TUNING_DROPOUT_RATES),
	},
	tune_config=tune.TuneConfig(
		metric="validation_loss",
		mode="min",
		max_concurrent_trials=1,
	),
	run_config=tune.RunConfig(
		name=RAY_EXPERIMENT_NAME,
		storage_path=str(RAY_RESULTS_DIR),
		verbose=0,
		log_to_file=True,
		callbacks=[progress_logger],
	),
)
tuning_results = tuner.fit()
best_tuning_result = tuning_results.get_best_result(metric="validation_loss", mode="min")

if best_tuning_result.checkpoint is None:
	raise RuntimeError("Ray Tune did not save a checkpoint for the best configuration.")

with best_tuning_result.checkpoint.as_directory() as checkpoint_directory:
	optimized_checkpoint = torch.load(
		Path(checkpoint_directory) / "model.pth",
		map_location=DEVICE,
		weights_only=True,
	)

torch.save(optimized_checkpoint, OPTIMIZED_WEIGHTS_PATH)
ray.shutdown()

print(f"Saved optimized weights: {OPTIMIZED_WEIGHTS_PATH.name}")
print(
	"Best optimized configuration - "
	f"learning rate: {optimized_checkpoint['learning_rate']}, "
	f"batch size: {optimized_checkpoint['batch_size']}, "
	f"dropout: {optimized_checkpoint['dropout_rate']}, "
	f"training epochs: {optimized_checkpoint['training_epochs']}, "
	f"best epoch: {optimized_checkpoint['best_epoch']}, "
	f"validation loss: {optimized_checkpoint['validation_loss']:.4f}, "
	f"validation accuracy: {optimized_checkpoint['validation_accuracy']:.2%}"
)
