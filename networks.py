import torch
from torch import nn


class BaselineCNN(nn.Module):
	def __init__(self, num_classes, image_size, input_channels=3):
		super().__init__()
		if isinstance(image_size, int):
			image_height = image_size
			image_width = image_size
		else:
			image_height, image_width = image_size

		if image_height < 8 or image_width < 8:
			raise ValueError("image_size must be at least 8 pixels in both dimensions.")

		self.features = nn.Sequential(
			nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2),
			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2),
			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2),
		)
		with torch.no_grad():
			feature_maps = self.features(torch.zeros(1, input_channels, image_height, image_width))
			flattened_features = feature_maps.flatten(start_dim=1).shape[1]

		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Linear(flattened_features, 256),
			nn.ReLU(inplace=True),
			nn.Dropout(0.5),
			nn.Linear(256, num_classes),
		)
		self.softmax = nn.Softmax(dim=1)

	def forward(self, images):
		return self.classifier(self.features(images))

	def predict_probabilities(self, images):
		return self.softmax(self(images))