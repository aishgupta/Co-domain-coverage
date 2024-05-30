import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LeNet"]

class LeNet(nn.Module):
	"""
	LeNet model for MNIST dataset
	"""
	def __init__(self, num_classes=10):
		super(LeNet, self).__init__()
		self.conv1 = nn.Conv2d(1, 6, (5,5), padding=2)
		self.conv2 = nn.Conv2d(6, 16, (5,5))
		self.fc1   = nn.Linear(16*5*5, 120)
		self.fc2   = nn.Linear(120, 84)
		self.fc3   = nn.Linear(84, num_classes)
		self.layer_names = ["conv1", "conv2", "fc1", "fc2", "fc3"]
		
	def forward(self, x, normalize=True):
		x = F.relu(self.conv1(x))
		x = F.max_pool2d(x, (2,2))
		x = F.relu(self.conv2(x))
		x = F.max_pool2d(x, (2,2))
		x = x.view(-1, self.num_flat_features(x))
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		
		return x

	def num_flat_features(self, x):
		size = x.size()[1:]
		num_features = 1
		for s in size:
			num_features *= s
		return num_features


