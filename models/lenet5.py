"""
LeNet-5 for MNIST dataset.
Src: https://github.com/peikexin9/deepxplore/blob/master/MNIST/Model3.py

Note: Exactly similar to the LeNet model architecture in the mnist_models.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class lenet5(nn.Module):
    def __init__(self, pretrained=False):
        super(lenet5, self).__init__()
        self.model = nn.Sequential( nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2),
                                    nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2),
                                    nn.Flatten(),
                                    nn.Linear(16*5*5, 120),
                                    nn.ReLU(),
                                    nn.Linear(120, 84),
                                    nn.ReLU(),
                                    nn.Linear(84, 10)
                                )
        
    def forward(self, x):
        return self.model(x)

# model = lenet5()
# torch.save(model.state_dict(), "temp.pt")
# model.load_state_dict(torch.load("temp.pt"))