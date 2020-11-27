import torch.nn as nn
import torch.nn.functional as F

from .model import Model

class WordCNN(Model):
  def __init__(self, classes=4, num_features=100, dropout=0.5, length=20000):
    super(WordCNN, self).__init__()

    self.args = [classes, num_features, dropout, length]
    
    # Embedding layer.
    self.embd = nn.Embedding(length, num_features)

    # Convolution layers.
    self.conv1 = nn.Sequential(
      nn.Conv1d(num_features, 256, kernel_size=7, stride=1),
      nn.ReLU(),
      nn.MaxPool1d(kernel_size=3, stride=3)
    )
    self.conv2 = nn.Sequential(
      nn.Conv1d(256, 256, kernel_size=7, stride=1),
      nn.ReLU(),
      nn.MaxPool1d(kernel_size=3, stride=3)
    )
    self.conv3 = nn.Sequential(
      nn.Conv1d(256, 256, kernel_size=3, stride=1),
      nn.ReLU()
    )
    self.conv4 = nn.Sequential(
      nn.Conv1d(256, 256, kernel_size=3, stride=1),
      nn.ReLU()  
    )
    self.conv5 = nn.Sequential(
      nn.Conv1d(256, 256, kernel_size=3, stride=1),
      nn.ReLU()
    )
    self.conv6 = nn.Sequential(
      nn.Conv1d(256, 256, kernel_size=3, stride=1),
      nn.ReLU(),
      nn.MaxPool1d(kernel_size=3, stride=3)
    )      
    
    # Classification layers.
    self.fc1 = nn.Sequential(
      nn.Linear(3584, 1024),
      nn.ReLU(),
      nn.Dropout(p=dropout)
    )
    self.fc2 = nn.Sequential(
      nn.Linear(1024, 1024),
      nn.ReLU(),
      nn.Dropout(p=dropout)
    )
    self.fc3 = nn.Linear(1024, classes)

  def forward(self, x):

    # Get embedding.
    x = self.embd(x)
    x = x.transpose(1,2)

    # Forward convolutions.
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = self.conv5(x)
    x = self.conv6(x)
    x = x.view(x.size(0), -1)

    # Classification.
    x = self.fc1(x)
    x = self.fc2(x)
    x = self.fc3(x)
    return F.log_softmax(x, dim=1)

  def get_args(self):
    return self.args
