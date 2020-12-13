import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Model

class WordRNN(Model):

  def __init__(self, classes=4, bidirectional=False, layer_num=1, length=20000, embedding_size=100, hidden_size=100):
    super(WordRNN, self).__init__()

    self.args = [classes, bidirectional, layer_num, length, embedding_size, hidden_size]

    # Layers.
    self.embd = nn.Embedding(length, embedding_size)
    self.lstm = nn.LSTM(embedding_size, hidden_size, layer_num, bidirectional=bidirectional)
    self.hidden_size = hidden_size
    num_directions = 1 + bidirectional
    self.n_states = num_directions * layer_num
    self.linear = nn.Linear(hidden_size * num_directions, classes)

  def forward(self, x):

    # Get embedding.
    x = self.embd(x)

    # Forward LSTM.
    h0 = torch.zeros(self.n_states, x.size(0), self.hidden_size)
    c0 = torch.zeros(self.n_states, x.size(0), self.hidden_size)
    x, _ = self.lstm(x.transpose(0, 1), (h0, c0))
    x = x[-1]

    # Classification.
    x = self.linear(x)
    return F.log_softmax(x, dim=1)

  def get_args(self):
    return self.args
