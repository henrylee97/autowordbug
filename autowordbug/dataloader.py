from pathlib import Path
from torch.utils.data import Dataset
import csv
import pickle as pkl
import random
import torch

from .preprocess import create_char_index
from .preprocess import pad_sequences

class AdversarialDataset(Dataset):

  def __init__(self, data, char_index):
    self.data = data
    self.len = len(data)
    self.char_index = char_index

  def __len__(self):
    return self.len
    
  def __getitem__(self, idx):
    org, adv = self.data[idx]
    org = [self.char_index[c] for c in org]
    adv = [1] + [self.char_index[c] for c in adv] + [2]
    return org, adv

  def save(self, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f:
      pkl.dump({
        'data': self.data,
        'char_index': self.char_index
      }, f)

  @classmethod
  def load(cls, path):
    path = Path(path)
    with path.open('rb') as f:
      d = pkl.load(f)
    return cls(**d)


def create_dataset(path, train=0.8, val=0.1, test=0.1):

  s = train + val + test
  train = train / s
  val = train + val / s
  test = val + test / s

  path = Path(path)
  train_data = []
  val_data = []
  test_data = []
  
  for p in path.glob('*.csv'):
    with p.open() as f:
      for o, _, a, _ in csv.reader(f):
        rand = random.random()
        if rand < train:
          train_data.append((o, a))
        elif rand < val:
          val_data.append((o, a))
        else: # rand < test
          test_data.append((o, a))

  char_index = create_char_index(train_data, val_data, test_data)
  
  train_dataset = AdversarialDataset(train_data, char_index) if len(train_data) > 0 else None
  val_dataset = AdversarialDataset(val_data, char_index) if len(val_data) > 0 else None
  test_dataset = AdversarialDataset(test_data, char_index) if len(test_data) > 0 else None
  return train_dataset, val_dataset, test_dataset

class CollateFN:

  def __init__(self, device):
    self.device = device

  def __call__(self, samples):
    org, adv = list(zip(*samples))
    maxlen = max(map(len, [*org, *adv]))
    org = torch.tensor(pad_sequences(org, maxlen=maxlen)).long()
    adv = torch.tensor(pad_sequences(org, maxlen=maxlen)).long()
    return org.transpose(0, 1).to(self.device), adv.transpose(0, 1).to(self.device)
