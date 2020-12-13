import torch
import torch.nn as nn
from abc import ABC
from abc import abstractmethod
from pathlib import Path

class Model(nn.Module, ABC):

  # Must be implemented.
  @abstractmethod
  def get_args(self):
    return []

  def save(self, path, word_index):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    # Create model dict.
    model_dict = {
      'args': self.get_args(),
      'state': self.state_dict(),
      'word_index': word_index,
    }

    # Save.
    torch.save(model_dict, path)
  
  @classmethod
  def load(cls, path, device='cpu'):
    path = Path(path)

    # Load args and settings.
    model_dict = torch.load(path, map_location=device)

    # Create and load a trained model.
    model = cls(*model_dict['args'])
    model.load_state_dict(model_dict['state'])

    return model, model_dict['word_index']
