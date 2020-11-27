from pathlib import Path
import csv
csv.field_size_limit(2147483647)

from .preprocess import make_word_index
from .preprocess import text_to_word_sequence
from .preprocess import pad_sequence

class Dataset:

  def __init__(self, data, n_words=20000, maxlen=None, word_index=None):
    data = Path(data)

    # Read inputs and outputs.
    self.inputs = []
    self.outputs = []
    self.len = 0
    with data.open(encoding='utf8') as f:
      reader = csv.reader(f)
      for row in reader:
        if not row:
          continue
        self.inputs.append(' '.join(row[1:]).lower())
        self.outputs.append(int(row[0]) - 1)
        self.len += 1

    # Create word to index dict.
    if word_index:
      self.word_index = word_index
    else:
      self.word_index = make_word_index(self.inputs, index_from=3)

    # Variables for preprocessing.
    self.n_words = n_words
    self.maxlen = maxlen
    self.start_index = 1
    self.oov_index = 2

  def __len__(self):
    return self.len

  def __getitem__(self, idx):

    # Preprocess input.
    i = self.inputs[idx]
    i = text_to_word_sequence(i)
    i = [self.start_index] + [self.word_index[w] if self.word_index[w] < self.n_words else self.oov_index for w in i]
    if self.maxlen:
      i = pad_sequence(i, maxlen=self.maxlen)

    # Get output.
    o = self.outputs[idx]

    return i, o
