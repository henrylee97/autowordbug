from pathlib import Path
import csv
import numpy as np

def create_char_index(*data):
  chars = set()
  for d in data:
    for org, adv in d:
      for c in org:
        chars.add(c)
      for c in adv:
        chars.add(c)
  chars = list(chars)
  chars.sort()

  char_index = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2}
  for i, c in enumerate(chars, 3):
    char_index[c] = i

  return char_index

def create_index_char(char_index):
  return {i: c for c, i in char_index.items()}

def pad_sequences(sequences, maxlen=None, dtype='int32', value=0.):

  if maxlen is None:
    lengths = [len(x) for x in sequences]
    maxlen = np.max(lengths)

  num_samples = len(sequences)
  x = np.full((num_samples, maxlen), value, dtype=dtype)
  for idx, s in enumerate(sequences):
    if not len(s):
      continue  # empty list/array was found
    trunc = s[-maxlen:]
    x[idx, -len(trunc):] = trunc

  return x
