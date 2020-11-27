from collections import defaultdict
import numpy as np

def text_to_word_sequence(text, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', split=' '):
  translate_dict = dict((c, split) for c in filters)
  translate_map = str.maketrans(translate_dict)
  text = text.translate(translate_map)
  seq = text.split(split)
  return [i for i in seq if i]

def make_word_index(texts, index_from=1, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', split=' '):

  # Count words.
  word_counts = defaultdict(int)
  for text in texts:
    seq = text_to_word_sequence(text, filters, split)
    for w in seq:
      word_counts[w] += 1
  
  # Sort words.
  wcounts = list(word_counts.items())
  wcounts.sort(key=lambda x: x[1], reverse=True)
  sorted_voc = [wc[0] for wc in wcounts]

  return {w: i + index_from for i, w in enumerate(sorted_voc)}

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

def pad_sequence(sequence, maxlen=None, dtype='int32', value=0.):
  if maxlen is None:
    maxlen = len(sequence)
  x = np.full(maxlen, value, dtype=dtype)
  trunc = sequence[-maxlen:]
  x[-len(trunc):] = trunc
  return x