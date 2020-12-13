
def make_sentences(batch, index_char):
  sentences = []
  for indices in batch:
    sent = []
    for i in indices:
      c = index_char[i.item()]
      if c == '<SOS>' or c == '<PAD>':
        continue
      elif c == '<EOS>':
        break
      sent.append(c)
    sent = ''.join(sent)
    sentences.append(sent)
  return sentences
