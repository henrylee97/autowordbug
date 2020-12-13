from pathlib import Path
from torch.utils.data import DataLoader
import argparse
import sys
import torch

from autowordbug.dataloader import AdversarialDataset
from autowordbug.dataloader import CollateFN
from autowordbug.model import AutoWordBug
from autowordbug.score import bleu
from autowordbug.postprocess import make_sentences
from autowordbug.preprocess import create_index_char
import autowordbug.base_models as base_models

def text_to_word_sequence(text, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', split=' '):
  translate_dict = dict((c, split) for c in filters)
  translate_map = str.maketrans(translate_dict)
  text = text.translate(translate_map)
  seq = text.split(split)
  return [i for i in seq if i]

def word_sequence_to_tensor(seq, word_index, n_words=20000, start_index=1, oov_index=2):
  seq = [start_index] + [word_index[w] if word_index[w] < n_words else oov_index for w in seq]
  return torch.tensor(seq).long()

def main(argv):
  parser = argparse.ArgumentParser(description='AutoWordBug evalutaion script')
  parser.add_argument('-t', '--test', required=True, type=Path, metavar='<.pkl>', help='Test set')
  parser.add_argument('-m', '--model', required=True, type=Path, metavar='<.pt>', help='Trained AutoWordBug')
  parser.add_argument('-b', '--base', type=str, default='WordRNN', metavar='<class>', help='Type of base model, see autowordbug/base_models (default: WordRNN)')
  parser.add_argument('-bp', '--base-path', type=Path, default=Path('base_models/ag_news/WordRNN.pt'), metavar='<.pt>', help='Trained base model')
  parser.add_argument('--cuda', action='store_true', help='Use GPU if possible')

  args = parser.parse_args(argv)

  print('========== Evaluation ==========')

  # Device setting
  device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
  
  # Load dataset
  collate_fn = CollateFN(device)
  test_set = AdversarialDataset.load(args.test)
  index_char = create_index_char(test_set.char_index)
  test_loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=collate_fn, drop_last=False)
  print(f'len(test_set): {len(test_set)}')

  # Load AutoWordBug
  model = AutoWordBug.load(args.model)
  model.to(device)
  model.eval()

  # Load base model
  base_cls = getattr(base_models, args.base)
  base, word_index = base_cls.load(args.base_path)
  base.to(device)
  base.eval()

  total_bleu_score = 0
  adversarials = 0

  for original, _ in test_loader:

    # Make generated sentences
    with torch.no_grad():
      generated = model(original)
    generated = generated.transpose(0, 1)
    generated = make_sentences(generated, index_char)

    # Make original sentences
    original = original.transpose(0, 1)
    original = make_sentences(original, index_char)

    # Calculate bleu score
    bleu_score = bleu(generated, original)
    
    # Find original label
    original = original[0]
    origial_seq = text_to_word_sequence(original)
    origial_seq = word_sequence_to_tensor(origial_seq, word_index)
    with torch.no_grad():
      orig_pred = base(origial_seq.unsqueeze(0))
    orig_pred = orig_pred.argmax(-1).item()

    # Find generated label
    generated = generated[0]
    generated_seq = text_to_word_sequence(generated)
    generated_seq = word_sequence_to_tensor(generated_seq, word_index)
    with torch.no_grad():
      gen_pred = base(generated_seq.unsqueeze(0))
    gen_pred = gen_pred.argmax(-1).item()

    # Track values
    total_bleu_score += bleu_score
    adversarials += 1 if orig_pred != gen_pred else 0

    # Loging
    print()
    print('Original:', original)
    print('  Label:', orig_pred)
    print('Genreated:', generated)    
    print('  Label:', gen_pred)
    print(f'BLEU: {bleu_score:6.4f}')

  print('============ Summary ===========')
  print(f'Average BLEU: {total_bleu_score / len(test_loader):6.4f}')
  print(f'Performance degradation: {adversarials / len(test_loader) * 100:5.2f}%')
  print()


if __name__ == "__main__":
  main(sys.argv[1:])
