from datetime import datetime
from pathlib import Path
from torch.optim import Adam
from torch.utils.data import DataLoader
import argparse
import sys
import torch
import torch.nn as nn

from autowordbug.dataloader import AdversarialDataset
from autowordbug.dataloader import CollateFN
from autowordbug.model import AutoWordBug
from autowordbug.score import bleu
from autowordbug.postprocess import make_sentences
from autowordbug.preprocess import create_index_char

def train_epoch(model, optimizer, dataloader, teaching_force, print_every=0):
  model.train()

  total_loss = 0

  for batch_idx, (inp, tar) in enumerate(dataloader):

    # Forward
    loss, _ = model(inp, tar=tar, teaching_force=teaching_force)
    
    # Back prop
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1)
    optimizer.step()
    total_loss += loss.item()

    if print_every > 0 and batch_idx % print_every == 0:
      print(f'  Batch {batch_idx + 1:>5} of {len(dataloader):>5}.\tLoss: {loss.item():>6.4f}.')
  return total_loss / len(dataloader)

def validation(model, dataloader, index_char, print_every=0):
  model.eval()

  total_loss = 0
  total_bleu = 0

  for batch_idx, (inp, tar) in enumerate(dataloader):

    # Get validation loss
    with torch.no_grad():
      loss, output = model(inp, tar=tar)
    total_loss += loss.item()

    # Get BLEU score
    output = output.transpose(0, 1)
    output = make_sentences(output, index_char)
    tar = tar.transpose(0, 1)
    tar = make_sentences(tar, index_char)
    bleu_score = bleu(output, tar)
    total_bleu += bleu_score

    if print_every > 0 and batch_idx % print_every == 0:
      print(f'  Batch {batch_idx + 1:>5} of {len(dataloader):>5}.\tLoss: {loss.item():>6.4f}.\tBLEU: {bleu_score}')
  return total_loss / len(dataloader), total_bleu / len(dataloader)

def main(argv):
  parser = argparse.ArgumentParser('AutoWordBug training script')
  parser.add_argument('-t', '--train', required=True, type=Path, metavar='<.pkl>', help='Training set')
  parser.add_argument('-v', '--val', type=Path, metavar='<.pkl>', default=None, help='Validation set')

  parser.add_argument('--cuda', action='store_true', help='Use GPU if possible')
  parser.add_argument('--embed-dim', metavar='<int>', type=int, default=300, help='Embedding dimension (default: 300)')
  parser.add_argument('--hidden', metavar='<int>', type=int, default=500, help='Hidden size (default: 500)')
  parser.add_argument('--num-layers', metavar='<int>', type=int, default=4, help='Number of layers (default: 4)')
  parser.add_argument('--teaching-force', metavar='<float>', type=float, default=0.5, help='Portion of teaching forch (default: 0.5)')
  parser.add_argument('--dropout', metavar='<float>', type=float, default=0.1, help='Dropout portion (default: 0.1)')

  parser.add_argument('--batch-size', metavar='<int>', type=int, default=20, help='Batch size (default: 20)')
  parser.add_argument('--epoch', metavar='<int>', type=int, default=70, help='Epochs (default: 70)')
  parser.add_argument('--lr', metavar='<float>', type=float, default=3e-4, help='Learning rate (default: 3e-4)')
  parser.add_argument('--print-every', metavar='<int>', type=int, default=10, help='Print log every n batchs (defalut: 10)')

  parser.add_argument('--to', metavar='<dir>', type=Path, default=Path('experiments'), help='Directory to store trained model (default: experiments)')

  args = parser.parse_args(argv)

  # Device setting
  device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
  
  # Load dataset
  collate_fn = CollateFN(device)
  train_set = AdversarialDataset.load(args.train)
  train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=False)
  print(f'len(train_set): {len(train_set)}')
  if args.val:
    val_set = AdversarialDataset.load(args.val)
    index_char = create_index_char(val_set.char_index)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, drop_last=True)
    print(f'len(val_set): {len(val_set)}')

  # Create a model
  model = AutoWordBug(len(train_set.char_index), args.embed_dim, args.hidden, args.num_layers, args.dropout)
  model.to(device)
  
  # Create an optimizer
  optimizer = Adam(model.parameters(), lr=args.lr)

  # max bleu score
  max_bleu = 0

  for i in range(args.epoch):
    print()
    print(f'========== Epoch {i + 1} / {args.epoch} ==========')
    
    # Training
    print('Training starts.')
    training_start = datetime.now()
    loss = train_epoch(model, optimizer, train_loader, args.teaching_force, args.print_every)
    print(f'Time elapsed: {datetime.now() - training_start}')
    print(f'Average training loss: {loss:6.4f}')

    # Validation
    if args.val:
      print()
      print('Validation starts.')
      val_start = datetime.now()
      loss, bleu_score = validation(model, val_loader, index_char, args.print_every)
      print(f'Time elapsed: {datetime.now() - val_start}')
      print(f'Average validation loss: {loss:6.4f}')
      print(f'Average BLEU score: {bleu_score:6.4f}')

      # If bleu score get better
      if max_bleu < bleu_score:
        max_bleu = bleu_score
        model.save(args.to / 'AutoWordBug_best.pt')
        print(f'Best model saved at {args.to / "AutoWordBug_best.pt"}')
    
    # Save model
    model.save(args.to / f'AutoWordBug_epoch_{i + 1}.pt')

if __name__ == "__main__":
  main(sys.argv[1:])
