from ast import literal_eval
from pathlib import Path
from torch.utils.data import DataLoader
import argparse
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim

from src.data import Dataset
from src.data.preprocess import pad_sequences
import src.model as model_module

def make_args(args):
  return [literal_eval(arg) for arg in args]

def make_kwargs(kwargs):
  kwargs = [kwarg.split('=') for kwarg in kwargs]
  return {key: literal_eval(value) for key, value in kwargs}

def train_epoch(model, data_loader, optimizer, device):
  losses = []
  model.train()

  for inputs, targets in data_loader:
    inputs = inputs.long().to(device)
    targets = targets.to(device)

    outputs = model(inputs)
    loss = F.nll_loss(outputs, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.cpu().item())

  return sum(losses) / len(losses) if len(losses) > 0 else 0

def validate_accuracy(model, data_loader, device):
  correct = 0
  model.eval()
  for inputs, targets in data_loader:
    inputs = inputs.long().to(device)
    targets = targets.to(device)

    outputs = model(inputs)
    pred = outputs.data.max(1, keepdim=True)[1]

    correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
  return correct / len(data_loader.dataset)

def main(argv=None):

  if not argv:
    argv = sys.argv[1:]
  
  parser = argparse.ArgumentParser(description='Training script.')
  # Training data.
  parser.add_argument('-d', '--data', type=str, metavar='<.csv>', required=True, help='Training data')
  parser.add_argument('-v', '--validation', type=str, metavar='<.csv>', default=None, help='Validation data')
  parser.add_argument('-c', '--num-classes', type=int, metavar='<int>', required=True, help='Number of classes')
  parser.add_argument('--dictionary-size', type=int, default=20000, metavar='<int>', help='Number of words')
  parser.add_argument('--seq-len', type=int, default=500, metavar='<int>', help='Length of sequences to pad')
  parser.add_argument('--num-workers', type=int, default=1, metavar='<int>', help='Number of parellal nodes of preprocessing')
  # Model config.
  parser.add_argument('-m', '--model', type=str, default='WordRNN', metavar='<class>', help='Name of class in src.model')
  parser.add_argument('--args', type=str, nargs='*', default=[], metavar='<arg>', help='Arguments used when creating model')
  parser.add_argument('--kwargs', type=str, nargs='*', default=[], metavar='<key=arg>', help='Keyword arguments used when creating model')
  # Training config.
  parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
  parser.add_argument('--epochs', type=int, default=10, metavar='<int>', help='Number of epochs')
  parser.add_argument('--batch-size', type=int, default=128, metavar='<int>', help='Batch size')
  parser.add_argument('-lr', '--learning-rate', type=float, default=0.0005, metavar='<float>', help='Learning rate')
  parser.add_argument('-p', '--save-path', type=str, default='model.pt', metavar='<.pt>', help='Path for trained model')
  parser.add_argument('--best', action='store_true', help='Save the model with the best accuracy, validation data must provided')

  args = parser.parse_args(argv)

  # Device setting.
  device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

  # Create model.
  model_cls = getattr(model_module, args.model)
  model = model_cls(*make_args(args.args), **make_kwargs(args.kwargs))
  model.to(device)

  # Create optimizer.
  optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
  
  # Load training data.
  train_data = Dataset(args.data, args.dictionary_size, args.seq_len)
  train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

  # Load validation data.
  if args.validation:
    val_data = Dataset(args.validation, args.dictionary_size, args.seq_len, train_data.word_index)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

  # Best accuracy model.
  if args.best and args.validation:
    best_accuracy = 0.
    best_model = model_cls(*model.get_args())

  # Start training.
  for epoch in range(args.epochs):
    print(f'Start epoch {epoch + 1}', end='', flush=True)
    
    # Training.
    loss = train_epoch(model, train_loader, optimizer, device)

    # Validation.
    if args.validation:
      accuracy = validate_accuracy(model, val_loader, device)
      # Track the model with the best accuracy.
      if args.best and accuracy > best_accuracy:
        best_model.load_state_dict(model.state_dict())

    print(f'\rEpoch {epoch + 1} loss: {loss:6.4f}', f'  accuracy: {accuracy:6.4f}' if args.validation else '')

  # Save model.
  save_path = Path(args.save_path)
  save_path.parent.mkdir(parents=True, exist_ok=True)
  save_model = model
  if args.best and args.validation:
    save_model = best_model
  save_model.save(save_path, train_data.word_index)

if __name__ == '__main__':
  main()
