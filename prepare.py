from pathlib import Path
import argparse
import sys

from autowordbug.dataloader import create_dataset

def main(argv):
  parser = argparse.ArgumentParser('Data preparation script')
  parser.add_argument('-d', '--data', metavar='<dir>', type=Path, default=Path('data/ag_news/WordRNN'), help='Folder that contains csv files')
  parser.add_argument('--train', metavar='<float>', type=float, default=0.8, help='Portion of train set (default: 0.8)')
  parser.add_argument('--val', metavar='<float>', type=float, default=0.1, help='Portion of validation set (default: 0.1)')
  parser.add_argument('--test', metavar='<float>', type=float, default=0.1, help='Portion of test set (default: 0.1)')
  parser.add_argument('-t', '--to', metavar='<dir>', type=Path, default=Path('experiments'), help='Directory to save divided data (default: experiments)')
  args = parser.parse_args(argv)

  train, val, test = create_dataset(args.data, args.train, args.val, args.test)
  print('len(train):', len(train))
  train.save(args.to / 'train.pkl')
  print('len(val):', len(val))
  val.save(args.to / 'val.pkl')
  print('len(test):', len(test))
  test.save(args.to / 'test.pkl')

if __name__ == "__main__":
  main(sys.argv[1:])
