# AutoWordBug
Deep neural network based sentence classification attack.

## Requirments
Following libraries are required. Install them using your `pip`.
* numpy
* [pytorch](https://pytorch.org)

## NOTE: Before Start
Before start, all scripts (.py files at the root directory) offer `--help` option.
Please, see the usage message for more details which not explained in this document.

## Training
You can train `AutoWordBug` with the following steps.
First, you need to prepare the datasets.
The following command will create `train.pkl`, `val.pkl`, and `test.pkl` at `experiments` folder
```bash
$ python prepare.py
```
Then, you can train an `AutoWordBug` with the following command. If you want to utilize your GPU, type in `--cuda` option additionaly.
```bash
$ python train.py --train experiments/train.pkl --val experiments/val.pkl
# Training will take a while.
```
Now, you can find trained model at the `experiments` folder

## Evaluation
With trained model, you can evaluate the performance of `AutoWordBug`.
```bash
$ python evaluate.py --model experiments/AutoWordBug_best.pt --test experiments/test.pkl
# Generation result of each sentences will be print out.
```

## Maintainers
* [Dohyun Kim](https://github.com/dha8102)
* [Guhnoo Yun](https://github.com/DoranLyong)
* [Jihyun Kim](https://github.com/rabBit64)
* [Seokhyun Lee](https://github.com/HenryLee97)
