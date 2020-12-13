import os
import dill
import csv
import random
from config import Configuration

cfg = Configuration()


class Sentence:
    def __init__(self):
        self.orig_sent = []
        self.orig_pred = []
        self.adv_sent = []
        self.adv_pred = []
        self.size = 0


def make_vocab():
    path = "./ai-security-project-main/data/ag_news/WordCNN/"
    ver = ['insert.csv', 'remove.csv', 'substitute.csv', 'swap.csv']
    chars = []
    for i in range(4):
        thispath = path + ver[i]
        with open(thispath, 'r', encoding='utf-8') as f:
            rdr = csv.reader(f)
            for line in rdr:
                for sent in line:
                    for thischar in sent:
                        if thischar not in chars:
                            chars.append(thischar)

    chars.sort()
    print("[chars used in the dataset]")
    print(chars)
    print(len(chars))
    chardict = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2}
    chardict_inv = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>'}
    charcnt = 3
    for thischar in chars:
        chardict[thischar] = charcnt
        chardict_inv[charcnt] = thischar
        charcnt += 1
    print(chardict)
    print(chardict_inv)
    return chardict, chardict_inv


def create_raw_data():
    train_sent = Sentence()
    val_sent = Sentence()
    eval_sent = Sentence()
    path = "./ai-security-project-main/data/ag_news/WordRNN/"
    ver = ['insert.csv', 'remove.csv', 'substitute.csv', 'swap.csv']
    # ver = ['swap.csv']
    for i in range(len(ver)):
        thispath = path + ver[i]
        with open(thispath, 'r', encoding='utf-8') as f:
            rdr = csv.reader(f)
            for line in rdr:
                rand = random.random()
                if rand < 0.8:
                    train_sent.orig_sent.append(line[0])
                    train_sent.orig_pred.append(int(line[1]))
                    train_sent.adv_sent.append(line[2])
                    train_sent.adv_pred.append(int(line[3]))
                elif 0.8 <= rand < 0.9:
                    val_sent.orig_sent.append(line[0])
                    val_sent.orig_pred.append(int(line[1]))
                    val_sent.adv_sent.append(line[2])
                    val_sent.adv_pred.append(int(line[3]))
                else:
                    eval_sent.orig_sent.append(line[0])
                    eval_sent.orig_pred.append(int(line[1]))
                    eval_sent.adv_sent.append(line[2])
                    eval_sent.adv_pred.append(int(line[3]))

    train_sent.size = len(train_sent.orig_pred)
    val_sent.size = len(val_sent.orig_pred)
    eval_sent.size = len(eval_sent.orig_pred)

    return train_sent, val_sent, eval_sent


if __name__ == "__main__":
    # 문장에 등장하는 문자들 vocab 만들기
    chardict, chardict_inv = make_vocab()
    train_sent, val_sent, eval_sent = create_raw_data()
    with open('RNN_chardict.dill', 'wb+') as f:
        dill.dump([chardict, chardict_inv], f)

    with open('./exp_settings/combine/RNN_train_sentences.dill', 'wb+') as f:
        dill.dump(train_sent, f)

    with open('./exp_settings/combine/RNN_val_sentences.dill', 'wb+') as f:
        dill.dump(val_sent, f)

    with open('./exp_settings/combine/RNN_eval_sentences.dill', 'wb+') as f:
        dill.dump(eval_sent, f)
    print("preprocess finished.")
