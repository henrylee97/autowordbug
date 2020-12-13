import torch
import dill
from torch.utils.data import Dataset
from config import Configuration

cfg = Configuration()


class Mycollator:
    """
    Mycollator 역할 정리
    1. device gpu로 옮김
    2. 각 batch sequence별로 maxlen 구해서 그만큼 패딩
    """

    def __init__(self):
        pass

    def __call__(self, samples):
        device = cfg.device
        maxlen = 0
        for sample in samples:
            if len(sample["inp_ind"]) > maxlen:
                maxlen = len(sample["inp_ind"])
            if len(sample["tar_ind"]) > maxlen:
                maxlen = len(sample["tar_ind"])
        maxlen += 2
        inp_chars = torch.zeros(cfg.lc_batch_size, maxlen).long()
        tar_chars = torch.zeros(cfg.lc_batch_size, maxlen).long()

        for i, sample in enumerate(samples):
            for j in range(len(sample['inp_ind'])):
                inp_chars[i][j] = sample['inp_ind'][j]
            tar_chars[i][0] = 1
            for j in range(1, len(sample['tar_ind']) + 1):
                tar_chars[i][j] = sample['tar_ind'][j - 1]
            tar_chars[i][len(sample['tar_ind']) + 1] = 2
        return inp_chars.transpose(0, 1).to(device), tar_chars.transpose(0, 1).to(device)


class create_dataset(Dataset):

    def __init__(self, mode):
        print("create " + mode + " dataset...")
        if mode == "train":
            with open("./exp_settings/"+cfg.mode+"/RNN_train_sentences.dill", "rb") as f:
                self.data = dill.load(f)
        elif mode == "val":
            with open("./exp_settings/"+cfg.mode+"/RNN_val_sentences.dill", "rb") as f:
                self.data = dill.load(f)
        else :
            with open("./exp_settings/"+cfg.mode+"/RNN_eval_sentences.dill", "rb") as f:
                self.data = dill.load(f)
        with open("RNN_chardict.dill", "rb") as f:
            self.chardict, self.chardict_inv = dill.load(f)

        '''
        Dataloader Input 정리
        1. input_sentences index.  ex) [25, 23, 666, 2, 44]
        2. target_sentences index.   ex) [25,23, 2, 666, 44]
        '''

    def __len__(self):
        return self.data.size

    def __getitem__(self, idx):
        inp_ind = []
        tar_ind = []
        for thischar in self.data.orig_sent[idx]:
            inp_ind.append(self.chardict[thischar])
        for thischar in self.data.adv_sent[idx]:
            tar_ind.append(self.chardict[thischar])

        item = {'inp_ind': inp_ind, 'tar_ind': tar_ind}
        return item
