import dill
import numpy as np
import torch
import time, datetime

from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from dataloader import create_dataset, Mycollator
from main_model import Main_model
from config import Configuration
from scoring import bleu_score
import pandas as pd
import torch.nn as nn

cfg = Configuration()


# 시간 측정 함수
def format_time(elapsed):
    # Round to the nearest second.
    elapsed_rounded = int(round(elapsed))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


if __name__ == "__main__":
    print("Adversarial sample evaluation")
    print("==============configuration==============")
    cfg.configprint()

    with open("chardict.dill", "rb") as f:
        chardict, chardict_inv = dill.load(f)
    with open("./exp_settings/" + cfg.mode + "/eval_sentences.dill", "rb") as f:
        eval_sent = dill.load(f)

    chars_len = len(chardict)

    eval_size = len(eval_sent.orig_sent)
    # eval_size = len(eval_sent.orig_sent)

    device = torch.device(cfg.device)

    eval_dataset = create_dataset("eval")
    collate_fn = Mycollator()
    eval_dataloader = DataLoader(eval_dataset, batch_size=cfg.lc_batch_size,
                                 shuffle=False, collate_fn=collate_fn,
                                 drop_last=False)

    model = torch.load(cfg.save_path + '_RNNcombine_model_epoch68.pt', map_location=cfg.device)
    checkpoint = torch.load(cfg.save_path + 'combine_RNN_model_state_dict68.tar', map_location=cfg.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # ========================================
    #               Evaluation
    # ========================================
    print('Evaluation...')
    t0 = time.time()
    total_loss = 0
    model.eval()

    orig_sent = []
    deepwordbug_sent = []
    our_sent = []

    for batch_idx, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            loss, outputs = model(batch, "eval")
            total_loss += loss.item()

        orig_sent.append('')
        deepwordbug_sent.append('')
        our_sent.append('')
        # print("prediction:", end="")
        # outputs = [batch_size, maxlen]
        # print("Batch:") # ([maxlen, batch_size], [maxlen, batch_size])
        for i in range(batch[0].shape[1]):
            for j in range(len(outputs[i])):
                # print(chardict_inv[outputs[i][j].item()], end="")

                if chardict_inv[outputs[i][j].item()] == '<EOS>':
                    break
                if j != 0:
                    our_sent[-1] += chardict_inv[outputs[i][j].item()]
            # print("\ntarget:", end="")
            for j in range(batch[1].shape[0]):
                # print(chardict_inv[batch[1][j][i].item()], end="")
                if chardict_inv[batch[1][j][i].item()] == '<EOS>':
                    break
                if j != 0:
                    deepwordbug_sent[-1] += chardict_inv[batch[1][j][i].item()]
            for j in range(batch[0].shape[0]):
                if chardict_inv[batch[0][j][i].item()] == '<PAD>':
                    break
                orig_sent[-1] += chardict_inv[batch[0][j][i].item()]

            # print("\n------------------------------------")

        if batch_idx % 10 == 0 and not batch_idx == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {0:>5,}  of  {1:>5,}.    Elapsed: {2:}.    Loss: {3:>5.5f}.   '
                  .format(batch_idx, len(eval_dataloader), elapsed, loss))

    print("  Evaluation took: {:}".format(format_time(time.time() - t0)))
    print("  Average Evaluation loss: {0:.4f}".format(total_loss / len(eval_dataloader)))

    print("evaluation finished.")
    print("==============configuration==============")
    cfg.configprint()

    # Saving sentences
    for i in range(len(deepwordbug_sent)):
        print("orig:",orig_sent[i])
        print("deep:",deepwordbug_sent[i])
        print("our:",our_sent[i])
        print("\n------------------------------------")
    data = {'Original_sentence': orig_sent,
            'Deepwordbug_sentence': deepwordbug_sent, 'Our_sentence': our_sent}
    data = pd.DataFrame(data)
    data.to_excel(excel_writer='RNN_sentences_on_test_dataset.xlsx')


    # calculating bleu score
    deep_vs_origin = bleu_score(deepwordbug_sent, orig_sent) / len(deepwordbug_sent)
    our_vs_origin = bleu_score(our_sent, orig_sent) / len(deepwordbug_sent)
    print(deep_vs_origin, our_vs_origin)