import dill
import numpy as np
import torch
import time, datetime
import pickle

from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from dataloader import create_dataset, Mycollator
from main_model import Main_model
from config import Configuration
from scoring import bleu_score
import torch.nn as nn

cfg = Configuration()


# 시간 측정 함수
def format_time(elapsed):
    # Round to the nearest second.
    elapsed_rounded = int(round(elapsed))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


if __name__ == "__main__":
    print("Adversarial sample training")
    print("==============configuration==============")
    cfg.configprint()
    print("★★★★RNN★★★★")

    with open("RNN_chardict.dill", "rb") as f:
        chardict, chardict_inv = dill.load(f)
    with open("./exp_settings/" + cfg.mode + "/RNN_train_sentences.dill", "rb") as f:
        train_sent = dill.load(f)
    with open("./exp_settings/" + cfg.mode + "/RNN_val_sentences.dill", "rb") as f:
        val_sent = dill.load(f)

    chars_len = len(chardict)
    train_size = len(train_sent.orig_sent)
    val_size = len(val_sent.orig_sent)
    print("train_size:", train_size)
    print("val_size:", val_size)

    # eval_size = len(eval_sent.orig_sent)

    device = torch.device(cfg.device)

    train_dataset = create_dataset("train")
    val_dataset = create_dataset("val")
    collate_fn = Mycollator()
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.lc_batch_size,
                                  shuffle=True, collate_fn=collate_fn,
                                  drop_last=False)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.lc_batch_size,
                                shuffle=False, collate_fn=collate_fn,
                                drop_last=True)

    model = Main_model(chars_len)
    optimizer = Adam(model.parameters(), lr=cfg.lc_learning_rate)
    criterion = nn.CrossEntropyLoss()
    max_bleu = 0

    train_loss = []
    val_loss = []
    val_bleu = []
    for epoch in range(cfg.lc_epoch):
        # ========================================
        #               Training
        # ========================================
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, cfg.lc_epoch))
        print('Training...')
        t0 = time.time()
        # 시간 측정을 위한 현재시간 저장
        total_loss = 0
        batch_loss = 0
        imsicnt = 0
        model.train()
        for batch_idx, batch in enumerate(train_dataloader):
            print("Batch:", batch[0].shape, batch[1].shape)
            print(len(train_dataloader))
            quit()
            optimizer.zero_grad()
            enc_hidden, enc_cell = model.encoder.init_hidden()
            dec_hidden, dec_cell = model.decoder.init_hidden()
            loss, outputs = model(batch, "train")
            # output = [batch size, maxlen]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 10 == 0 and not batch_idx == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {0:>5,}  of  {1:>5,}.    Elapsed: {2:}.    Loss: {3:>5.5f}.   '
                      .format(batch_idx, len(train_dataloader), elapsed, loss))
        print("  Training epoch {0:} took: {1:}".format(epoch + 1, format_time(time.time() - t0)))
        print("  Average training loss: {0:.4f}".format(total_loss / len(train_dataloader)))
        train_loss.append(total_loss / len(train_dataloader))
        # ========================================
        #               Validation
        # ========================================

        print('Validation...')
        t0 = time.time()
        total_loss = 0
        model.eval()
        print_pred = False
        total_bleu = 0
        for batch_idx, batch in enumerate(val_dataloader):
            prediction = []
            target = []
            with torch.no_grad():
                loss, outputs = model(batch, "val")
                total_loss += loss.item()
            # pred와 target 출력해보기
            if not print_pred:
                for i in range(5):
                    print("prediction:", end="")
                    for j in range(len(outputs[i])):
                        print(chardict_inv[outputs[i][j].item()], end="")
                        if chardict_inv[outputs[i][j].item()] == '<EOS>':
                            break
                    print("\ntarget:", end="")
                    for j in range(len(outputs[i])):
                        print(chardict_inv[batch[1][j][i].item()], end="")
                        if chardict_inv[batch[1][j][i].item()] == '<EOS>':
                            break
                    print("\n------------------------------------")
                print_pred = True
            # bleu score 계산을 위한 문장 저장

            prediction.append('')
            target.append('')
            for j in range(len(outputs[0])):
                if chardict_inv[outputs[0][j].item()] == '<EOS>':
                    break
                if chardict_inv[outputs[0][j].item()] == '<SOS>':
                    continue
                prediction[-1] += chardict_inv[outputs[0][j].item()]

            for j in range(len(outputs[1])):
                if chardict_inv[batch[1][j][1].item()] == '<EOS>':
                    break
                if chardict_inv[batch[1][j][1].item()] == '<SOS>':
                    continue
                target[-1] += chardict_inv[batch[1][j][1].item()]

            bleu = bleu_score(prediction, target)
            total_bleu += bleu

            if batch_idx % 10 == 0 and not batch_idx == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {0:>5,}  of  {1:>5,}.    Elapsed: {2:}.    Loss: {3:>5.5f}.   Bleu: {4:>5.5f}.'
                      .format(batch_idx, len(val_dataloader), elapsed, loss, bleu))

        print("  Validation took: {:}".format(format_time(time.time() - t0)))
        print("  Average Validation loss: {0:.4f}".format(total_loss / len(val_dataloader)))
        print("  Average Bleu score: {0:.4f}".format(total_bleu / len(val_dataloader)))
        val_loss.append(total_loss / len(val_dataloader))
        val_bleu.append(total_bleu / len(val_dataloader))

        if max_bleu < total_bleu:
            max_bleu = total_bleu
            print("highest bleu score. model save.")
            torch.save(model, cfg.save_path + '_RNN' + cfg.mode + '_model_epoch' + str(epoch+1) + '.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss
            }, cfg.save_path + cfg.mode + '_RNN' + '_model_state_dict' + str(epoch+1) + '.tar')

    print("Training finished.")
    print("==============configuration==============")
    cfg.configprint()
    print("★★★★RNN★★★★")
    print("train_size:", train_size)
    print("val_size:", val_size)
    with open('RNN_loss_and_bleu.pkl', 'wb+') as f:
        pickle.dump([train_loss, val_loss, val_bleu], f)
    print("[Loss and Bleu]")
    for i in range(len(train_loss)):
        print("Epoch {0} : Training loss : {1:>5.5f}   Validation loss : {2:>5.5f}   Bleu score : {3:>5.5f}"
              .format(i + 1, train_loss[i], val_loss[i], val_bleu[i]))


