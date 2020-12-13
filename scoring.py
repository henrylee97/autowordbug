import pickle
import pandas as pd


def bleu_score(pred, tar):
    bleu = 0
    for i in range(len(pred)):
        pred_gram_1 = []
        pred_gram_2 = []
        pred_gram_3 = []
        pred_gram_4 = []
        tar_gram_1 = []
        tar_gram_2 = []
        tar_gram_3 = []
        tar_gram_4 = []

        for j in range(len(pred[i])):
            pred_gram_1.append(pred[i][j])
            if j < len(pred[i]) - 1:
                pred_gram_2.append(pred[i][j] + pred[i][j + 1])
            if j < len(pred[i]) - 2:
                pred_gram_3.append(pred[i][j] + pred[i][j + 2])
            if j < len(pred[i]) - 3:
                pred_gram_4.append(pred[i][j] + pred[i][j + 3])

        for j in range(len(tar[i])):
            tar_gram_1.append(tar[i][j])
            if j < len(tar[i]) - 1:
                tar_gram_2.append(tar[i][j] + tar[i][j + 1])
            if j < len(tar[i]) - 2:
                tar_gram_3.append(tar[i][j] + tar[i][j + 2])
            if j < len(tar[i]) - 3:
                tar_gram_4.append(tar[i][j] + tar[i][j + 3])

        match_1gram = 0
        match_2gram = 0
        match_3gram = 0
        match_4gram = 0
        for j in range(len(tar_gram_1)):
            if tar_gram_1[j] in pred_gram_1:
                match_1gram += 1
        match_1gram /= len(tar_gram_1)
        for j in range(len(tar_gram_2)):
            if tar_gram_2[j] in pred_gram_2:
                match_2gram += 1
        match_2gram /= len(tar_gram_2)
        for j in range(len(tar_gram_3)):
            if tar_gram_3[j] in pred_gram_3:
                match_3gram += 1
        match_3gram /= len(tar_gram_3)
        for j in range(len(tar_gram_4)):
            if tar_gram_4[j] in pred_gram_4:
                match_4gram += 1
        match_4gram /= len(tar_gram_4)
        precision = (match_1gram * match_2gram * match_3gram * match_4gram) ** (1 / 4)

        bp = min((1, len(pred[i]) / len(tar[i])))
        bleu += bp * precision
    return bleu


if __name__ == "__main__":
    # print("scoring print and excel save...")
    with open('RNN_loss_and_bleu.pkl', 'rb') as f:
        train_loss, val_loss, val_bleu = pickle.load(f)

    epoch = [i + 1 for i in range(len(train_loss))]
    data = {'epoch': epoch, 'Training_loss': train_loss,
            'Validation_loss': val_loss, 'Validation Bleu:': val_bleu}
    data = pd.DataFrame(data)
    data.to_excel(excel_writer='RNN_loss and bleu.xlsx')
