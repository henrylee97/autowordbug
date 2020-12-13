
def bleu(predictions, targets):
  score = []
  for pred, tar in zip(predictions, targets):
    # pred n grams
    pred_gram_1 = []
    pred_gram_2 = []
    pred_gram_3 = []
    pred_gram_4 = []

    for j in range(len(pred)):
      pred_gram_1.append(pred[j])
      if j < len(pred) - 1:
        pred_gram_2.append(pred[j] + pred[j + 1])
      if j < len(pred) - 2:
        pred_gram_3.append(pred[j] + pred[j + 2])
      if j < len(pred) - 3:
        pred_gram_4.append(pred[j] + pred[j + 3])

    # tar n grams
    tar_gram_1 = []
    tar_gram_2 = []
    tar_gram_3 = []
    tar_gram_4 = []

    for j in range(len(tar)):
      tar_gram_1.append(tar[j])
      if j < len(tar) - 1:
        tar_gram_2.append(tar[j] + tar[j + 1])
      if j < len(tar) - 2:
        tar_gram_3.append(tar[j] + tar[j + 2])
      if j < len(tar) - 3:
        tar_gram_4.append(tar[j] + tar[j + 3])

    # n gram matching
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

    bp = min((1, len(pred) / len(tar)))
    score.append(bp * precision)

  return sum(score) / len(score)
