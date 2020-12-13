import torch
import torch.nn as nn
import random
from config import Configuration

cfg = Configuration()
device = cfg.device


class Encoder(nn.Module):
    def __init__(self, chars_len):
        super(Encoder, self).__init__()
        self.embeddings = nn.Embedding(num_embeddings=chars_len,
                                       embedding_dim=cfg.embed_dim).to(device)
        self.lstm = nn.LSTM(input_size=cfg.embed_dim,
                            hidden_size=cfg.hidden_size,
                            num_layers=cfg.num_layers,
                            dropout=cfg.dropout,
                            bidirectional=True).to(device)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, inp_chars):
        embedded = self.dropout(self.embeddings(inp_chars))  # [batch_size, batch_maxlen, embed_dim]
        outputs, (hidden, cell) = self.lstm(embedded)
        '''
        outputs = [src_len, batch_size, hidden_dim * n directions]
        hidden = [n_layers * n_directions, batch_size, hidden_dim]
        cell = [n_layers * n_directions, batch_size, hidden_dim]
        '''
        return hidden, cell

    def init_hidden(self):
        hidden = torch.zeros((2 * cfg.num_layers, cfg.lc_batch_size,
                              cfg.hidden_size)).to(device).requires_grad_(True)
        cell = torch.zeros((2 * cfg.num_layers, cfg.lc_batch_size,
                            cfg.hidden_size)).to(device).requires_grad_(True)

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, chars_len):
        super(Decoder, self).__init__()
        self.embeddings = nn.Embedding(num_embeddings=chars_len,
                                       embedding_dim=cfg.embed_dim).to(device)
        self.lstm = nn.LSTM(input_size=cfg.embed_dim,
                            hidden_size=cfg.hidden_size,
                            num_layers=cfg.num_layers,
                            bidirectional=True).to(device)
        self.dropout = nn.Dropout(cfg.dropout)
        self.linear = nn.Linear(cfg.hidden_size * 2, chars_len).to(device)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embeddings(input))
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        pred = self.linear(output.squeeze(0))
        return pred, hidden, cell

    def init_hidden(self):
        hidden = torch.zeros((2 * cfg.num_layers, cfg.lc_batch_size,
                              cfg.hidden_size)).to(device).requires_grad_(True)
        cell = torch.zeros((2 * cfg.num_layers, cfg.lc_batch_size,
                            cfg.hidden_size)).to(device).requires_grad_(True)

        return hidden, cell


class Main_model(nn.Module):
    def __init__(self, chars_len):
        super(Main_model, self).__init__()
        self.chars_len = chars_len
        self.encoder = Encoder(chars_len)
        self.decoder = Decoder(chars_len)

        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, batch, phase):
        inp_chars = batch[0]  # [maxlen, batch_size]
        tar_chars = batch[1]  # [maxlen, batch_size]
        maxlen = inp_chars.shape[0]
        outputs = torch.zeros(maxlen, cfg.lc_batch_size, self.chars_len).to(device)
        hidden, cell = self.encoder(inp_chars)
        # hidden: [2*num_layer, batch_size, hidden_size]
        # cell: [2*num_layer, batch_size, hidden_size]
        dec_input = tar_chars[0, :]  # [batch_size]
        real_output = torch.zeros(cfg.lc_batch_size, maxlen).to(device)
        real_output[:, 0] = 1
        for t in range(1, maxlen):
            output, hidden, cell = self.decoder(dec_input, hidden, cell)
            # output: [batch_size, chars_len]
            # hidden: [2*num_layer, batch_size, hidden_size]
            # cell: [2*num_layer, batch_size, hidden_size]
            outputs[t] = output

            top1 = output.argmax(1)  # [batch_size]
            if phase == "train":
                teacher_force = random.random() < cfg.teacher_forcing
                dec_input = tar_chars[t, :] if teacher_force else top1
            else:
                dec_input = top1

            real_output[:, t] = dec_input

        output_dim = outputs.shape[-1]

        final_output = outputs.view(-1, output_dim)  # [maxlen*batch_size, chars_len]
        trg = tar_chars.reshape(-1)
        loss = self.loss(final_output, trg)
        return loss, real_output
