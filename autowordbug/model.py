from pathlib import Path
import torch
import torch.nn as nn
import random

class Encoder(nn.Module):

  def __init__(self, n_chars, embed_dim, hidden_size, num_layers, dropout):
    super(Encoder, self).__init__()

    self.embeddings = nn.Embedding(n_chars, embed_dim)
    self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, dropout=dropout, bidirectional=True)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    # x: [maxlen, batch_size]
    x = self.embeddings(x)
    x = self.dropout(x)
    # x: [maxlen, batch_size, embed_dim]
    _, (hidden, cell) = self.lstm(x)
    return hidden, cell

class Decoder(nn.Module):

  def __init__(self, n_chars, embed_dim, hidden_size, num_layers, dropout):
    super(Decoder, self).__init__()

    self.embeddings = nn.Embedding(n_chars, embed_dim)
    self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, bidirectional=True)
    self.dropout = nn.Dropout(dropout)
    self.linear = nn.Linear(hidden_size * 2, n_chars)

  def forward(self, x, hidden, cell):
    # x: [batch_size]

    # x: [1, batch_size]
    x = x.unsqueeze(0)
    # x: [1, batch_size, embed_dim]
    x = self.embeddings(x)
    # output: [1, batch_size, hidden_size]
    output, (hidden, cell) = self.lstm(x, (hidden, cell))
    # output: [batch_size, hidden_size]
    output = output.squeeze(0)
    # pred: [batch_size, n_chars]
    pred = self.linear(output)
    return pred, hidden, cell

class AutoWordBug(nn.Module):
  
  def __init__(self, n_chars, embed_dim, hidden_size, num_layers, dropout):
    super(AutoWordBug, self).__init__()

    self.n_chars = n_chars
    self.embed_dim = embed_dim
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.dropout = dropout

    self.encoder = Encoder(n_chars, embed_dim, hidden_size, num_layers, dropout)
    self.decoder = Decoder(n_chars, embed_dim, hidden_size, num_layers, dropout)
    self.loss = nn.CrossEntropyLoss()

    def init_weights(m):
      if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    self.encoder.apply(init_weights)
    self.decoder.apply(init_weights)

  def forward(self, inp, tar=None, teaching_force=0):
    # inp: [maxlen, batch_size]
    # tar: [maxlen, batch_size]
    maxlen, batch_size = inp.shape

    hidden, cell = self.encoder(inp)

    # real_output: [maxlen, batch_size]
    real_output = torch.zeros_like(inp, dtype=torch.float32)
    # outputs: [maxlen, batch_size]
    outputs = torch.zeros_like(inp, dtype=torch.float32)
    # outputs: [maxlen, batch_size, n_chars]
    outputs = outputs.unsqueeze(-1).repeat(1, 1, self.n_chars)

    # dec_input: [batch_size]
    dec_input = torch.ones_like(inp[0])
    # real_output: [maxlen, batch_size]
    real_output[0, :] = dec_input
    for t in range(1, maxlen):
      # output: [batch_size, n_chars]
      output, hidden, cell = self.decoder(dec_input, hidden, cell)
      # outputs: [maxlen, batch_size, n_chars]
      outputs[t] = output

      # dec_input: [batch_size]
      dec_input = output.argmax(1)
      if tar is not None and random.random() < teaching_force:
        dec_input = tar[t, :]

      # real_output: [maxlen, batch_size]
      real_output[t, :] = dec_input

    # If target is not given
    if tar is None:
      return real_output

    # outputs: [maxlen * batch_size, n_chars]
    outputs = outputs.view(-1, self.n_chars)
    # tar: [maxlen *batch_size]
    tar = tar.reshape(-1)

    loss = self.loss(outputs, tar)
    return loss, real_output
    
  def save(self, path):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    # Create model dict.
    model_dict = {
      'args': [
        self.n_chars,
        self.embed_dim,
        self.hidden_size,
        self.num_layers,
        self.dropout
      ],
      'state': self.state_dict()
    }

    # Save.
    torch.save(model_dict, path)

  @classmethod
  def load(cls, path, device='cpu'):
    path = Path(path)

    # Load args and settings.
    model_dict = torch.load(path, map_location=device)

    # Create and load a trained model.
    model = cls(*model_dict['args'])
    model.load_state_dict(model_dict['state'])

    return model
