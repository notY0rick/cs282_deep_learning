import torch as th
from torch import nn
from torch.nn import functional as F
from torch import optim
import numpy as np

## NEED TO COPY THIS TO language_model.py

# Using a basic RNN/LSTM for Language modeling
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, rnn_size, num_layers=1, dropout=0):
        super().__init__()

        # Create an embedding layer of shape [vocab_size, rnn_size]
        # Use nn.Embedding
        # That will map each word in our vocab into a vector of rnn_size size.
        self.embedding = nn.Embedding(vocab_size, rnn_size, padding_idx=2)

        # Create an LSTM layer of rnn_size size. Use any features you wish.
        # We will be using batch_first convention
        self.lstm = nn.LSTM(rnn_size, 512, num_layers=num_layers, batch_first=True)
        # LSTM layer does not add dropout to the last hidden output.
        # Add this if you wish.
        self.dropout = nn.Dropout(p=dropout)
        # Use a dense layer to project the outputs of the RNN cell into logits of
        # the size of vocabulary (vocab_size).
        self.output = nn.Linear(512, vocab_size)

    def forward(self,x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)
        logits = self.output(lstm_out)
        return logits
