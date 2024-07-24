import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch.nn as nn
import torch

class RNN(nn.Module):
    def __init__(self, embedding_dim=8, vocab_size=90, hidden_size=256):
        super(RNN, self).__init__()
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_seq):
        embeds = self.embeddings(input_seq)
        # Note that the order of mini-batch is random so there is no hidden relationship among batches.
        # So we do not input the previous batch's hidden state,
        # leaving the first hidden state zero `self.lstm(embeds, None)`.
        lstm_out, _ = self.lstm(embeds)
        # use the final hidden state as the next character prediction
        final_hidden_state = lstm_out[:, -1]
        # output = self.fc(final_hidden_state)
        # For fed_shakespeare
        output = self.fc(lstm_out[:,:])
        output = torch.transpose(output, 1, 2)
        return output