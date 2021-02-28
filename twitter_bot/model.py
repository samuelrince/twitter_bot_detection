import torch.nn as nn
import torch.nn.functional as F


class LSTMForSequenceClassification(nn.Module):

    def __init__(self, n_tokens, n_inputs, n_hidden, n_layers, dropout=0.2):
        super(LSTMForSequenceClassification, self).__init__()
        self.n_tokens = n_tokens
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.Embedding(n_tokens, n_inputs)
        self.lstm = nn.LSTM(n_inputs, n_hidden, n_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(n_inputs, 1)

    def forward(self, input_, hidden):
        output = self.dropout(self.encoder(input_))
        output, hidden = self.lstm(output, hidden)
        output = self.fc(output)
        output = output[:, -1]
        output = F.sigmoid(output)
        output = output.flatten()
        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.n_layers, batch_size, self.n_hidden),
                weight.new_zeros(self.n_layers, batch_size, self.n_hidden))


if __name__ == '__main__':
    # Example of LSTM neural network for binary classification
    model = LSTMForSequenceClassification(10**6, 200, 200, 4, 0.5)
    print(model)
