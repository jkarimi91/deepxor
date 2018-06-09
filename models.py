import torch.nn as nn

INPUT_SIZE = 1
OUTPUT_SIZE = 1
HIDDEN_SIZE = 1
NUM_LAYERS = 1
BATCH_FIRST = True


class BaseRNN(nn.Module):
    def __init__(self, cell):
        super(BaseRNN, self).__init__()
        self.Cell = cell
        self.Linear = nn.Linear(in_features=HIDDEN_SIZE, out_features=OUTPUT_SIZE)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        outputs, _ = self.Cell(inputs)
        outputs = self.Linear(outputs[:, -1])
        return self.Sigmoid(outputs)


class LSTM(BaseRNN):
    def __init__(self):
        cell = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, batch_first=BATCH_FIRST)
        super(LSTM, self).__init__(cell)


class RNN(BaseRNN):
    def __init__(self):
        cell = nn.RNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, batch_first=BATCH_FIRST)
        super(RNN, self).__init__(cell)


class GRU(BaseRNN):
    def __init__(self):
        cell = nn.GRU(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, batch_first=BATCH_FIRST)
        super(GRU, self).__init__(cell)
