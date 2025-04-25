import torch

BATCH_SIZE = 1000
BUFFER_SIZE = 128
MODEL_HIDDEN_SIZE = 512
MODEL_LSTM_HIDDEN_SIZE = 512
TRAIN_PART = 0.7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
