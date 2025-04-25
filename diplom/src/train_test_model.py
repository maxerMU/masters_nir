import torch

from config import *
from train_model import train_epoch
from test_model_matches import test_model
from model.model import PageAccModel

for hidden_size in range(64, 257, 64):
    print(f"=================== {hidden_size} ===================")

    model = PageAccModel(hidden_size, MODEL_LSTM_HIDDEN_SIZE, BUFFER_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    for epoch in range(0, 20):
        print(f"Epoch {epoch}")
        train_epoch(epoch, model, optimizer)
        test_model(model)

