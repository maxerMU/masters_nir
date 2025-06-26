import torch
import os

from config import *
from train_model import train_epoch
from test_model_matches import test_model
from model.model import PageAccModel


# import matplotlib.pyplot as plt
# import torch.nn as nn
# 
# def plot_model_weights_histograms(model, bins=500, figsize=(10, 6)):
#     weights = []
#     layer_names = []
#     
#     # Собираем веса из всех слоев
#     for name, module in model.named_modules():
#         if isinstance(module, (nn.Linear, nn.Conv2d)):
#             if hasattr(module, 'weight'):
#                 weights.append(module.weight.data.cpu().flatten().numpy())
#                 layer_names.append(f"{name} ({module.__class__.__name__})")
#     
#     # Создаем график
#     plt.figure(figsize=figsize)
#     # plt.suptitle(f"Weight Distributions: {model_name}", y=1.02)
#     
#     # Определяем сетку для субплотов
#     n = len(weights)
#     rows = int(n**0.5)
#     cols = (n + rows - 1) // rows
#     
#     # Рисуем гистограммы
#     for i, (weight, name) in enumerate(zip(weights, layer_names)):
#         plt.subplot(rows, cols, i+1)
#         plt.hist(weight, bins=bins, alpha=0.7, edgecolor='black')
#         plt.title(name, fontsize=8)
#         plt.xlabel("Weight Value")
#         plt.ylabel("Count")
#     
#     plt.tight_layout()
#     plt.show()
# 
# device = "cpu"
# save_dir = os.path.join("trained_models", f"size_{512}")
# model = PageAccModel(512, MODEL_LSTM_HIDDEN_SIZE, BUFFER_SIZE).to(device)
# model.load_state_dict(torch.load(os.path.join(save_dir, "model_0.pth"), map_location=device, weights_only=True))
# model.eval()
# plot_model_weights_histograms(model)


for hidden_size in range(512, 513, 64):
    print(f"=================== {hidden_size} ===================")
# for batch_size in range(200, 400, 100):
#     print(f"=================== {batch_size} ===================")
    save_dir = os.path.join("trained_models", f"buf_{BUFFER_SIZE}_size_{hidden_size}")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    model = PageAccModel(hidden_size, MODEL_LSTM_HIDDEN_SIZE, BUFFER_SIZE).to(device)
    # model.load_state_dict(torch.load(os.path.join(save_dir, "model_99.pth"), map_location=device, weights_only=True))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    # optimizer.load_state_dict(torch.load(os.path.join(save_dir, "opt_99.pth"), map_location=device, weights_only=True))
    for epoch in range(0, 100):
        print(f"Epoch {epoch}")
        train_epoch(epoch, model, optimizer, BATCH_SIZE, save_dir)
        test_model(model)

# for hidden_size in range(566, 600, 64):
#     print(f"=================== {hidden_size} ===================")
#     save_dir = os.path.join("trained_models", f"size_{hidden_size}")
#     if not os.path.exists(save_dir):
#         os.mkdir(save_dir)
# 
#     model = PageAccModel(hidden_size, MODEL_LSTM_HIDDEN_SIZE, BUFFER_SIZE).to(device)
#     # model.load_state_dict(torch.load(os.path.join(save_dir, "model_39.pth"), map_location=device, weights_only=True))
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
#     # optimizer.load_state_dict(torch.load(os.path.join(save_dir, "opt_39.pth"), map_location=device, weights_only=True))
#     for epoch in range(0, 40):
#         print(f"Epoch {epoch}")
#         train_epoch(epoch, model, optimizer, BATCH_SIZE, save_dir)
#         test_model(model)

