import torch
from tqdm import tqdm

from config import (
    BUFFER_SIZE,
    device,
    MODEL_HIDDEN_SIZE,
    MODEL_LSTM_HIDDEN_SIZE,
    BATCH_SIZE,
    TRAIN_DATA_FOLDER
)
from dataset import load_train_data
from model.model import PageAccModel


pages_acc, buffers, optimal_predictions, hit_fail_mask = load_train_data(TRAIN_DATA_FOLDER)
def get_batch(batch_start, batch_end, device):
    pages_acc_batch = pages_acc.get_batch(batch_start, batch_end, device)
    buffers_batch = [page.get_batch(batch_start, batch_end, device) for page in buffers]
    optimal_predictions_batch = optimal_predictions[batch_start:batch_end].to(device)
    hit_fail_mask_batch = hit_fail_mask[batch_start:batch_end].to(device)

    return pages_acc_batch, buffers_batch, optimal_predictions_batch, hit_fail_mask_batch


def train_epoch(epoch: int, model: PageAccModel, optimizer: torch.optim.Optimizer, batch_size: int, save_folder: str):
    model.train()

    history = None
    loss = torch.nn.CrossEntropyLoss(reduction='none') # Установим 'none' для получения потерь по каждому элементу
    loss_sum = 0
    matches_sum = 0
    hit_fail_count = 0
    pbar = tqdm(range(0, len(pages_acc.rel_id), batch_size))
    for i in pbar:
        batch_start = i
        batch_end = i + batch_size
        if batch_end >= len(pages_acc.rel_id):
            continue

        pages_acc_batch, buffers_batch, optimal_predictions_batch, hit_fail_mask_batch = get_batch(batch_start, batch_end, device)

        optimizer.zero_grad()

        out, history = model.forward(pages_acc_batch, buffers_batch, history)
        for i in range(len(history)):
            history[i] = history[i].to(device).detach()

        if any(hit_fail_mask_batch):
            losses = loss(out, optimal_predictions_batch)
            masked_losses = losses[hit_fail_mask_batch]
            loss_value = masked_losses.mean()
    
            loss_value.backward()

            optimizer.step()

            matches_sum += torch.sum(torch.argmax(out[hit_fail_mask_batch], dim=1) == optimal_predictions_batch[hit_fail_mask_batch]).item()
            hit_fail_count += sum(hit_fail_mask_batch)
            matches_avg = matches_sum / hit_fail_count

            loss_sum += loss_value.item()
            loss_avg = loss_sum / (batch_end // batch_size)

            pbar.set_postfix_str(f"loss={loss_avg:.3f} matches={matches_avg:.3f}")


    torch.save(model.state_dict(), f"{save_folder}/model_{epoch}.pth")
    torch.save(optimizer.state_dict(), f"{save_folder}/opt_{epoch}.pth")


if __name__ == "__main__":
    model = PageAccModel(MODEL_HIDDEN_SIZE, MODEL_LSTM_HIDDEN_SIZE, BUFFER_SIZE).to(device)
    # model.load_state_dict(torch.load("trained_models/model_369.pth", map_location=device, weights_only=True))
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    # optimizer.load_state_dict(torch.load("trained_models/opt_369.pth", map_location=device, weights_only=True))

    for epoch in range(0, 50):
        train_epoch(epoch, model, optimizer, BATCH_SIZE, "trained_models")