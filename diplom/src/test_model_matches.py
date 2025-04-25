import torch
from tqdm import tqdm

from config import *
from dataset import load_train_data
from model.model import PageAccModel

pages_acc, buffers, optimal_predictions, hit_fail_mask = load_train_data("test_data")
def get_batch(batch_start, batch_end, device):
    pages_acc_batch = pages_acc.get_batch(batch_start, batch_end, device)
    buffers_batch = [page.get_batch(batch_start, batch_end, device) for page in buffers]
    optimal_predictions_batch = optimal_predictions[batch_start:batch_end].to(device)
    hit_fail_mask_batch = hit_fail_mask[batch_start:batch_end].to(device)

    return pages_acc_batch, buffers_batch, optimal_predictions_batch, hit_fail_mask_batch

def test_model(model):
    model.eval()

    h, c = None, None
    matches_sum = 0
    matches = []
    hit_fail_count = 0
    pbar = tqdm(range(0, len(pages_acc.rel_id), BATCH_SIZE))
    for i in pbar:
        batch_start = i
        batch_end = i + BATCH_SIZE
        if batch_end >= len(pages_acc.rel_id):
            continue

        pages_acc_batch, buffers_batch, optimal_predictions_batch, hit_fail_mask_batch = get_batch(batch_start, batch_end, device)

        out, h, c = model.forward(pages_acc_batch, buffers_batch, h, c)
        h.to(device)
        c.to(device)

        if any(hit_fail_mask_batch):
            matches_sum += torch.sum(torch.argmax(out[hit_fail_mask_batch], dim=1) == optimal_predictions_batch[hit_fail_mask_batch]).item()
            hit_fail_count += sum(hit_fail_mask_batch)
            matches_avg = matches_sum / hit_fail_count
            matches.append(matches_avg)

            pbar.set_postfix_str(f"matches={matches_avg:.3f}")

        h = h.detach()
        c = c.detach()
    
    # with open("model_match_results", "w") as f:
    #     for match in matches:
    #         f.write(f"{match}\n")

if __name__ == "__main__":
    for epoch in range(0, 50):
        print(f"Epoch: {epoch}")
        model = PageAccModel(MODEL_HIDDEN_SIZE, MODEL_LSTM_HIDDEN_SIZE, BUFFER_SIZE).to(device)
        model.load_state_dict(torch.load(f"trained_models/model_{epoch}.pth", map_location=device, weights_only=True))

        test_model(model)