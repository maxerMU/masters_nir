from dataclasses import fields
import torch
from tqdm import tqdm
import os

from config import *
from model.model import PageBatch, load_page
from logfile_reader import Page, read_optimal_results, read_pages


def get_model_optimal_res(pages, optimal_results, buffer, current_index):
    res = [0] * (len(buffer))
    if (current_index >= len(pages)):
        print(f"ERROR: current_index=={current_index} pages.size() == {len(pages)}")

    if len(optimal_results[current_index]) == 0:
        page_in_buffer = next(filter(lambda el: el[1].get_page_id() == pages[current_index].get_page_id(), enumerate(buffer)), None)
        return res, page_in_buffer[0]
    
    victims_rates = optimal_results[current_index]
    res[victims_rates[0][0]] = 1

    return res, victims_rates[0][0]

def int_to_binary_tensor(x: int, bits: int = 32) -> torch.Tensor:
    binary_str = bin(x)[2:]  # Убираем префикс '0b'
    binary_str = binary_str.zfill(bits)
    binary_list = [int(bit) for bit in binary_str]

    return torch.tensor(binary_list, dtype=torch.float32)

def page_to_batch(page: Page, i: int, pos_in_buf: int, page_batch: PageBatch):
    assert(len(fields(PageBatch)) == 6)

    page_batch.rel_id[i] = page.rel_id
    page_batch.fork_num[i] = page.fork_num
    page_batch.block_num[i] = page.block_num
    page_batch.relfilenode[i] = page.relfilenode
    page_batch.rel_kind[i] = page.relkind
    page_batch.position[i] = pos_in_buf

def create_empty_pages_batch(batch_size, buffer_size):
    page_accs = PageBatch(rel_id=torch.empty(batch_size, dtype=torch.int), fork_num=torch.empty(batch_size, dtype=torch.int), block_num=torch.empty(batch_size, dtype=torch.int), relfilenode=torch.empty(batch_size, dtype=torch.int), rel_kind=torch.empty(batch_size, dtype=torch.int), position=torch.empty(batch_size, dtype=torch.int))
    buffers = [PageBatch(rel_id=torch.empty(batch_size, dtype=torch.int), fork_num=torch.empty(batch_size, dtype=torch.int), block_num=torch.empty(batch_size, dtype=torch.int), relfilenode=torch.empty(batch_size, dtype=torch.int), rel_kind=torch.empty(batch_size, dtype=torch.int), position=torch.empty(batch_size, dtype=torch.int)) for _ in range(buffer_size)]

    return page_accs, buffers

def save_train_data(pages, optimal_results, dir: str):
    buffer = [Page(0, 0, 0, 0, 0, 0)] * BUFFER_SIZE

    page_accs, buffers = create_empty_pages_batch(len(pages), len(buffer))
    optimal_predictions = []
    hit_fail_mask = []

    for i in tqdm(range(len(pages))):
        page = pages[i]
        res, victim = get_model_optimal_res(pages, optimal_results, buffer, i)
        page_missed = sum(res) > 0
        hit_fail_mask.append(page_missed)
        optimal_predictions.append(res)

        pos_in_buf = victim if not page_missed else len(buffer)
        page_to_batch(page, i, pos_in_buf, page_accs)
        for pos, buf_page in enumerate(buffer):
            page_to_batch(buf_page, i, pos, buffers[pos])

        if page_missed:
            buffer[victim] = page

    optimal_predictions = torch.argmax(torch.Tensor(optimal_predictions), dim=1)
    hit_fail_mask = torch.tensor(hit_fail_mask, dtype=torch.bool)

    page_accs.save(os.path.join(dir, "page_accs.pth"))
    for i, page in enumerate(buffers):
        page.save(os.path.join(dir, f"buf_page_{i}.pth"))
    torch.save(optimal_predictions, os.path.join(dir, "optimal_predictions.pth"))
    torch.save(hit_fail_mask, os.path.join(dir, "hit_fail_mask.pth"))

def load_train_data(dir: str):
    print("loading pages accs")
    page_accs = load_page(os.path.join(dir, "page_accs.pth"))

    print("loading buffers")
    buffers = []
    for i in tqdm(range(BUFFER_SIZE)):
        buffers.append(load_page(os.path.join(dir, f"buf_page_{i}.pth")))

    print("loading optimal predictions")
    optimal_predictions = torch.load(os.path.join(dir, "optimal_predictions.pth"))

    print("loading hit fail mask")
    hit_fail_mask = torch.load(os.path.join(dir, "hit_fail_mask.pth"))

    return page_accs, buffers, optimal_predictions, hit_fail_mask

if __name__ == "__main__":
    pages = read_pages("train_data/tpcc_logfile")
    train_size = int(len(pages) * TRAIN_PART)
    # train_pages = pages[:train_size]
    test_pages = pages[train_size:]
    del pages

    # %%
    # optimal_results = read_optimal_results("train_data/tpcc_logfile_train_victims")
    optimal_results = read_optimal_results("train_data/tpcc_logfile_test_victims")

    # %%
    # assert(len(optimal_results) == len(train_pages))
    print(len(test_pages))

    # save_train_data(train_pages, optimal_results, "train_data")
    save_train_data(test_pages, optimal_results, "test_data")
