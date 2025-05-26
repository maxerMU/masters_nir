from abc import ABC, abstractmethod
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch

import config
from dataset import create_empty_pages_batch, page_to_batch
from model.model import PageAccModel
from logfile_reader import read_pages, read_optimal_results, Page


class IPageVictimStrategy(ABC):
    @abstractmethod
    def forward(page: Page, buffer: list[Page], expected_victim: int):
        pass

    def get_page_pos(self, page: Page, buffer: list[Page]):
        page_in_buffer = next(filter(lambda x: x[1].get_page_id() == page.get_page_id(), enumerate(buffer)), None)
        return page_in_buffer[0] if page_in_buffer else None

def get_matches(pages: list[Page], victims: list[list[int]], evict_strategy: IPageVictimStrategy):
    buffer = [Page(0, 0, 0, 0, 0)] * config.BUFFER_SIZE

    matches = 0
    hits = 0
    match_rates = []
    pbar = tqdm(range(len(pages)))
    for i in pbar:
        page = pages[i]
        optimal_victim = victims[i][0][0] if len(victims[i]) > 0 else None
        victim = evict_strategy.forward(page, buffer, optimal_victim)
        if victim < 0:
            hits += 1
        else:
            # print(f"{optimal_victim} {victim}")
            if optimal_victim == victim:
                matches += 1
            buffer[optimal_victim] = page
        
        match_rates.append(matches / (i + 1 - hits))

        if i % 1000 == 0:
            pbar.set_postfix_str(f"match_rate={match_rates[-1]}")
    
    return match_rates, buffer

def get_hits(pages: list[Page], evict_strategy: IPageVictimStrategy):
    buffer = [Page(0, 0, 0, 0, 0)] * config.BUFFER_SIZE

    hits = 0
    hit_rates = []
    pbar = tqdm(range(len(pages)))
    for i in pbar:
        page = pages[i]
        victim = evict_strategy.forward(page, buffer, None)
        if victim < 0:
            hits += 1
        else:
            buffer[victim] = page
        
        hit_rates.append(hits / (i + 1))

        if i % 1000 == 0:
            pbar.set_postfix_str(f"hit_rate={hit_rates[-1]}")
    
    return hit_rates

# def get_pg_hits(pages: list[Page]):
#     hits = 0
#     hit_rates = []
#     pbar = tqdm(range(len(pages)))
#     for i in pbar:
#         page = pages[i]
#         hits += page.hit
#         hit_rates.append(hits / (i + 1))
#         pbar.set_postfix_str(f"hit_rate={hit_rates[-1]}")
#     
#     return hit_rates

class ModelVictimStrategy(IPageVictimStrategy):
    def __init__(self, epoch):
        # self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = "cpu"
        self._model = PageAccModel(
            config.MODEL_HIDDEN_SIZE,
            config.MODEL_LSTM_HIDDEN_SIZE,
            config.BUFFER_SIZE).to(self._device)
        self._model.load_state_dict(torch.load(f"trained_models/model_{epoch}.pth", map_location=self._device, weights_only=True))
        self._model.eval()

        self._h = None
        self._c = None
        self._current_iter = 0

    def forward(self, page: Page, buffer: list[Page], expected_victim: int):
        self._current_iter += 1
        page_accs, buffers = create_empty_pages_batch(1, len(buffer))
        page_pos = self.get_page_pos(page, buffer)
        if page_pos is None:
            page_pos = len(buffer)

        page_to_batch(page, 0, page_pos, page_accs)
        page_accs.to(self._device)
        for pos, buf_page in enumerate(buffer):
            page_to_batch(buf_page, 0, pos, buffers[pos])
            buffers[pos].to(self._device)

        with torch.no_grad():
            out, self._h, self._c = self._model.forward(page_accs, buffers, self._h, self._c)

        self._h.detach()
        self._c.detach()
        self._h = self._h.to(self._device)
        self._c = self._c.to(self._device)

        # if self._current_iter % 1000 == 0:
        #     plt.figure(figsize=(12, 4))
        #     plt.bar(range(1024), self._h.view(1024))
        #     plt.xlabel("Dimension")
        #     plt.ylabel("Value")
        #     plt.title("Vector Value Distribution")
        #     plt.show()

        if page_pos < len(buffer):
            return -1

        return torch.argmax(out, dim=1)[0]


class LRUStrategy(IPageVictimStrategy):
    def __init__(self):
        self._current_iter = 0
        self._buffer_pos_last_acc = [0] * config.BUFFER_SIZE

    def forward(self, page: Page, buffer: list[Page], expected_victim: int):
        self._current_iter += 1
        page_pos = self.get_page_pos(page, buffer)
        if page_pos is None:
            victim, _ = min(enumerate(self._buffer_pos_last_acc), key=lambda x: x[1])
            if expected_victim is not None:
                self._buffer_pos_last_acc[expected_victim] = self._current_iter
            else:
                self._buffer_pos_last_acc[victim] = self._current_iter

            return victim

        self._buffer_pos_last_acc[page_pos] = self._current_iter
        return -1


class ClockSweepStrategy(IPageVictimStrategy):
    def __init__(self):
        self._current_item = 0
        self._buffer_pos_acc = [0] * config.BUFFER_SIZE

    def forward(self, page: Page, buffer: list[Page], expected_victim: int):
        page_pos = self.get_page_pos(page, buffer)
        if page_pos is None:
            while self._buffer_pos_acc[self._current_item] > 0:
                self._buffer_pos_acc[self._current_item] -= 1
                self._current_item = self._current_item + 1 if self._current_item < len(self._buffer_pos_acc) - 1 else 0

            victim = self._current_item
            self._buffer_pos_acc[self._current_item] = 1

            return victim

        self._buffer_pos_acc[page_pos] += 1
        return -1


class OptimalStrategy(IPageVictimStrategy):
    def __init__(self, optimal_victims: list[list[int]]):
        self._current_iter = 0
        self._optimal_victims = optimal_victims

    def forward(self, page: Page, buffer: list[Page], expected_victim: int):
        page_pos = self.get_page_pos(page, buffer)
        optimal_victim_pos = self._current_iter
        self._current_iter += 1

        if page_pos is None:
            return self._optimal_victims[optimal_victim_pos][0][0]

        return -1


if __name__ == "__main__":
    pages = read_pages("train_data/tpcc_logfile")
    print(len(pages))
    train_size = int(len(pages) * config.TRAIN_PART)
    test_size = len(pages) - train_size
    # train_pages = pages[:train_size]
    # train_victims = read_optimal_results("train_data/tpcc_logfile_train_victims")
    test_pages = pages[train_size:]
    test_victims = read_optimal_results("train_data/tpcc_logfile_test_victims")

    # strategy = ModelVictimStrategy(28)
    strategy = LRUStrategy()
    # strategy = ClockSweepStrategy()
    # strategy = OptimalStrategy(test_victims)
    # hit_rates = get_hits(test_pages, strategy)
    # hit_rates = get_pg_hits(test_pages)
    matches = get_matches(test_pages, test_victims, strategy)

    # for i in range(349, 370, 5):
    #     print(f"epoch {i}")
    # 
    #     model_evict_strategy = ModelVictimStrategy()
    # 
    #     hit_rates = get_hits(test_pages, model_evict_strategy)
    #     match_rates = get_matches(train_pages, train_victims, model_evict_strategy)

    with open("lru_match_results", "w") as f:
        for rate in matches:
            f.write(f"{rate}\n")

