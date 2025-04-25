from dataclasses import dataclass, fields
import torch
import torch.nn as nn


@dataclass
class PageBatch:
    rel_id: torch.Tensor        # uint32_t (batch_size)
    fork_num: torch.Tensor      # uint32_t (batch_size)
    block_num: torch.Tensor     # uint32_t (batch_size)
    relfilenode: torch.Tensor   # uint32_t (batch_size)
    rel_kind: torch.Tensor      # enum 10 possible values (batch_size)
    position: torch.Tensor      # [0-BUF_SIZE] (batch_size)

    def get_batch(self, batch_start, batch_end, device):
        kwargs = {
            field.name: getattr(self, field.name)[batch_start:batch_end].to(device)
            for field in fields(self)
        }
        return self.__class__(**kwargs)

    def save(self, file):
        torch.save(
            {field.name: getattr(self, field.name) for field in fields(self)},
            file
        )

    def to(self, device):
        for field in fields(self):
            setattr(
                self, 
                field.name, 
                getattr(self, field.name).to(device)
            )
        return self


def load_page(file) -> PageBatch:
    loaded_data = torch.load(file)

    return PageBatch(**loaded_data)


class HashEmbedding(nn.Module):
    def __init__(self, hash_size: int, embed_dim: int):
        super().__init__()
        self.hash_size = hash_size
        self.embedding = nn.Embedding(
            num_embeddings=hash_size + 1,  # +1 для обработки 0
            embedding_dim=embed_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Применяет хэш-функцию и embedding.

        Args:
            x: Входной тензор произвольной размерности (например, [batch_size])

        Returns:
            Тензор с embedding'ами размерности [..., embed_dim]
        """
        hashed = x % (self.hash_size + 1)

        hashed = hashed.long()

        return self.embedding(hashed)


class PageAccEncoder(nn.Module):
    def __init__(self, hidden_size, embedding_size, buf_size):
        super(PageAccEncoder, self).__init__()

        self._hash_size = 5000
        page_params = len(fields(PageBatch)) * embedding_size

        assert(len(fields(PageBatch)) == 6)
        self._emb_layer = nn.ModuleDict({
            "rel_id": HashEmbedding(self._hash_size, embedding_size),
            "fork_num": HashEmbedding(self._hash_size, embedding_size),
            "block_num": HashEmbedding(self._hash_size, embedding_size),
            "relfilenode": HashEmbedding(self._hash_size, embedding_size),
            "rel_kind": nn.Embedding(10, embedding_size),
            "position": nn.Embedding(buf_size + 1, embedding_size)
        })

        self._page_enc = nn.Sequential(
            nn.Linear(page_params, hidden_size * 2),
            nn.ReLU(),
        )
    
    def forward(self, page_batch: PageBatch):
        embeddings = []
        for field in fields(PageBatch):
            val = getattr(page_batch, field.name)
            emb = self._emb_layer[field.name](val)
            embeddings.append(emb)

        page_acc_input = torch.cat(embeddings, dim=1)
        page_acc_enc = self._page_enc(page_acc_input)

        return page_acc_enc

class PageBufferEncoder(PageAccEncoder):
    def __init__(self, hidden_size, embedding_size, buf_size):
        super().__init__(hidden_size, embedding_size, buf_size)

    def forward(self, buffer_batch: list[PageBatch]):
        embeddings = []
        for field in fields(PageBatch):
            val_stack = torch.stack([getattr(page_in_buf, field.name) for page_in_buf in buffer_batch])
            emb = self._emb_layer[field.name](val_stack)
            embeddings.append(emb)

        # (buffer_size, batch_size, num_features)
        buf_input = torch.cat(embeddings, dim=2)
        buf_res = self._page_enc(buf_input)

        return buf_res.permute(1, 0, 2)

class PageEviction(nn.Module):
    def __init__(self, input_page_size, input_context_size, hidden_size):
        super(PageEviction, self).__init__()

        self._hidden_size = hidden_size

        self._attention_page = nn.Linear(input_page_size, hidden_size, bias=False)
        self._attention_context = nn.Linear(input_context_size, hidden_size, bias=False)
        self._attention_v = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, buffers, context):
        """
        buffers (batch_size, buffer_size, num_features)
        context (batch_size, num_features)
        """

        # (batch_size, 1, hidden_size)
        context_attention = self._attention_context(context).unsqueeze(1)
        # (batch_size, buf_size, hidden_size)
        context_attention_expanded = context_attention.expand(-1, buffers.size(1), -1)

        buffers_attention = self._attention_page(buffers)

        combined_attention = torch.tanh(context_attention_expanded + buffers_attention)
        scores = self._attention_v(combined_attention).squeeze(-1)

        return scores


class PageAccModel(nn.Module):
    def __init__(self, hidden_size, lstm_hidden_size, buf_size, embedding_size = 32):
        super(PageAccModel, self).__init__()

        self._page_acc_encoder = PageAccEncoder(hidden_size, embedding_size, buf_size)
        self._lstm = nn.LSTM(input_size=hidden_size, hidden_size=lstm_hidden_size, batch_first=True)
        self._buf_page_encoder = PageBufferEncoder(hidden_size, embedding_size, buf_size)
        self._page_evict = PageEviction(hidden_size, lstm_hidden_size, hidden_size)


    def forward(self, page_batch: PageBatch, buffer_batch: list[PageBatch], h0=None, c0=None):
        page_acc_res = self._page_acc_encoder(page_batch)

        lstm_input = page_acc_res.view(page_acc_res.shape[0], 1, page_acc_res.shape[1])
        if h0 is None or c0 is None:
            lstm_out, (hn, cn) = self._lstm(lstm_input)
        else:
            lstm_out, (hn, cn) = self._lstm(lstm_input, (h0, c0))

        lstm_out_flat = lstm_out.view(lstm_out.shape[0], lstm_out.shape[2])

        buf_res = self._buf_page_encoder(buffer_batch)

        res = self._page_evict(buf_res, lstm_out_flat)

        return res, hn, cn
