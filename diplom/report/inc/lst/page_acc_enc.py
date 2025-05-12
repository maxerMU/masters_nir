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
