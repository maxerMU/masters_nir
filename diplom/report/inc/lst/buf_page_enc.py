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
