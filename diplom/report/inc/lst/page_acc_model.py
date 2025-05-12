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
