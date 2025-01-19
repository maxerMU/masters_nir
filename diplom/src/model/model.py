import torch
import torch.nn as nn

class PageAccModel(nn.Module):
    def __init__(self, page_params, hidden_size, buf_size):
        super(PageAccModel, self).__init__()

        self._page_params = page_params
        self._hidden_size = hidden_size
        self._buf_size = buf_size
        
        self._page_acc_enc = nn.Sequential(
            nn.Linear(page_params, hidden_size),
            nn.ReLU()
        )

        self._lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        
        self._buf_page_enc = nn.Sequential(
            nn.Linear(page_params * buf_size, hidden_size * buf_size),
            nn.ReLU()
        )

        self._page_evict = nn.Sequential(
            nn.Linear(hidden_size + hidden_size * buf_size, buf_size + 1),
            nn.Softmax(dim=1)
        )

    def forward(self, page_params, buf_pages, h0=None, c0=None):
        page_acc_enc = self._page_acc_enc(page_params)

        lstm_input = page_acc_enc.view(page_acc_enc.shape[0], 1, page_acc_enc.shape[1])
        if h0 is None or c0 is None:
            lstm_out, (hn, cn) = self._lstm(lstm_input)
        else:
            lstm_out, (hn, cn) = self._lstm(lstm_input, (h0, c0))
        lstm_out_flat = lstm_out.view(lstm_out.shape[0], lstm_out.shape[2])
        
        buf_page_out = self._buf_page_enc(buf_pages)

        page_evict_input = torch.cat((lstm_out_flat, buf_page_out), dim=1)
        res = self._page_evict(page_evict_input)
        
        return res, hn, cn