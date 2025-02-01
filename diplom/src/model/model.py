import torch
import torch.nn as nn

class PageAccModel(nn.Module):
    def __init__(self, page_params, hidden_size, lstm_hidden_size, buf_size):
        super(PageAccModel, self).__init__()

        self._trace_file = open("model.txt", "w")

        self._page_params = page_params
        self._hidden_size = hidden_size
        self._lstm_hidden_size = lstm_hidden_size
        self._buf_size = buf_size
        
        self._page_acc_enc = nn.Sequential(
            nn.Linear(page_params, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_size),
            nn.ReLU(),
        )

        self._lstm = nn.LSTM(input_size=hidden_size, hidden_size=lstm_hidden_size, batch_first=True)
        
        self._buf_page_enc = nn.Sequential(
            nn.Linear(page_params * buf_size, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_size * buf_size),
            nn.ReLU(),
        )

        self._page_evict = nn.Sequential(
            nn.Linear(lstm_hidden_size + hidden_size * buf_size, 512),
            nn.ReLU(),
            nn.Linear(512, buf_size),
            nn.ReLU()
            # nn.Softmax(dim=1)
        )

        def init_weights(layer):
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(layer.bias, 0.0)  # Инициализация смещений нулями
        self._page_acc_enc.apply(init_weights)
        self._buf_page_enc.apply(init_weights)
        self._page_evict.apply(init_weights)
        nn.init.orthogonal_(self._lstm.weight_hh_l0)
        nn.init.kaiming_normal_(self._lstm.weight_ih_l0)

    def forward(self, page_params, buf_pages, h0=None, c0=None):
        # self._trace_file.write("//////////////////////////////////////////////\nn")
        page_acc_enc = self._page_acc_enc(page_params)

        lstm_input = page_acc_enc.view(page_acc_enc.shape[0], 1, page_acc_enc.shape[1])
        if h0 is None or c0 is None:
            lstm_out, (hn, cn) = self._lstm(lstm_input)
        else:
            lstm_out, (hn, cn) = self._lstm(lstm_input, (h0, c0))

        lstm_out_flat = lstm_out.view(lstm_out.shape[0], lstm_out.shape[2])
        
        buf_page_out = self._buf_page_enc(buf_pages)

        # for i in range(len(lstm_out)):
        #     for el in lstm_out[i]:
        #         self._trace_file.write(f"{i}. ===================\n")
        #         self._trace_file.write(f"page_params\n{page_params[i]}\n")
        #         self._trace_file.write(f"lstm_input\n{lstm_input[i][0]}\n")
        #         self._trace_file.write(f"lstm_output\n{el}\n")
        #         self._trace_file.write(f"buf_page_out\n{buf_page_out[i]}\n")
        #         self._trace_file.write("===================\n")


        page_evict_input = torch.cat((lstm_out_flat, buf_page_out), dim=1)
        res = self._page_evict(page_evict_input)

        # self._trace_file.write("//////////////////////////////////////////////\nn")
        
        return res, hn, cn