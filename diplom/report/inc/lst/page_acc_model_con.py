        else:
            lstm_out, (hn, cn) = self._lstm(lstm_input, (h0, c0))

        lstm_out_flat = lstm_out.view(lstm_out.shape[0], lstm_out.shape[2])

        buf_res = self._buf_page_encoder(buffer_batch)

        res = self._page_evict(buf_res, lstm_out_flat)

        return res, hn, cn
