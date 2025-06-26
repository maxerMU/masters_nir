        context_attention = self._attention_context(context).unsqueeze(1)
        # (batch_size, buf_size, hidden_size)
        context_attention_expanded = context_attention.expand(-1, buffers.size(1), -1)

        buffers_attention = self._attention_page(buffers)

        combined_attention = torch.tanh(context_attention_expanded + buffers_attention)
        scores = self._attention_v(combined_attention).squeeze(-1)

        return scores
