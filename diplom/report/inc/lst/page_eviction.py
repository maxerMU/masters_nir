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
