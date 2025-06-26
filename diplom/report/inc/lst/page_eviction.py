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


