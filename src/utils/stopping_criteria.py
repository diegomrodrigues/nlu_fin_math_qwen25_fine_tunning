import re
import torch # type: ignore
from transformers import StoppingCriteria
from transformers import StoppingCriteriaList

class CodeBlockStoppingCriteria(StoppingCriteria):
    """Stopping criteria for code blocks."""
    def __init__(self, tokenizer, device):
        super().__init__()
        self.python_start_pattern = "```output"
        self.python_start_tokens = tokenizer(
            [self.python_start_pattern],
            return_tensors="pt"
        ).to(device)
        self.start_token_ids = self.python_start_tokens.input_ids[0]
        self.pattern_length = len(self.start_token_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
        last_tokens = input_ids[:, -self.pattern_length:]
        matches = torch.all(last_tokens == self.start_token_ids, dim=1)
        return torch.any(matches).item() 