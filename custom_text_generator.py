import random
import torch
import torch.nn as nn

class CustomGenerator(nn.Module):
    """
    Domain-agnostic example generator.
    Produces simple generic text payloads for infrastructure testing.
    """
    def __init__(self) -> None:
        super().__init__()
        self._anchor = nn.Parameter(torch.zeros(1), requires_grad=False)

    @torch.no_grad()
    def forward(self, batch_size: int, prompt: str = "Generic structured content:", max_new_tokens: int = 128, temperature: float = 0.9):
        outputs = []
        for _ in range(batch_size):
            token_count = max(8, min(max_new_tokens, 128))
            body = " ".join(f"token{random.randint(0, 9999)}" for _ in range(token_count // 4))
            outputs.append(f"{prompt} {body}")
        return outputs


