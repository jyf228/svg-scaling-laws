"""
Autoregressive sampler for the SVG transformer.
"""

import torch
import torch.nn.functional as F

from src.data.tokenizer import SVGTokenizer
from src.model.transformer import Transformer


class Sampler:
    """Sample from a trained SVG transformer."""

    def __init__(self, model: Transformer, tokenizer: SVGTokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_k: int | None = 50,
        top_p: float | None = 0.9,
    ) -> list[int]:
        """
        Generate token ids autoregressively from a prompt.
        Returns the generated sequence of token ids (including the original prompt).
        """
        block_size = self.model.config.block_size
        eos_id = self.tokenizer.eos_token_id
        ids = list(prompt_ids)

        for _ in range(max_new_tokens):
            # Crop context to block_size
            context = ids[-block_size:]
            idx = torch.tensor([context], dtype=torch.long, device=self.device)

            logits, _ = self.model(idx)           # (1, 1, vocab_size) — last position only
            logits = logits[0, -1, :]             # (vocab_size,)

            if temperature == 0.0:
                next_id = int(logits.argmax())
            else:
                logits = logits / temperature

                # Top-k filtering
                if top_k is not None and top_k > 0:
                    k = min(top_k, logits.size(-1))
                    kth_val = logits.topk(k).values[-1]
                    logits = logits.masked_fill(logits < kth_val, float("-inf"))

                # Top-p filtering
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    # Remove tokens once cumulative probability exceeds top_p
                    remove = cumprobs - F.softmax(sorted_logits, dim=-1) > top_p
                    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                    # Return to the original ordering
                    logits = torch.zeros_like(logits).scatter_(0, sorted_idx, sorted_logits)

                probs = F.softmax(logits, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1))

            ids.append(next_id)
            if next_id == eos_id:
                break

        return ids
