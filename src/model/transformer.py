"""
Implementation of a causal decoder-only transformer architecture.
This code is adapted from the nanoGPT implementation by Andrej Karpathy:
https://github.com/karpathy/nanoGPT
"""

import logging
import math

import mup
import mup.init
import torch
import torch.nn as nn
from torch.nn import functional as F


logger = logging.getLogger(__name__)


class LayerNorm(nn.Module):
    """
    LayerNorm but with an optional bias.
    Adapted from nanoGPT with minimal modifications.
    """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    

class CausalSelfAttention(nn.Module):
    """Causal self-attention layer with multiple heads. Adapted from nanoGPT."""

    def __init__(self, config):
        super().__init__()

        if config.d_model % config.n_head != 0:
            raise ValueError(f"d_model ({config.d_model}) must be divisible by n_head ({config.n_head})")

        # Key, query, value projections for all heads, but in a batch
        self.c_attn = (mup.Linear if config.use_mup else nn.Linear)(config.d_model, 3 * config.d_model, bias=config.bias)

        # Output projection
        self.c_proj = (mup.Linear if config.use_mup else nn.Linear)(config.d_model, config.d_model, bias=config.bias)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.d_model = config.d_model
        self.dropout = config.dropout
        self.use_mup = config.use_mup

        # Flash attention requires PyTorch >= 2.0.  
        # If not available, will use the manual (slower) implementation of attention.
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            logger.warning("Warning: Using slow attention. Flash Attention requires PyTorch >= 2.0")
            # Causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (d_model)

        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.d_model, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        d_head = C // self.n_head   # dimension per attention head 
        # Use alternative attention scaling for µP per https://github.com/microsoft/mup#basic-usage
        attn_scale = (1.0 / d_head) if self.use_mup else (1.0 / math.sqrt(d_head))

        # Causal self-attention - Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # Efficient attention using Flash Attention CUDA kernels
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True,
                scale=attn_scale,
            )
        else:
            # Manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * attn_scale
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    

class MLP(nn.Module):
    """MLP layer. Adapted from nanoGPT."""

    def __init__(self, config):
        super().__init__()
        # nanoGPT uses 4*d_model for the hidden dimensions of the MLP - here we use d_ff
        self.c_fc    = (mup.Linear if config.use_mup else nn.Linear)(config.d_model, config.d_ff, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = (mup.Linear if config.use_mup else nn.Linear)(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    Transformer block - attention followed by a MLP, with skip connections and layer norm.
    Adapted from nanoGPT with minimal modifications.
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.d_model, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.d_model, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) # skip connection around attention
        x = x + self.mlp(self.ln_2(x))  # skip connection around MLP
        return x


class Transformer(nn.Module):
    """
    The full transformer language model. Adapted from nanoGPT.
    """

    def __init__(self, config):
        super().__init__()
        if config.vocab_size is None or config.block_size is None:
            raise ValueError("vocab_size and block_size must be specified")
        
        self.config = config
        self.use_mup = config.use_mup

        self.transformer = nn.ModuleDict(dict(
            token_embedding = nn.Embedding(config.vocab_size, config.d_model),
            pos_embedding = nn.Embedding(config.block_size, config.d_model),
            dropout = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.d_model, bias=config.bias),
        ))

        if config.use_mup:
            # Replace output layer with MuSharedReadout (which also uses weight tying) for µP
            self.lm_head = mup.MuSharedReadout(self.transformer.token_embedding.weight, bias=False)
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
            # Weight tying (https://paperswithcode.com/method/weight-tying)
            self.transformer.token_embedding.weight = self.lm_head.weight

        # Initialize all weights
        self.apply(self._init_weights)

        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        logger.info(f"Number of model parameters: {self._get_num_params()/1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def reinit_weights_mup(self) -> None:
        """
        Replace init weights with mup.init after base shapes are set
        per https://github.com/microsoft/mup#basic-usage
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                mup.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

        # Re-apply the scaled init to the residual projections as we did in __init__ but with mup.init
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                mup.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))

        # Initialize the query matrix to 0 for µP
        self._zero_init_queries()

    def _zero_init_queries(self) -> None:
        """
        Zero-initialize the query matrix per 
        https://github.com/microsoft/mup#making-your-own-coord-check-plots.
        """
        for block in self.transformer.h:
            with torch.no_grad():
                block.attn.c_attn.weight[:self.config.d_model, :].zero_()

    def _get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.pos_embedding.weight.numel()
        return n_params

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        if t > self.config.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
            )
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        tok_emb = self.transformer.token_embedding(idx) # token embeddings of shape (b, t, d_model)
        pos_emb = self.transformer.pos_embedding(pos) # position embeddings of shape (t, d_model)
        x = self.transformer.dropout(tok_emb + pos_emb)
        
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # If we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
    