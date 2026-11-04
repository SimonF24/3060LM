from dataclasses import dataclass, field
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import VOCAB_SIZE

# We use a standard decoder only Transformer++ architecture (as coined in https://arxiv.org/pdf/2312.00752,
# described in section 4.2.1) i.e. use rotary embeddings, SwiGLU activations, RMSNorm instead of LayerNorm,
# no bias, and higher learning rates

@dataclass
class TransformerConfig:
    vocab_size: int = VOCAB_SIZE
    model_dim: int = 512 # dimension of the token embeddings and the model's hidden states
    num_layers: int = 12 # number of transformer layers/blocks
    num_heads: int = 8 # number of attention heads in each multi-head attention mechanism
    num_kv_heads: int = 0 # number of key/value heads for grouped-query attention, 0 means num_heads (standard MHA)
    # num_heads must divide model_dim evenly, so that each head has an integer dimension
    # (head_dim = model_dim // num_heads)
    dropout: float = 0.0 # dropout rate for regularization, following LLaMA, we set this to 0.0
    rope_base: float = 10000.0 # base for the rotary positional embeddings
    head_dim: int = field(init=False) # dimension of each attention head
    # computed as model_dim // num_heads in __post_init__
    max_seq_len: int = 2048 # maximum sequence length for the model, used for positional embeddings
    qk_norm: bool = False # RMSNorm each head's queries and keys, which bounds the attention logits
    zero_init_residual: bool = False # zero-init the residual-writing projections so each block starts as the identity
    tie_embeddings: bool = True # share weights between the token embedding and the output vocab projection
    
    def __post_init__(self):
        if self.model_dim % self.num_heads != 0:
            raise ValueError(f"model_dim ({self.model_dim}) must be divisible by num_heads ({self.num_heads})")
        self.head_dim = self.model_dim // self.num_heads
        if self.num_kv_heads == 0:
            self.num_kv_heads = self.num_heads
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(f"num_kv_heads ({self.num_kv_heads}) must divide num_heads ({self.num_heads})")
    

class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Projection matrices for query, key, and value
        # These will be reshaped to (num_heads, head_dim) for multi-head attention
        self.q_proj = nn.Linear(config.model_dim, config.model_dim, bias=False)
        kv_dim = config.num_kv_heads * config.head_dim
        self.k_proj = nn.Linear(config.model_dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(config.model_dim, kv_dim, bias=False)
        
        # Attention output projection
        # This allows for mixing information across the heads of the multi-head attention mechanism
        self.attention_out_proj = nn.Linear(config.model_dim, config.model_dim, bias=False)
        
        # SwiGLU MLP and activation layers
        expanded_dim = (8 / 3) * config.model_dim
        # 8/3 comes from a historical desire to roughly match the number of parameters with standard GLU
        # with an expansion factor of 4. Standard GLU uses 2 linear layers, while SwiGLU uses 3.
        # Accordingly, we multiply the expansion factor of 4 by 2/3 to get 8/3. 
        # This convention is used in Llama (see https://github.com/meta-llama/llama/blob/main/llama/model.py#L307)
        # and other models
        # In the SwiGLU paper: https://arxiv.org/pdf/2002.05202, this variant is denoted FFN_SwiGLU
        expanded_dim = math.ceil(expanded_dim / 256) * 256 # round up to nearest multiple of 256
        self.mlp_expand = nn.Linear(config.model_dim, expanded_dim, bias=False)
        self.mlp_contract = nn.Linear(expanded_dim, config.model_dim, bias=False)
        self.glu_gate = nn.Linear(config.model_dim, expanded_dim, bias=False)
        
        self.attention_prenorm = nn.RMSNorm(config.model_dim)
        self.mlp_prenorm = nn.RMSNorm(config.model_dim)
        if config.qk_norm:
            self.q_norm = nn.RMSNorm(config.head_dim)
            self.k_norm = nn.RMSNorm(config.head_dim)
        
    def apply_rope(self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, num_heads, seq_len, head_dim)
        seq_len = x.size(2)
        x1, x2 = torch.chunk(x, 2, dim=-1) # split along the model dimension
        x_swapped = torch.cat([-x2, x1], dim=-1) # merge the halves with negation to set up the -sin term
        # This realizes the ind j to ind j + d/2 pairing of cos and sin in the half-split RoPE formulation
        # Apply RoPE: x_rotated = x * cos + x_swapped * sin
        return x * rope_cos[:seq_len] + x_swapped * rope_sin[:seq_len]
        
    def forward(self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, model_dim = x.size()
        # Save the original input for residual connection
        x_orig = x
        # We use pre-norm
        x = self.attention_prenorm(x)
        # Compute multi-head attention
        head_outputs = []
        # Project to query, key, and value for each head
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # Reshape to (batch_size, seq_len, num_heads, head_dim)
        # then (batch_size, num_heads, seq_len, head_dim) for multi-head attention
        q = q.view(batch_size, seq_len, self.config.num_heads, self.config.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.config.num_kv_heads, self.config.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.config.num_kv_heads, self.config.head_dim).transpose(1, 2)
        if self.config.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        # Apply RoPE positional embeddings to the query and key projections
        q = self.apply_rope(q, rope_cos, rope_sin)
        k = self.apply_rope(k, rope_cos, rope_sin)
        # Grouped-query attention: each group of query heads shares one key/value head
        kv_repeats = self.config.num_heads // self.config.num_kv_heads
        if kv_repeats > 1:
            k = k.repeat_interleave(kv_repeats, dim=1)
            v = v.repeat_interleave(kv_repeats, dim=1)
        # Perform causal multi-head attention
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.config.dropout, is_causal=True)
        # Reshape to (batch_size, seq_len, num_heads, head_dim) then (batch_size, seq_len, model_dim)
        # to effectively concatenate the outputs from all heads
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, model_dim)
        # Perform the output projection to mix information across heads
        x = self.attention_out_proj(attn_out)
        # Apply first residual
        x = x + x_orig
        x_orig = x
        # Apply second RMSNorm pre-norm
        x = self.mlp_prenorm(x)
        # Apply SwiGLU MLP and activation
        x = self.mlp_contract(self.mlp_expand(x) * F.silu(self.glu_gate(x)))
        # Apply second residual connection
        x = x + x_orig
        return x

class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.model_dim)
        
        # Compute RoPE frequencies
        # RoPE paper: https://arxiv.org/pdf/2104.09864
        # (I hate that the paper called these frequencies theta, since theta is usually for angles)
        freqs = config.rope_base ** (-torch.arange(0, config.head_dim, 2).float() / config.head_dim)
        # Calculate angles for each position in the sequence
        # by multiplying the frequencies by the position indices
        positions = torch.arange(0, config.max_seq_len).float()
        angles = torch.outer(positions, freqs) # shape: (max_seq_len, head_dim // 2)
        # Calculate the cosine and sine of the angles, then concat
        # The concat is due to us using the half-split version of RoPE
        # where ind j in cos pairs with ind j + d/2 in sin, where d is the head dimension
        # This is the version used in LLaMA and preserves locality better than the original RoPE formulation
        rope_cos = torch.cat([angles.cos(), angles.cos()], dim=-1) # shape: (max_seq_len, head_dim)
        rope_sin = torch.cat([angles.sin(), angles.sin()], dim=-1) # shape: (max_seq_len, head_dim)
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

        self.transformer_layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        self.vocab_prenorm = nn.RMSNorm(config.model_dim)
        self.vocab_projection = nn.Linear(config.model_dim, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.vocab_projection.weight = self.token_embedding.weight # weight tying
        # This makes the token embedding and output vocab projection share the same weights,
        # making them work in the same embedding space, which saves parameters
        
        # Setting initialization for Linear and Embedding layers to
        # the normal distribution with mean 0 and std 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
        # Scaling residual-writing projections by 1/sqrt(2 * num_layers) to stabilize training
        residual_std = 0.02 / math.sqrt(2 * config.num_layers)
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("mlp_contract.weight"):
                if config.zero_init_residual:
                    nn.init.zeros_(p)
                else:
                    nn.init.normal_(p, mean=0.0, std=residual_std)

    def forward(self, input_tokens: torch.Tensor) -> torch.Tensor:
        assert input_tokens.size(1) <= self.config.max_seq_len, (
            f"Input sequence length ({input_tokens.size(1)}) "
            f"exceeds maximum sequence length ({self.config.max_seq_len})"
        ) # Otherwise we would be indexing out of bounds in the RoPE buffers
        x = self.token_embedding(input_tokens)
        for layer in self.transformer_layers:
            x = layer(x, self.rope_cos, self.rope_sin)
        x = self.vocab_prenorm(x)
        x = self.vocab_projection(x)
        return x
    
if __name__ == "__main__":
    # Test the model, to make sure it runs without errors
    batch_size = 2 # Test batch size
    seq_len = 128  # Test sequence length
    loss_tolerance = 0.5 # Allowable deviation from expected loss (which is log(vocab_size))
    config = TransformerConfig()
    model = Transformer(config)
    test_input = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    output = model(test_input)
    assert output.shape == (batch_size, seq_len, config.vocab_size), (
        f"Output shape {output.shape} does not match expected shape "
        f"({batch_size}, {seq_len}, {config.vocab_size})"
    )
    random_targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(output.view(-1, config.vocab_size), random_targets.view(-1))
    assert torch.isfinite(output).all(), "Output contains non-finite values (NaN or Inf)"
    assert abs(loss.item() - math.log(config.vocab_size)) < loss_tolerance, (
        f"Test loss {loss.item():.4f} deviates from expected loss "
        f"{math.log(config.vocab_size):.4f} by more than {loss_tolerance:.4f}"
    )
    print("Success! The model initialized and ran a forward pass without errors.")