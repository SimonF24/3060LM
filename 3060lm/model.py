from dataclasses import dataclass, field
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# We use a standard decoder only Transformer++ architecture (as coined in https://arxiv.org/pdf/2312.00752,
# described in section 4.2.1) i.e. use rotary embeddings, SwiGLU activations, RMSNorm instead of LayerNorm,
# no bias, and higher learning rates

@dataclass
class TransformerConfig:
    vocab_size: int = 32_768  # matches tokenizer/stem32k.json
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
    activation: str = "swiglu" # "swiglu", or "relu2", which drops the gate matrix and squares a ReLU
    mlp_dim: int = 0 # hidden dimension of the MLP, 0 means the default rounding of 8/3 * model_dim
    embed_dim: int = 0 # if > 0 and < model_dim, factorize the embedding and output head through this dimension
    logit_softcap: float = 0.0 # if > 0, smoothly limit logits to +/- this value with a tanh
    bigram_rows: int = 0 # if > 0, add a zero-initialized embedding of the hashed (previous, current) token pair
    embed_skip: bool = False # add the token embedding back in before every block, with a learned scale starting at 0
    value_residual: bool = False # mix the first layer's attention values into every later layer's, with a learned weight
    # Weight sharing: layers[loop_start:loop_end] run num_loops times in a row
    loop_start: int = 0
    loop_end: int = 0
    num_loops: int = 1
    
    def __post_init__(self):
        if self.model_dim % self.num_heads != 0:
            raise ValueError(f"model_dim ({self.model_dim}) must be divisible by num_heads ({self.num_heads})")
        self.head_dim = self.model_dim // self.num_heads
        if self.num_kv_heads == 0:
            self.num_kv_heads = self.num_heads
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(f"num_kv_heads ({self.num_kv_heads}) must divide num_heads ({self.num_heads})")
    

class KVCache:
    """
    Keys and values of the tokens seen so far, so generation only runs the newest token through the model. Each row
    of the batch has its own positions, so rows can hold sequences of different lengths. The buffers are allocated
    once, so a decode step can be captured as a CUDA graph.
    """
    def __init__(self, keys: list[torch.Tensor], values: list[torch.Tensor]):
        self.keys, self.values = keys, values  # per layer application: (batch, kv_heads, max_len, head_dim)
        self.positions = self.mask = None
        self.used = 0

    def start(self, positions: torch.Tensor, used: int | None = None):
        """
        Sets the positions (batch, seq_len) of the tokens in the next forward pass. Attention covers the first used
        cache entries, which must include every position; passing it avoids a GPU sync, as CUDA graphs require.
        """
        self.positions = positions
        self.used = used if used is not None else int(positions.max()) + 1
        # Each token attends to the cache entries at or before its own position
        keys = torch.arange(self.used, device=positions.device)
        self.mask = (keys <= positions[..., None]).unsqueeze(1)  # (batch, 1, seq_len, used)

    def write(self, slot: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Stores this pass's keys and values at their positions and returns the first used entries."""
        index = self.positions[:, None, :, None].expand_as(k)
        self.keys[slot].scatter_(2, index, k.to(self.keys[slot].dtype))
        self.values[slot].scatter_(2, index, v.to(self.values[slot].dtype))
        return self.keys[slot][:, :, :self.used], self.values[slot][:, :, :self.used]


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        if config.value_residual:
            self.value_mix = nn.Parameter(torch.tensor(0.5))
        
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
        if config.mlp_dim > 0:
            expanded_dim = config.mlp_dim
        self.mlp_expand = nn.Linear(config.model_dim, expanded_dim, bias=False)
        self.mlp_contract = nn.Linear(expanded_dim, config.model_dim, bias=False)
        if config.activation == "swiglu":
            self.glu_gate = nn.Linear(config.model_dim, expanded_dim, bias=False)
        
        self.attention_prenorm = nn.RMSNorm(config.model_dim)
        self.mlp_prenorm = nn.RMSNorm(config.model_dim)
        if config.qk_norm:
            self.q_norm = nn.RMSNorm(config.head_dim)
            self.k_norm = nn.RMSNorm(config.head_dim)
        
    def apply_rope(self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, num_heads, seq_len, head_dim)
        seq_len = x.size(2)
        if rope_cos.dim() == 2:  # the full tables, for positions 0 to seq_len - 1
            rope_cos, rope_sin = rope_cos[:seq_len], rope_sin[:seq_len]
        x1, x2 = torch.chunk(x, 2, dim=-1) # split along the model dimension
        x_swapped = torch.cat([-x2, x1], dim=-1) # merge the halves with negation to set up the -sin term
        # This realizes the ind j to ind j + d/2 pairing of cos and sin in the half-split RoPE formulation
        # Apply RoPE: x_rotated = x * cos + x_swapped * sin
        return x * rope_cos + x_swapped * rope_sin
        
    def forward(self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor,
                v_first: torch.Tensor | None = None, cache: KVCache | None = None,
                slot: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
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
        # Value residual: later layers blend in the first layer's values, which are returned for them to use
        if v_first is None:
            v_first = v
        elif self.config.value_residual:
            v = self.value_mix * v + (1 - self.value_mix) * v_first
        if self.config.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        # Apply RoPE positional embeddings to the query and key projections
        q = self.apply_rope(q, rope_cos, rope_sin)
        k = self.apply_rope(k, rope_cos, rope_sin)
        if cache is not None:
            k, v = cache.write(slot, k, v)
        # Grouped-query attention: each group of query heads shares one key/value head
        kv_repeats = self.config.num_heads // self.config.num_kv_heads
        if kv_repeats > 1:
            k = k.repeat_interleave(kv_repeats, dim=1)
            v = v.repeat_interleave(kv_repeats, dim=1)
        # Perform causal multi-head attention
        if cache is not None:
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=cache.mask)
        else:
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
        if self.config.activation == "swiglu":
            x = self.mlp_contract(self.mlp_expand(x) * F.silu(self.glu_gate(x)))
        else:
            x = self.mlp_contract(F.relu(self.mlp_expand(x)) ** 2)
        # Apply second residual connection
        x = x + x_orig
        return x, v_first

class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        embed_dim = config.embed_dim if config.embed_dim > 0 else config.model_dim
        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        if config.bigram_rows > 0:
            self.bigram_embedding = nn.Embedding(config.bigram_rows, embed_dim)
        self.factorized = embed_dim < config.model_dim
        if self.factorized:
            self.embed_in = nn.Linear(embed_dim, config.model_dim, bias=False)
            self.embed_out = nn.Linear(config.model_dim, embed_dim, bias=False)
        
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
        
        if config.embed_skip:
            if config.num_loops > 1:
                raise ValueError("embed_skip doesn't support looped layers")
            self.embed_skip = nn.Parameter(torch.zeros(config.num_layers))
        self.vocab_prenorm = nn.RMSNorm(config.model_dim)
        self.vocab_projection = nn.Linear(embed_dim, config.vocab_size, bias=False)
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
        
        if config.bigram_rows > 0:
            nn.init.zeros_(self.bigram_embedding.weight)

        # Scaling residual-writing projections by 1/sqrt(2 * num_layers) to stabilize training
        residual_std = 0.02 / math.sqrt(2 * config.num_layers)
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("mlp_contract.weight"):
                if config.zero_init_residual:
                    nn.init.zeros_(p)
                else:
                    nn.init.normal_(p, mean=0.0, std=residual_std)

    def make_cache(self, batch_size: int, dtype: torch.dtype = torch.bfloat16, length: int = 0) -> KVCache:
        """A KV cache for batch_size rows of up to length tokens (max_seq_len by default)."""
        c = self.config
        applications = c.num_layers + (c.loop_end - c.loop_start) * (c.num_loops - 1)
        shape = (batch_size, c.num_kv_heads, length or c.max_seq_len, c.head_dim)
        device = self.rope_cos.device
        return KVCache([torch.zeros(shape, dtype=dtype, device=device) for _ in range(applications)],
                       [torch.zeros(shape, dtype=dtype, device=device) for _ in range(applications)])

    def forward(self, input_tokens: torch.Tensor, return_hidden: bool = False, cache: KVCache | None = None,
                positions: torch.Tensor | None = None, select: torch.Tensor | None = None,
                used: int | None = None) -> torch.Tensor:
        """
        With a cache, positions (batch, seq_len) gives each token's position (below max_seq_len), and the keys and
        values are added to the cache; used is passed to KVCache.start. select (batch,) picks one index along
        seq_len per row, so only those logits are computed.
        """
        if cache is not None:
            assert self.config.bigram_rows == 0, "the bigram embedding needs the previous token, which a cache lacks"
            cache.start(positions, used)
            rope_cos, rope_sin = self.rope_cos[positions].unsqueeze(1), self.rope_sin[positions].unsqueeze(1)
        else:
            rope_cos, rope_sin = self.rope_cos, self.rope_sin
        assert input_tokens.size(1) <= self.config.max_seq_len, (
            f"Input sequence length ({input_tokens.size(1)}) "
            f"exceeds maximum sequence length ({self.config.max_seq_len})"
        ) # Otherwise we would be indexing out of bounds in the RoPE buffers
        x = self.token_embedding(input_tokens)
        if self.config.bigram_rows > 0:
            # Hash the pair into a table row. The first position has no previous token, so it uses id 0
            previous = F.pad(input_tokens[:, :-1], (1, 0))
            x = x + self.bigram_embedding((previous * 1_000_003 + input_tokens) % self.config.bigram_rows)
        if self.factorized:
            x = self.embed_in(x)
        c = self.config
        layers = list(self.transformer_layers)
        if c.num_loops > 1:
            layers = layers[:c.loop_start] + layers[c.loop_start:c.loop_end] * c.num_loops + layers[c.loop_end:]
        embedded = x
        v_first = None
        for i, layer in enumerate(layers):
            if c.embed_skip:
                x = x + self.embed_skip[i] * embedded
            x, v_first = layer(x, rope_cos, rope_sin, v_first if i > 0 else None, cache, i)
        if select is not None:
            x = x[torch.arange(x.size(0), device=x.device), select].unsqueeze(1)
        x = self.vocab_prenorm(x)
        if self.factorized:
            x = self.embed_out(x)
        if return_hidden:
            return x # the state just before the output head, so a caller can score only some of the vocab
        x = self.vocab_projection(x)
        if c.logit_softcap > 0:
            x = c.logit_softcap * torch.tanh(x / c.logit_softcap)
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