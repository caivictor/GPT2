import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from typing import Any, Callable, Sequence, Optional
from dataclasses import dataclass

# --- GPTConfig, CausalSelfAttention, MLP, Block, GPT (from previous responses) ---
@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257 # Number of tokens
    n_layer: int = 12   # number of layers
    n_head: int = 12    # number of heads
    n_embd: int = 768   # embedding dimension

class CausalSelfAttention(nn.Module):
    config: GPTConfig
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, use_causal_mask: bool = True) -> jnp.ndarray:
        B, T, C = x.shape # batch size, sequence length, embedding dimensionality (n_embd)
        # key, query, value projections for all heads, but in a batch
        qkv = nn.Dense(3 * self.config.n_embd, dtype=self.dtype)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        k = k.reshape(B, T, self.config.n_head, C // self.config.n_head).transpose(0, 2, 1, 3) # (B, nh, T, hs)
        q = q.reshape(B, T, self.config.n_head, C // self.config.n_head).transpose(0, 2, 1, 3) # (B, nh, T, hs)
        v = v.reshape(B, T, self.config.n_head, C // self.config.n_head).transpose(0, 2, 1, 3) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        attn = (q @ jnp.swapaxes(k, -2, -1)) * (1.0 / jnp.sqrt(k.shape[-1]))
        if use_causal_mask:
            mask = jnp.tril(jnp.ones((T, T), dtype=bool))
            attn = jnp.where(mask, attn, -jnp.inf)
        attn = nn.softmax(attn, axis=-1)
        y = attn @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = nn.Dense(self.config.n_embd, kernel_init=jax.random.normal(key=self.make_rng('params'), stddev=1/jnp.sqrt(self.config.n_embd)), dtype=self.dtype)(y)
        return y

class MLP(nn.Module):
    config: GPTConfig
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        intermediate_dim = 4 * self.config.n_embd
        x = nn.Dense(intermediate_dim, dtype=self.dtype)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.config.n_embd, kernel_init=jax.random.normal(key=self.make_rng('params'), stddev=1/jnp.sqrt(intermediate_dim)), dtype=self.dtype)(x)
        return x

class Block(nn.Module):
    config: GPTConfig
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        ln_1 = nn.LayerNorm(dtype=self.dtype)(x)
        attn_output = CausalSelfAttention(self.config, dtype=self.dtype)(ln_1)
        x += attn_output

        ln_2 = nn.LayerNorm(dtype=self.dtype)(x)
        mlp_output = MLP(self.config, dtype=self.dtype)(ln_2)
        x += mlp_output
        return x

class GPT(nn.Module):
    config: GPTConfig
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, idx, deterministic: bool = True):
        B, T = idx.shape
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        tok_emb = nn.Embed(num_embeddings=self.config.vocab_size, features=self.config.n_embd, embedding_init=jax.random.normal(key=self.make_rng('params'), stddev=0.02), dtype=self.dtype)(idx)
        positions = jnp.arange(0, T, dtype=jnp.int32)
        pos_emb = nn.Embed(num_embeddings=self.config.block_size, features=self.config.n_embd, embedding_init=jax.random.normal(key=self.make_rng('params'), stddev=0.02), dtype=self.dtype)(positions)
        x = tok_emb + pos_emb

        for _ in range(self.config.n_layer):
            x = Block(self.config, dtype=self.dtype)(x)

        x = nn.LayerNorm(dtype=self.dtype)(x)
        logits = nn.Dense(self.config.vocab_size, use_bias=False, kernel_init=nn.initializers.zeros_init(), dtype=self.dtype)(x) # Bias is False for weight sharing

        return logits

# Loss function
def loss_fn(logits, targets):
    """Cross-entropy loss function."""
    one_hot_targets = jax.nn.one_hot(targets, num_classes=logits.shape[-1])
    return jnp.mean(optax.softmax_cross_entropy(logits=logits, one_hot_labels=one_hot_targets))

# Training step
@jax.jit
def train_step(params, opt_state, batch, key, config: GPTConfig):
    """Performs a single training step."""
    inputs, targets = batch
    dropout_key = jax.random.fold_in(key, opt_state.count) # Use optimizer step count for distinct dropout keys

    def loss_value_fn(params):
        logits = GPT(config=config).apply({'params': params}, inputs, deterministic=False, rngs={'dropout': dropout_key})
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return loss

    grad_fn = jax.value_and_grad(loss_value_fn)
    loss, grads = grad_fn(params)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss


#data loader
class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        self.tokens = jnp.array(enc.encode(text))
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

    def next_batch(self, current_position):
        B, T = self.B, self.T
        start_index = current_position
        end_index = current_position + B * T + 1

        if end_index <= len(self.tokens):
            buf = self.tokens[start_index:end_index]
            x = buf[:-1].reshape(B, T)
            y = buf[1:].reshape(B, T)
            next_position = current_position + B * T
        else:
            # Handle wrapping
            remaining_len = len(self.tokens) - start_index
            needed_len = (B * T + 1) - remaining_len
            x1 = self.tokens[start_index:]
            y1 = self.tokens[start_index + 1 :]
            x2 = self.tokens[:needed_len-1]
            y2 = self.tokens[1:needed_len]
            x = jnp.concatenate([x1, x2]).reshape(B, T)
            y = jnp.concatenate([y1, y2]).reshape(B, T)
            next_position = needed_len -1

        return x, y, next_position










# Initialize model, optimizer
config = GPTConfig()
model = GPT(config=config)
key = jax.random.PRNGKey(0)
params_key, train_key = jax.random.split(key)

dummy_input = jnp.ones((1, config.block_size), dtype=jnp.int32)
variables = model.init({'params': params_key}, dummy_input)
params = variables['params']

learning_rate = 3e-4
optimizer = optax.adamw(learning_rate)
opt_state = optimizer.init(params)

# Replace dummy data with your data loader
def data_loader(batch_size, seq_length, vocab_size, key):
    # ... your data loading logic here ...
    # Yields batches of (inputs, targets) as JAX arrays
    while True:
        inputs = jax.random.randint(key, (batch_size, seq_length), 0, vocab_size)
        targets = jax.random.randint(key, (batch_size, seq_length), 0, vocab_size)
        yield (inputs, targets)
        key, _ = jax.random.split(key) # Update key for next batch (if needed)



# Dummy data for demonstration
batch_size = 4
seq_length = config.block_size
vocab_size = config.vocab_size
dummy_inputs = jax.random.randint(train_key, (batch_size, seq_length), 0, vocab_size)
dummy_targets = jax.random.randint(train_key, (batch_size, seq_length), 0, vocab_size)
dummy_batch = (dummy_inputs, dummy_targets)

# Training loop
num_epochs = 2
steps_per_epoch = 10
for epoch in range(num_epochs):
    for step in range(steps_per_epoch):
        train_key, subkey = jax.random.split(train_key)
        params, opt_state, loss = train_step(params, opt_state, dummy_batch, subkey, config)
        print(f"Epoch: {epoch+1}, Step: {step+1}, Loss: {loss:.4f}")