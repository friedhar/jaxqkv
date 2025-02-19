import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state

class DecoderLayer(nn.Module):
    embed_dim: int
    num_heads: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        attn_out = nn.MultiHeadAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            )(x)
        x = nn.LayerNorm()(x + attn_out)
        
        ff = nn.Dense(self.hidden_dim)(x)
        ff = nn.relu(ff)
        ff = nn.Dense(self.embed_dim)(ff)
        x = nn.LayerNorm()(x + ff)

        return x

class Transformer(nn.Module):
    vocab_size: int
    embed_dim: int
    num_heads: int
    num_layers: int
    hidden_dim: int
    seq_len: int
    learning_rate: float

    @nn.compact
    def __call__(self, x):
        seq_len = x.shape[1]
        
        tok_embed = nn.Embed(self.vocab_size, self.embed_dim)(x)
        pos_embed = nn.Embed(seq_len, self.embed_dim)(jnp.arange(seq_len)[None, :])
        x = tok_embed + pos_embed
        
        for _ in range(self.num_layers):
            x = DecoderLayer(self.embed_dim, self.num_heads, self.hidden_dim)(x)
            
        return nn.Dense(self.vocab_size)(x)
