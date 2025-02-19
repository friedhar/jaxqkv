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

def get_batch(rng, batch_size, seq_len, vocab_size):
    data = jax.random.randint(rng, (batch_size, seq_len + 1), 0, vocab_size)
    return data[:, :-1], data[:, 1:]

def create_train_state(rng, config):
    model = Transformer(**config)
    params = model.init(rng, jnp.ones((1, config['seq_len']), dtype=jnp.int32))['params']
    tx = optax.adam(config['learning_rate'])
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, inputs, targets):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, inputs)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits[:, :-1], targets[:, 1:]).mean()
        return loss
    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)

def main():
    config = {
        'vocab_size': 100,
        'seq_len': 32,
        'embed_dim': 128,
        'num_heads': 4,
        'num_layers': 2,
        'hidden_dim': 512,
        'learning_rate': 0.001,
    }
    batch_size = 64
    num_epochs = 10
    
    rng = jax.random.PRNGKey(42)
    state = create_train_state(rng, config)
    
    for epoch in range(num_epochs):
        rng, data_rng = jax.random.split(rng)
        inputs, targets = get_batch(data_rng, batch_size, config['seq_len'], config['vocab_size'])
        state = train_step(state, inputs, targets)
        
        # Calculate training loss
        logits = state.apply_fn({'params': state.params}, inputs)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits[:, :-1], targets[:, 1:]).mean()
        print(f"Epoch {epoch+1}, Loss: {loss:.3f}")

if __name__ == "__main__":
    main()