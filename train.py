import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
from model import *
from dataclasses import dataclass
from typing import List
import matplotlib.pyplot as plt

@dataclass
class TrainOutput:
    loss_v: List[float]

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

def train(config):

    batch_size = 64
    num_epochs = 10
    
    rng = jax.random.PRNGKey(42)
    state = create_train_state(rng, config)
    loss_v = []
        
    for epoch in range(num_epochs):
        rng, data_rng = jax.random.split(rng)
        inputs, targets = get_batch(data_rng, batch_size, config['seq_len'], config['vocab_size'])
        state = train_step(state, inputs, targets)
        
        logits = state.apply_fn({'params': state.params}, inputs)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits[:, :-1], targets[:, 1:]).mean()

        loss_v.append(loss)
        print(f"Epoch {epoch+1}, Loss: {loss:.3f}")

    return TrainOutput(loss_v=loss_v)


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
    plt.plot(train(config).loss_v)
    plt.show()

if __name__ == "__main__":
    main()