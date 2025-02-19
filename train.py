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

def train(config, num_epochs: int = 100, batch_size: int = 64):
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

def print_train_header(config, hr_width: int = 32) -> None:
    print("-"*hr_width)
    print("Jaxqkv :: TRAINING")
    print("Avaialble Local Devices: ", jax.local_devices())
    print("Config: ", config)
    print("-"*hr_width)

def main():
    lr_v = [ 2**5,2**6,2**7,2**8]
    print(lr_v)
    for lr_i in lr_v:
        config = {
            'vocab_size': 1024,
            'seq_len': 32,
            'embed_dim': 32,
            'num_heads': 2,
            'num_layers': 2,
            'hidden_dim': lr_i,
            'learning_rate': 1e-2,
        }
        print_train_header(config)


        plt.plot(train(config, num_epochs=64).loss_v, label=f"{lr_i}")

    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()