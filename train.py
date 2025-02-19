import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
from model import *
from dataclasses import dataclass
from typing import List
import matplotlib.pyplot as plt
from jax import Array
import numpy as np
import json

@dataclass
class TrainOutput:
    loss_v: List[float]

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
            logits[:, :-1], targets[:, :-1]).mean()
        return loss
    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)

def train(config, inputs: Array, targets: Array, num_epochs: int = 100, batch_size: int = 64):
    rng = jax.random.PRNGKey(42)
    state = create_train_state(rng, config)
    loss_v = []

        
    for epoch in range(num_epochs):
        rng, data_rng = jax.random.split(rng)
        # inputs, targets = get_batch(data_rng, batch_size, config['seq_len'], config['vocab_size'])
        print("shape_i: ", inputs.shape)
        print("shape_t: ", targets.shape)
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

def read_data_parsed(dataset: str):
    with open(f"./data_parsed/{dataset}_metadata.json","r") as f:metadata=json.loads(f.read())
    inputs = jnp.array(np.load(f"./data_parsed/{dataset}_inputs.npy"))
    targets = jnp.array(np.load(f"./data_parsed/{dataset}_targets.npy"))

    return inputs, targets, metadata["vocab_size"]
    

def main():
    inputs, targets, vocab_size = read_data_parsed("samplegamma")
    inputs = inputs[:32]
    targets = targets[:32]
    print(targets.dtype)
    config = {
        'vocab_size': vocab_size,
        'seq_len': 32,
        'embed_dim': 32,
        'num_heads': 2,
        'num_layers': 2,
        'hidden_dim': 256,
        'learning_rate': 1e-2,
    }


    train(config=config, inputs=inputs, targets=targets)



if __name__ == "__main__":
    main()