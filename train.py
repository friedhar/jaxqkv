import jax.numpy as jnp
from jax import Array
import jax

from dataclasses import dataclass

## q,k,v : (B, S, E)

@dataclass
class Config:
    vocab_size: int
    embedding_size: int 

key = jax.random.key(42)

class Transformer:
    def __init__(self, config: Config) -> None:
        self.config = config

