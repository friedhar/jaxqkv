import jax.numpy as jnp
from jax import Array
import jax

from dataclasses import dataclass

## q,k,v : (B, S, E)

@dataclass
class Config:
    vocab_size: int
    embedding_size: int 
    seq_len: int

key = jax.random.key(42)

class Transformer:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.theta_emb = jax.random.normal(key, self.config.vocab_size) 

    def __call__(self, x: Array) -> Array:
        x = jnp.matmul(x, self.theta_emb)
        return x

def main():
    print("init")

if __name__ == "__main__":
    main()
        

