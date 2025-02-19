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
        self.theta_emb = jax.random.normal(key, (self.config.vocab_size, self.config.embedding_size)) 

    def __call__(self, x: Array) -> Array:
        assert len(x.shape) == 2
        assert x.shape[1] == self.config.seq_len 

        x = jax.nn.one_hot(x, self.config.vocab_size)
        x = jnp.matmul(x, self.theta_emb)
        return x

def main():
    print("init")
    transformer = Transformer(Config(vocab_size=16, embedding_size=32, seq_len=1))
    print(transformer(jnp.array([[1]])))

if __name__ == "__main__":
    main()
        

