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

class AttentionBlock:
    def __init__(self, config: Config) -> None:
        self.config  = config

    def __call__(self, q: Array, k: Array, v: Array) -> Array:
        return jnp.matmul(jax.nn.softmax(jnp.matmul(q,jnp.swapaxes(k,-2,-1))/jnp.sqrt(k.shape[-1]),axis=1),v) ## ONE LINER JUSTIFICATION: computation doesn't change, constant.
            
        
# class TwoLayerMlp:
#     def __init__(self, config: Config, theta: Array = None) -> None:
#         self.theta0 = theta0 if theta0 else jax.random.normal(key, ())
#         self.config = config

#     def __call__(self, x: Array) -> Array:
#         return (x - x.mean()) / x.std()
        

class Transformer:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.theta_emb = jax.random.normal(key, (self.config.vocab_size, self.config.embedding_size)) 
        self.theta_qkv = jax.random.normal(key, (self.config.embedding_size, 3 * self.config.embedding_size))  ## 3 * emb_size since both q, k, v are of shape emb_size

        self.theta_fc1 = jax.random.normal(key, (self.config.seq_len*self.config.embedding_size,  4*self.config.seq_len*self.config.embedding_size   ))
        self.theta_fc2 = jax.random.normal(key, (4*self.config.seq_len*self.config.embedding_size,  self.config.seq_len*self.config.embedding_size   ))

        self.block = AttentionBlock(config)

    def __call__(self, x: Array) -> Array:
        assert len(x.shape) == 2
        assert x.shape[1] == self.config.seq_len 

        x = jax.nn.one_hot(x, self.config.vocab_size)
        x = jnp.matmul(x, self.theta_emb)

        qkv_fused = jnp.matmul(x, self.theta_qkv)

        q, k, v = jnp.split(qkv_fused, 3, axis=2)


        x = self.block(q, k, v)
        assert len(x.shape) == 3
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
        print(x.shape)
    
        


        return x

def main():
    print("init")
    transformer = Transformer(Config(vocab_size=16, embedding_size=32, seq_len=4))
    (transformer(jnp.array([[1, 2, 3, 4]])))

if __name__ == "__main__":
    main()
        

