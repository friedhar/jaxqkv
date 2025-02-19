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

@dataclass
class Thetas:
    emb: Array
    qkv:  Array
    fc1: Array
    fc2: Array

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
    def __init__(self, config: Config):
        self.config = config
        self.block = AttentionBlock(config)

    def gen_thetas(self):
        theta_emb = jax.random.normal(key, (self.config.vocab_size, self.config.embedding_size)) 
        theta_qkv = jax.random.normal(key, (self.config.embedding_size, 3 * self.config.embedding_size))  ## 3 * emb_size since both q, k, v are of shape emb_size

        theta_fc1 = jax.random.normal(key, (self.config.seq_len*self.config.embedding_size,  4*self.config.seq_len*self.config.embedding_size   ))
        theta_fc2 = jax.random.normal(key, (4*self.config.seq_len*self.config.embedding_size, self.config.vocab_size  ))
        return Thetas(emb=theta_emb, qkv=theta_qkv, fc1=theta_fc1, fc2=theta_fc2)



    def __call__(self, x: Array, theta_emb: Array, theta_qkv: Array, theta_fc1:  Array, theta_fc2: Array) -> Array:
        assert len(x.shape) == 2
        assert x.shape[1] == self.config.seq_len 

        x = jax.nn.one_hot(x, self.config.vocab_size)
        x = jnp.matmul(x, theta_emb)

        qkv_fused = jnp.matmul(x, theta_qkv)

        q, k, v = jnp.split(qkv_fused, 3, axis=2)


        x = self.block(q, k, v)
        assert len(x.shape) == 3
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
        x = jnp.matmul(x, theta_fc1)
        x = jnp.matmul(x, theta_fc2)
        x = x / (1.0 - x)
        x = jax.nn.softmax(x)

        return x

def main():
    print("init")
    transformer: Transformer = Transformer(Config(vocab_size=16, embedding_size=32, seq_len=4))
    thetas: Thetas = transformer.gen_thetas()

    X = jnp.array([
        [i, i+1, i+2, i+3]
        for i in range(1024)
    ])

    y = jnp.array([
        [i+4]
        for i in range(1024)
    ])

    def mse_loss(X, *args):
        y_hat = transformer(X, *args)
        return ((y_hat-y)**2).mean()

    steps = 1_000
    for step in range(steps):
        grad = jax.grad(mse_loss, allow_int=True)(X, thetas.emb, thetas.qkv, thetas.fc1, thetas.fc2)
        print(grad.shape)
        



if __name__ == "__main__":
    main()
        

