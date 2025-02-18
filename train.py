import jax.numpy as jnp
import jax.nn as nn
from jax import Array
import jax
## q,k,v : (B, S, E)

key = jax.random.key(42)


BATCHS_SIZE = 64
SEQ_LEN = 8
EMB_SIZE = 32

def add_norm(x: Array) -> Array:
    return (x - x.mean()) / x.std()

def embedding(x: Array, w_emb_i: Array) -> Array:
    return  jnp.matmul(x, w_emb_i)
    
def scaled_dot_attention(q: Array, k: Array, v: Array) -> Array:
    lhs = nn.softmax(jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / jnp.sqrt(k.shape[-1]), axis=1)
    return lhs.dot(v)


q =  jax.random.normal(key, (BATCHS_SIZE,SEQ_LEN,EMB_SIZE))
k =  jax.random.normal(key, (BATCHS_SIZE,SEQ_LEN,EMB_SIZE))
v =  jax.random.normal(key, (BATCHS_SIZE,SEQ_LEN,EMB_SIZE))

print(q.shape)

print(scaled_dot_attention(q,k,v).shape)
# k =  jax.random.normal(123, (BATCHS_SIZE,SEQ_LEN,EMB_SIZE))

# x = jnp.arange(5.0)
# print(q)