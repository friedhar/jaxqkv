import jax.numpy as jnp
import jax.nn as nn
from jax import Array
import jax
## q,k,v : (B, S, E)

key = jax.random.key(42)
