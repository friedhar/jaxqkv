# jaxqkv - Transformer Implementation & Baseline Expiriments In Jax

jaxqkv is a transformer framework in `jax`, with small amounts of `flax` (otimized layers) & `optax` (advance grad optimizers). The general goal is for it to be used both as a reference implementation, learning resource & general playground.

A sample loader & tokenizer of the [TinyStories]() is provided, and training on different datasets should be pretty low overhead.

## Setup
```
git clone https://github.com/friedhar/jaxqkv.git 
cd jaxqkv
```


## TODO - Possible Roadmap
* Flash Attention Support
* RoPe Token Embedding
* Non-Naive KV Cache
* Support for GRPO
* Full Training Run With Fineweb