# jaxqkv - Transformer Implementation & Baseline Expiriments In Jax

jaxqkv is a transformer framework in `jax`, with small amounts of `flax` (otimized layers) & `optax` (advance grad optimizers). The general goal is for it to be used both as a reference implementation, learning resource & general playground.

A sample loader & tokenizer of the [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories/tree/main) is provided, and training on different datasets should be pretty low overhead.

## Setup
```
git clone https://github.com/friedhar/jaxqkv.git 
cd jaxqkv
chmod +x setup_env.sh
./setup_env.sh
```


## TODO - Possible Roadmap
* Flash Attention Support
* RoPe Token Embedding
* Non-Naive KV Cache
* Support for GRPO
* Full Training Run With Fineweb