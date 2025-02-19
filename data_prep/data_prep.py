import tiktoken
import jax.numpy as jnp
import json

def process_naive(dataset_name: str, txt: str, seq_len: int = 1024):
    enc = tiktoken.get_encoding("o200k_base")
    tokens = enc.encode(txt)
    print(f"len(tokens): {len(tokens)}")

    inputs = []
    targets = []

    for i in range(seq_len, len(tokens)):
        xs = tokens[i-seq_len:i]
        y_i = tokens[i]
        print()
        inputs.append(xs)
        targets.append(y_i)

    metadata = {
        "vocab_size": enc.n_vocab
    }

    ## dump to IO
    jnp.save(f"./data_parsed/{dataset_name}_inputs.jnp", inputs)
    jnp.save(f"./data_parsed/{dataset_name}_targets.jnp", targets)
    with open(f"./data_parsed/{dataset_name}_metadata.json", "w") as f: f.write(json.dumps(metadata))

    

