import tiktoken
import jax.numpy as jnp
import numpy as np
import json

def process_naive(dataset_name: str, txt: str, seq_len: int = 1024, samples_limit: int = None):
    enc = tiktoken.get_encoding("o200k_base")
    tokens = enc.encode(txt)
    print(f"len(tokens): {len(tokens)}")

    inputs = []
    targets = []

    for i in range(seq_len, len(tokens)-seq_len):
        if samples_limit and len(inputs) >= samples_limit: break

        xs = tokens[i-seq_len:i]
        print(len(xs))
        y_i =tokens[i:i+seq_len] 
        print()
        inputs.append(xs)
        targets.append(y_i)

    assert len(inputs) == len(targets)

    inputs = jnp.array(inputs)

    targets = jnp.array(targets)

    metadata = {
        "vocab_size": enc.n_vocab
    }
    print(inputs.shape)
    print(targets.shape)
    ## dump to IO
    np.save(f"./data_parsed/{dataset_name}_inputs", inputs)
    np.save(f"./data_parsed/{dataset_name}_targets", targets)
    with open(f"./data_parsed/{dataset_name}_metadata.json", "w") as f: f.write(json.dumps(metadata))

    

