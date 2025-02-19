import tiktoken
import json

def main():
    enc = tiktoken.get_encoding("o200k_base")

    print(f"vocab size: {enc.n_vocab}")
    with open("./data_raw_tiny_samples/samplegamma.txt", "r") as f:
        x = f.read()

    encoded = enc.encode(x)
    print(encoded)

    metadata = {
        "vocab_size": enc.n_vocab
    }

    with open("./data_parsed/samplegamma.metdata.json", "w") as f: f.write(json.dumps(metadata))

    print("Finished.")

if __name__ == "__main__":
    main()