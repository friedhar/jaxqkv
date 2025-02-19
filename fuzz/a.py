import tiktoken
import json

def main():
    enc = tiktoken.get_encoding("o200k_base")

    print(enc.n_vocab)
    print(enc.encode("hello th'e"))
    with open("./fuzz/deeplearning_wiki.txt", "r") as f:
        x = f.read()

    encoded = enc.encode(x)
    print(encoded)

    o = {
        "metadata": {
            "vocab_size": enc.n_vocab
        },
        "data": encoded
    }

    with open("./data_parsed/samplegamma.txt", "w") as f: f.write(json.dumps(o))
    print("Finished.")

if __name__ == "__main__":
    main()

# To get the tokeniser corresponding to a specific model in the OpenAI API:
# enc = tiktoken.encoding_for_model("gpt-4o")