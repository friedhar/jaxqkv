import tiktoken
import json
from data_prep import process_naive

def main(seq_len: int = 32):
    with open("./data_raw_tiny_samples/samplegamma.txt", "r") as f:x=f.read()
    process_naive(dataset_name="samplegamma", txt=x, seq_len=seq_len, samples_limit=1024)

    print("Finished.")

if __name__ == "__main__":
    main()