from huggingface_hub import hf_hub_download

REPO_ID = "YOUR_REPO_ID"
FILENAME = "data.csv"

dataset =
    hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset")