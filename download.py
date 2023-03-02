# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: our gpt-j-6B model finetuned for title and teaser generation

from transformers import GPTJConfig, AutoTokenizer
import torch

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    print("downloading model checkpoint...")
    torch.hub.load_state_dict_from_url('https://huggingface.co/snipaid/gptj-title-teaser-10k/resolve/main/pytorch_model.pt', map_location=torch.device('cpu'))
    print("done")

    print("downloading model config...")
    GPTJConfig.from_pretrained("EleutherAI/gpt-j-6B")

    print("downloading tokenizer...")
    AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    print("done")

if __name__ == "__main__":
    download_model()