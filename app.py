from transformers import GPTJConfig, GPT2Tokenizer, models
import torch

from utils import GPTJBlock, GPTJForCausalLM, add_adapters

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Init is ran on server startup
# Load your model to GPU as a global variable here.
def init():
    global model
    global tokenizer

    print("patching for 8bit...")
    models.gptj.modeling_gptj.GPTJBlock = GPTJBlock  # monkey-patch GPT-J

    print("loading config...")
    config = GPTJConfig.from_pretrained("EleutherAI/gpt-j-6B")
    model = GPTJForCausalLM(config=config)
    
    print("adding LoRA adapters...")
    add_adapters(model)

    print("loading model to CPU...")
    checkpoint = torch.hub.load_state_dict_from_url('https://h2858852.stratoserver.net/snipaid/gptj-title-teaser-1k.pt')
    model.load_state_dict(checkpoint)
    model.eval()
    print("done")

    # conditionally load to GPU
    if device == "cuda:0":
        print("loading model to GPU...")
        model.cuda()
        print("done")

    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    config.pad_token_id = config.eos_token_id
    tokenizer.pad_token = config.pad_token_id


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    global tokenizer

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}
    
    # Tokenize inputs
    input_tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Run the model
    output = model.generate(input_tokens)

    # Decode output tokens
    output_text = tokenizer.batch_decode(output, skip_special_tokens = True)[0]

    result = {"output": output_text}

    # Return the results as a dictionary
    return result
