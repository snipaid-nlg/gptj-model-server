from transformers import GPTJConfig, AutoTokenizer, models, pipeline
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
    checkpoint = torch.hub.load_state_dict_from_url('https://huggingface.co/snipaid/gptj-title-teaser-10k/resolve/main/pytorch_model.pt', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    print("done")

    # conditionally load to GPU
    if device == "cuda:0":
        print("loading model to GPU...")
        model.cuda()
        print("done")

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    config.pad_token_id = config.eos_token_id
    tokenizer.pad_token = config.pad_token_id


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    global tokenizer

    # Parse out your arguments
    prompt = model_inputs.pop('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}
    
    # Initialize pipeline
    gen_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0, **model_inputs)

    # Run generation pipline
    output = gen_pipe(prompt)

    # Get output text
    output_text = output[0]['generated_text']

    result = {"output": output_text}

    # Return the results as a dictionary
    return result
