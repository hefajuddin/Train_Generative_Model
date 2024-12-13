from transformers import AutoModelForCausalLM

# Load a pre-trained GPT model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")