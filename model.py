from transformers import AutoModelForCausalLM

# Load a pre-trained GPT model
model = AutoModelForCausalLM.from_pretrained("gpt2")