from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# # Load the tokenizer and model from the fine_tuned_gpt folder
# tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_gpt")
# model = AutoModelForCausalLM.from_pretrained("./fine_tuned_gpt", weights_format="safetensors")


generator = pipeline("text-generation", model="./fine_tuned_gpt")

# # Initialize the pipeline
# generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Generate text
output = generator("Tell me about Bangladeshi stock market,", max_length=50, num_return_sequences=1)

# Print the generated text
print(output[0]["generated_text"])