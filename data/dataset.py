from datasets import Dataset
from transformers import AutoTokenizer
from data.context import sample_texts

# Convert the text into a Dataset object
data = {"text": sample_texts}
dataset = Dataset.from_dict(data)

# Step 2.2: Load a pre-trained GPT tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Assign a padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use the EOS token as the pad token

# Step 2.3: Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=50)

# Apply the tokenizer to the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Display a tokenized example
print(tokenized_dataset[0])