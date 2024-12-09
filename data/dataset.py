from datasets import Dataset
from transformers import AutoTokenizer
from data.context import sample_texts

# Convert the text into a Dataset object
data = {"text": sample_texts}
dataset = Dataset.from_dict(data)

# Step 2.2: Load a pre-trained GPT tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Step 2.3: Tokenize the dataset
def tokenize_function():
    return tokenizer(data, padding="max_length", truncation=True, max_length=20)

# Apply the tokenizer to the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Display a tokenized example
print(tokenized_dataset[0])