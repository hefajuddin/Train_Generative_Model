from model_tokenize import tokenizer
from transformers import DataCollatorForLanguageModeling

# Define a data collator for causal language modeling (no masking)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Masked Language Modeling is not used for GPT
)