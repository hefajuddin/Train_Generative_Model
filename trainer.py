from transformers import TrainingArguments
from transformers import Trainer
from pretrained_model import model
from model import tokenized_dataset
from data.collator import data_collator
from model import tokenizer

# Define training arguments
training_args = TrainingArguments(
    # output_dir="./gpt_model",       # Directory to save the model
    overwrite_output_dir=True,     
    num_train_epochs=3,            # Number of epochs
    per_device_train_batch_size=2, # Batch size per GPU
    save_steps=500,                # Save checkpoints every 500 steps
    save_total_limit=2,            # Keep the last 2 checkpoints
    logging_dir="./logs",          # Logging directory
    logging_steps=10               # Log every 10 steps
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)


# trainer.train()
# # Save the fine-tuned model
# model.save_pretrained("./fine_tuned_gpt")
# # Save the tokenizer
# tokenizer.save_pretrained("./fine_tuned_gpt")

# print("Model and tokenizer have been saved to './fine_tuned_gpt'")
