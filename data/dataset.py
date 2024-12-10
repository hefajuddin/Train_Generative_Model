from datasets import Dataset
from data.context import sample_texts

# Convert the text into a Dataset object
data = {"text": sample_texts}
dataset = Dataset.from_dict(data)
