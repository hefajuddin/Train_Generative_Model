from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

generator = pipeline("text-generation", model="./fine_tuned_gpt")

# Generate text
output = generator("Who was witch?", max_length=100, num_return_sequences=1)

# Print the generated text
print(output[0]["generated_text"])


# predefined_contexts = {
#     "Who was lost?": "Rapunzel was lost in the forest and could not find her way back to the tower.",
#     # Add more questions and contexts
# }

# question = "Who was lost?"
# context = predefined_contexts.get(question, "")

# if context:
#     from transformers import pipeline

#     # Load the fine-tuned QA model
#     qa_pipeline = pipeline("question-answering", model="./fine_tuned_gpt", tokenizer="./fine_tuned_gpt")

#     # Get the answer
#     output = qa_pipeline(question=question, context=context)

#     print(f"Answer: {output['answer']}")
# else:
#     print("No relevant context found for the question.")