# Train_Generative_Model
1. Install requirment using following commands-
pip install -r requirements.txt

2. Run cmd on your project directory and execute the project using following command-
python app.py

3. Input your prompt regarding Rapunzel Story into- "Input to generate from Rapunzel story:"

4. Now you will see generated text


How the project deveoloped
================================
1. Keep my context (here it's a story named Rapunzel) in data/context.py
2. Crate dataset in data/dataset.py
3. Use Data-Collator to prepare batches of data for training GPT2 Model in data/collator.py
4. Load pretrained model GPT2 in pretrained_model.py
5. Tokenizing context(Rapunzel story) using gpt2 tokenizer in model_tokenize.py
6. Train and save the model using transformer's Trainer in Trainer.py
7. After training complete, upload the model into Hugging face hub named 'hefajuddin/Rapunzel_Story_Gen' in upload.py
8. Use the newly created model to generate texts from Rapunzel story

* That's finish
* Model will be optimize, fine-tuned and developed more later

