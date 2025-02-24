import tensorflow as tf
from tensorflow.keras.models import Model,load_model
import pandas as pd
import numpy as np
import json
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
import random
with open('chat-bot-data.json') as intents:
    data=json.load(intents)
tags=[]
inputs=[]
responses={}
for intent in data['intents']:
    responses[intent['tag']]=intent['responses']
    for lines in intent['patterns']:
        inputs.append(lines)
        tags.append(intent['tag'])
df=pd.DataFrame({'Inputs':inputs,'tags':tags})
def tokenize_data(tokenizer, texts):
  tokenized_texts = []
  for text in texts:
    tokens = tokenizer(text, padding='max_length', truncation=True, max_length=70)['input_ids']
    tokenized_texts.append(tokens)

  return np.array(tokenized_texts)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
le=LabelEncoder()
y=le.fit_transform(df['tags'])
model=load_model('intents-chatbot.keras')
def chat():
    while True:
        Input = (input('You: '))
        Input = tokenize_data(tokenizer, np.array([Input]))
        prediction = model.predict(Input)
        output = prediction.argmax()
        response_tag = le.inverse_transform([output])[0]
        print(f"Predicted tag: {response_tag}") 
        responses = next(intent['responses'] for intent in data["intents"] if intent['tag'] == response_tag)
        print('Chatbot:', random.choice(responses))
chat()