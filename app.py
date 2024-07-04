import os
import torch
import transformers
from transformers import BertTokenizer, BertForSequenceClassification,AutoTokenizer
from flask import Flask, request, jsonify, render_template
from keras.models import load_model
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # Adjust num_labels as needed
model2=load_model('symptoms.h5',custom_objects={"TFBertModel": transformers.TFBertModel})

df=pd.read_csv('Symptom2Disease.csv')


le=LabelEncoder()

df['disease']=le.fit_transform(df['label'])
# Define intents and responses
intents = {
    0: "greet",
   
   
    1: "diagnose_symptoms"
}

responses = {
    "greet": "Hello! How can I assist you today?",
  
 
    "diagnose_symptoms": "Please describe your symptoms in detail."
}

# Define simple symptom-to-condition mappings
symptom_conditions = {
    "fever": ["Common Cold", "Flu", "COVID-19"],
    "cough": ["Common Cold", "Flu", "COVID-19", "Bronchitis"],
    "headache": ["Migraine", "Tension Headache", "Cluster Headache"],
    "sore throat": ["Common Cold", "Flu", "Strep Throat"]
}

# Function to classify intent
def classify_intent(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()
    return intents[predicted_class_id]

# Function to generate response
def generate_response(intent, user_input=None):
    if  user_input:
        conditions = diagnose_symptoms(user_input)
        return f"Based on your symptoms, you might have: {', '.join(conditions)}."
    return responses[intent]

# Function to diagnose symptoms
def diagnose_symptoms(symptoms):
    conditions = set()
    test1=tokenizer(
    text=symptoms,
    add_special_tokens=True,
    max_length=55,
    truncation=True,
    padding='max_length',
    return_tensors='tf',
    return_token_type_ids=False,
    return_attention_mask=True,
    verbose=True
    )
    predictions=model2.predict({'input_ids': test1['input_ids'], 'attention_mask':               test1['attention_mask']})
    predicted_class=np.argmax(predictions)
    predicted_class2=le.inverse_transform([predicted_class])

    
    conditions.update(predicted_class2)
    return conditions if conditions else ["No matching conditions found"]

# Create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    intent = classify_intent(user_input)
    response = generate_response(intent, user_input)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)

