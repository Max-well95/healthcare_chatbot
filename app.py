import torch
from transformers import BertTokenizer, BertForSequenceClassification
from flask import Flask, request, jsonify, render_template
import logging
from functools import lru_cache
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load pre-trained model and tokenizer
@lru_cache(maxsize=None)
def load_model():
    try:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
        return tokenizer, model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

tokenizer, model = load_model()

# Define intents and responses
intents = {
    0: "greet",
    1: "ask_medication",

    2: "diagnose_symptoms"
}

responses = {
    "greet": "Hello! How can I assist you today?",
    "ask_medication": "Please provide the name of the medication.",
 
    "diagnose_symptoms": "Please describe your symptoms in detail."
}

# Define simple symptom-to-condition mappings
symptom_conditions = {
    "fever": ["Common Cold", "Flu", "COVID-19"],
    "cough": ["Common Cold", "Flu", "COVID-19", "Bronchitis"],
    "headache": ["Migraine", "Tension Headache", "Cluster Headache"],
    "sore throat": ["Common Cold", "Flu", "Strep Throat"]
}
def is_greeting(text):
    greetings = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]
    return any(greeting in text.lower() for greeting in greetings)

# Function to classify intent
def classify_intent(text):
    try:
        # First, check if it's a greeting
        if is_greeting(text):
            return "greet"

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        predicted_class_id = torch.argmax(logits, dim=1).item()
        confidence = probabilities[0][predicted_class_id].item()
        
        if confidence < 0.5:  # Adjust this threshold as needed
            return "unknown"
        return intents[predicted_class_id]
    except Exception as e:
        logger.error(f"Error in intent classification: {str(e)}")
        return "unknown"


# Function to generate response
def generate_response(intent, user_input=None):
    if intent == "diagnose_symptoms" and user_input:
        conditions = diagnose_symptoms(user_input)
        return f"Based on your symptoms, you might have: {', '.join(conditions)}. Please consult a healthcare professional for accurate diagnosis."
    elif intent == "unknown":
        return "I'm not sure I understand. Could you please rephrase your question?"
    return responses.get(intent, "I'm sorry, I don't have a response for that.")

# Function to diagnose symptoms
def diagnose_symptoms(symptoms):
    conditions = set()
    for symptom in re.findall(r'\b\w+\b', symptoms.lower()):
        if symptom in symptom_conditions:
            conditions.update(symptom_conditions[symptom])
    return list(conditions) if conditions else ["No matching conditions found"]
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

