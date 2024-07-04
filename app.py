import torch
from transformers import BertTokenizer, BertForSequenceClassification
from flask import Flask, request, jsonify, render_template

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)  # Adjust num_labels as needed

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

# Function to classify intent
def classify_intent(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()
    return intents[predicted_class_id]

# Function to generate response
def generate_response(intent, user_input=None):
    if intent == "diagnose_symptoms" and user_input:
        conditions = diagnose_symptoms(user_input)
        return f"Based on your symptoms, you might have: {', '.join(conditions)}."
    return responses[intent]

# Function to diagnose symptoms
def diagnose_symptoms(symptoms):
    conditions = set()
    for symptom in symptoms.split():
        if symptom in symptom_conditions:
            conditions.update(symptom_conditions[symptom])
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

