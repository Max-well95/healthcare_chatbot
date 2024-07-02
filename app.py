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
    2: "schedule_appointment"
}

responses = {
    "greet": "Hello! How can I assist you today?",
    "ask_medication": "Please provide the name of the medication.",
    "schedule_appointment": "Sure, I can help you with scheduling an appointment. Please provide the date and time."
}

# Function to classify intent
def classify_intent(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()
    return intents[predicted_class_id]

# Function to generate response
def generate_response(intent):
    return responses[intent]

# Create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    intent = classify_intent(user_input)
    response = generate_response(intent)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)

