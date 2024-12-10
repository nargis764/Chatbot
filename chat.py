import random
from transformers import pipeline
import streamlit as st
import os
# Suppress TensorFlow info and warnings
TF_ENABLE_ONEDNN_OPTS = 0


# Define intents (simplified structure)
intents = [
    {"tag": "greeting", "responses": [
        "Hi there!", "Hello!", "Hey!", "I'm fine, thank you!", "Nothing much."]},
    {"tag": "goodbye", "responses": [
        "Goodbye!", "See you later!", "Take care!"]},
    {"tag": "thanks", "responses": [
        "You're welcome!", "No problem!", "Glad I could help!"]},
    {"tag": "help", "responses": ["Sure, what do you need help with?",
                                  "I'm here to help. What's the problem?", "How can I assist you?"]},
    {"tag": "about", "responses": [
        "I am a chatbot.", "My purpose is to assist you.", "I can answer questions and provide assistance."]}
]

# Hugging Face zero-shot classification pipeline
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

# Get all intent tags
intent_tags = [intent["tag"] for intent in intents]

# Chatbot function
if "conversation" not in st.session_state:
    st.session_state["conversation"] = []


def chatbot(input_text):
    # Use the zero-shot classifier to predict the intent
    result = classifier(input_text, candidate_labels=intent_tags)
    predicted_tag = result["labels"][0]
    confidence = result["scores"][0]

    # Check confidence threshold for fallback
    if confidence < 0.5:
        return "I'm sorry, I didn't understand that. Can you please rephrase?"

    # Find a matching response
    for intent in intents:
        if intent['tag'] == predicted_tag:
            return random.choice(intent['responses'])


def main():
    st.title("Chatbot with Transformers")
    st.write("Welcome! Start chatting.")

    # Get user input
    user_input = st.text_input("You:")
    if user_input:
        # Get chatbot response
        response = chatbot(user_input)

        # Append the conversation to session state
        st.session_state["conversation"].append(
            {"user": user_input, "bot": response})

    # Display the conversation history
    st.write("### Conversation History:")
    for convo in st.session_state["conversation"]:
        st.write(f"**You**: {convo['user']}")
        st.write(f"**Chatbot**: {convo['bot']}")


# Run the app
main()
