import streamlit as st
import pandas as pd
import json
import openai
from langdetect import detect
from googletrans import Translator
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from datetime import datetime

# Set your OpenAI API key here
openai.api_key = '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'

# Load FAQ dataset from JSON
def load_faq_dataset_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as json_file:
        faq_dataset = json.load(json_file)
    return faq_dataset

faq_dataset = load_faq_dataset_from_json('qna_2.json')
topic_keywords = [entry['topic'] for entry in faq_dataset]

# Function to detect language and translate
def translate_to_english(text, target_lang):
    translator = Translator()
    translation = translator.translate(text, src=target_lang, dest="en")
    return translation.text

def translate_to_detected_language(text, target_lang):
    translator = Translator()
    translation = translator.translate(text, src="en", dest=target_lang)
    return translation.text

# Function to detect topic
def detect_topic(user_input, faq_dataset):
    detected_topic = None
    for keyword in topic_keywords:
        if keyword.lower() in user_input.lower():
            detected_topic = keyword
            break
    return detected_topic

# Function to calculate similarity
def calculate_similarity(user_input, faq_questions):
    documents = [user_input] + faq_questions
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    cosine_similarities = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:])[0]
    most_similar_index = cosine_similarities.argmax()
    return most_similar_index

# Function to train topic classifier
def train_topic_classifier(faq_dataset):
    examples = []
    labels = []
    for entry in faq_dataset:
        topic = entry['topic']
        questions = entry['question']
        for question in questions:
            examples.append(question)
            labels.append(topic)
    text_clf = Pipeline([
        ('vectorizer', CountVectorizer()),  # You can use TF-IDF or other vectorizers as well
        ('classifier', MultinomialNB())
    ])
    text_clf.fit(examples, labels)
    return text_clf

# Function to get bot response
def get_bot_response(user_input, detected_language, faq_dataset, topic_classifier):
    detected_topic = detect_topic(translate_to_english(user_input, detected_language), faq_dataset)
    if detected_topic:
        matching_topic_entry = next((entry for entry in faq_dataset if entry['topic'] == detected_topic), None)
        if matching_topic_entry:
            faq_questions = matching_topic_entry['question']
            most_similar_index = calculate_similarity(user_input, faq_questions)
            bot_response = matching_topic_entry['answer']
        else:
            prompt = f"\nTopic: {user_input}\n\nChatbot:"
            response = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=500)
            bot_response = response.choices[0].text.strip()
    else:
        detected_topic = topic_classifier.predict([user_input])[0]
        if detected_topic:
            matching_topic_entry = next((entry for entry in faq_dataset if entry['topic'] == detected_topic), None)
            if matching_topic_entry:
                faq_questions = matching_topic_entry['question']
                most_similar_index = calculate_similarity(user_input, faq_questions)
                bot_response = matching_topic_entry['answer']
            else:
                prompt = f"\nTopic: {user_input}\n\nChatbot:"
                response = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=500)
                bot_response = response.choices[0].text.strip()
        else:
            prompt = f"\nTopic: {user_input}\n\nChatbot:"
            response = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=500)
            bot_response = response.choices[0].text.strip()
    bot_response_translated = translate_to_detected_language(bot_response, detected_language)
    return bot_response_translated

# Streamlit app
def main():
    st.title('Chatbot with Streamlit')

    faq_dataset = load_faq_dataset_from_json('qna_2.json')
    topic_classifier = train_topic_classifier(faq_dataset)

    user_input = st.text_input('Enter your message:')
    if st.button('Send'):
        detected_language = detect(user_input)
        bot_response = get_bot_response(user_input, detected_language, faq_dataset, topic_classifier)
        st.text(f'Bot Response: {bot_response}')

if __name__ == '__main__':
    main()
