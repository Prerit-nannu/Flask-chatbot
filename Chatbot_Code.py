from flask import Flask, render_template, request, jsonify
import pandas as pd
import json
import openai
from langdetect import detect
from googletrans import Translator
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from datetime import datetime

app = Flask(__name__)

# Set your OpenAI API key here
openai.api_key = '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'

# Set your Google Sheets credentials file (JSON)
google_sheets_credentials_file = 'big-maxim-349418-473b76f78d80.json'


def load_faq_dataset_from_json(file_path):
    with open(file_path, 'r') as json_file:
        faq_dataset = json.load(json_file)
    return faq_dataset

faq_dataset = load_faq_dataset_from_json('qna_2.json')
topic_keywords = [entry['topic'] for entry in faq_dataset]

# Define CSV file for feedback
feedback_csv = 'feedback.csv'

def translate_to_english(text, target_lang):
    translator = Translator()
    translation = translator.translate(text, src=target_lang, dest="en")
    return translation.text

def translate_to_detected_language(text, target_lang):
    translator = Translator()
    translation = translator.translate(text, src="en", dest=target_lang)
    return translation.text

def detect_topic(user_input, faq_dataset):
    detected_topic = None
    for keyword in topic_keywords:
        if keyword.lower() in user_input.lower():
            detected_topic = keyword
            break
    return detected_topic

def calculate_similarity(user_input, faq_questions):
    documents = [user_input] + faq_questions
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    cosine_similarities = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:])[0]
    most_similar_index = cosine_similarities.argmax()
    return most_similar_index

def train_topic_classifier():
    # Load and preprocess the data
    examples = []
    labels = []
    for entry in faq_dataset:
        topic = entry['topic']
        questions = entry['question']
        for question in questions:
            examples.append(question)
            labels.append(topic)
    # Create a pipeline for text classification
    text_clf = Pipeline([
        ('vectorizer', CountVectorizer()),  # You can use TF-IDF or other vectorizers as well
        ('classifier', MultinomialNB())
    ])
    # Train the model
    text_clf.fit(examples, labels)
    return text_clf

def send_feedback_to_sheets(user_input, bot_response, user_date, feedback_data, google_sheets_credentials_file):
    # Google Sheets integration
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/spreadsheets",
             "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
    credentials = ServiceAccountCredentials.from_json_keyfile_name(google_sheets_credentials_file, scope)
    client = gspread.authorize(credentials)

    spreadsheet_title = "Feedback_AGH"  # Replace with your actual spreadsheet title
    worksheet_title = "Sheet1"  # Use the index of the worksheet you want to write to

    try:
        # Try to open the spreadsheet, or create it if it doesn't exist
        spreadsheet = client.open(spreadsheet_title)
    except gspread.SpreadsheetNotFound:
        spreadsheet = client.create(spreadsheet_title)

    worksheet = None
    try:
        worksheet = spreadsheet.worksheet(worksheet_title)
    except gspread.WorksheetNotFound:
        # If the worksheet doesn't exist, create it
        worksheet = spreadsheet.add_worksheet(worksheet_title, 1, 1)

    data = {'Date': [user_date],
            'User Input': [user_input],
            'Bot Response': [bot_response],
            'Feedback': [feedback_data]}
    df = pd.DataFrame(data)

    # Convert DataFrame to a list of lists
    values = df.values.tolist()

    # Append data to the worksheet
    worksheet.insert_rows(values, 2)  # Change the row index as needed

#@app.route('/')
#def index():
 #   return render_template('chat.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('chat_3.html')


@app.route('/process_message', methods=['POST'])
def process_message():
 # try:

    topic_classifier = train_topic_classifier()
    #user_input = request.form.get('user_input')
    if request.method == 'POST':
        user_input = request.form['user_input']
    
    
    
    # Detect the language of the user's input
    detected_language = detect(user_input)

    # Translate user's input to English (assuming English is the common language)
    user_input_english = translate_to_english(user_input, detected_language)

    # Detect the topic
    detected_topic = detect_topic(user_input_english, faq_dataset)

    if detected_topic:
        matching_topic_entry = None
        for entry in faq_dataset:
            if entry["topic"] == detected_topic:
                matching_topic_entry = entry
                break

        if matching_topic_entry:
            faq_questions = matching_topic_entry["question"]
            most_similar_index = calculate_similarity(user_input_english, faq_questions)
            faq_answer = matching_topic_entry["answer"]
            bot_response_english = faq_answer
        else:
            prompt = f"\nTopic: {user_input_english}\n\nChatbot:"
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=500
            )
            bot_response_english = response.choices[0].text.strip()

        # Translate the bot's response back to the detected language
        bot_response_translated = translate_to_detected_language(bot_response_english, detected_language)
    else:
        detected_topic = topic_classifier.predict([user_input_english])[0]

        if detected_topic:
            matching_topic_entry = None
            for entry in faq_dataset:
                if entry["topic"] == detected_topic:
                    matching_topic_entry = entry
                    break

            if matching_topic_entry:
                faq_questions = matching_topic_entry["question"]
                most_similar_index = calculate_similarity(user_input_english, faq_questions)
                faq_answer = matching_topic_entry["answer"]
                bot_response_english = faq_answer
            else:
                prompt = f"\nTopic: {user_input_english}\n\nChatbot:"
                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=prompt,
                    max_tokens=500
                )
                bot_response_english = response.choices[0].text.strip()

            # Translate the bot's response back to the detected language
            bot_response_translated = translate_to_detected_language(bot_response_english, detected_language)
        else:
            prompt = f"\nTopic: {user_input_english}\n\nChatbot:"
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=500
            )
            bot_response_english = response.choices[0].text.strip()

            # Translate the bot's response back to the detected language
            bot_response_translated = translate_to_detected_language(bot_response_english, detected_language)

    # : Capture the current date and time
    user_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Send feedback to Google Sheets
    #send_feedback_to_sheets(user_input, bot_response_translated, user_date)

    #return jsonify({'bot_response': bot_response_translated})
    return render_template('chat_3.html',appended_text=bot_response_translated)


if __name__ == '__main__':
    app.run(debug=True)    
