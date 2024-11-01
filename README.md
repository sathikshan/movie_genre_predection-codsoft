# movie_genre_predection-codsoft
# Install required libraries
!pip install scikit-learn flask ngrok

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
from flask import Flask, request, jsonify
import os

# Load dataset
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.readlines()

    movie_names = []
    summaries = []
    genres = []

    # Process each line to extract movie name, summary, and genre
    for line in data:
        # Assuming each line is formatted as: "Movie Name|Summary|Genre"
        parts = line.strip().split('|')
        if len(parts) == 3:  # Ensure each line has all three components
            movie_names.append(parts[0])
            summaries.append(parts[1])
            genres.append(parts[2])

    return pd.DataFrame({'movie_name': movie_names, 'summary': summaries, 'genre': genres})

# Load the dataset
file_path = '/content/test_data.txt'  # Change this to the path of your uploaded file
data = load_data(file_path)

# Print basic info
print(f"Data shape: {data.shape}")
print(data.head())

# Split the dataset
X = data['summary']  # Features
y = data['genre']    # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a TF-IDF Vectorizer and a Multinomial Naive Bayes model
pipeline_nb = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])

# Train the model
pipeline_nb.fit(X_train, y_train)

# Evaluate the model
y_pred_nb = pipeline_nb.predict(X_test)
print("Naive Bayes Classification Report:")
print(classification_report(y_test, y_pred_nb))

# Create a Logistic Regression model
pipeline_lr = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Train the model
pipeline_lr.fit(X_train, y_train)

# Evaluate the model
y_pred_lr = pipeline_lr.predict(X_test)
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr))

# Create a Support Vector Machine model
pipeline_svc = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', SVC())
])

# Train the model
pipeline_svc.fit(X_train, y_train)

# Evaluate the model
y_pred_svc = pipeline_svc.predict(X_test)
print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svc))

# Save the best model (choose the best one based on evaluation)
best_model = pipeline_nb  # Assuming Naive Bayes performed best
joblib.dump(best_model, 'movie_genre_model.pkl')
print("Best model saved as 'movie_genre_model.pkl'")

# Set up Flask API for real-time predictions
app = Flask(__name__)

# Load the pre-trained model
best_model = joblib.load('movie_genre_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    summary = data.get('summary', None)
    
    if not summary:
        return jsonify({'error': 'No summary provided'}), 400
    
    # Make prediction
    prediction = best_model.predict([summary])
    
    return jsonify({'predicted_genre': prediction[0]})

# If you want to run Flask in Google Colab, you'd need to use ngrok.
# This is an example to get a public URL to the Flask app:
from flask_ngrok import run_with_ngrok

# Run the Flask app with ngrok
run_with_ngrok(app)

if __name__ == '__main__':
    app.run()
