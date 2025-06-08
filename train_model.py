import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os

# Load dataset
df = pd.read_csv("dataset/train.csv")

# Combine title + text if needed
df['text'] = df['title'].fillna('') + " " + df['text'].fillna('')

# Features and labels
X = df['text']
y = df['label']

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model and vectorizer
os.makedirs("model", exist_ok=True)
with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open("model/fake_news_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model and vectorizer saved!")
