# FAKE-NEWS-DETECTOR

# Fake News Detector

## Overview
This project builds a machine learning model to classify news articles as **Real** or **Fake** using NLP techniques with TF-IDF and Logistic Regression.

---

## Features
- Dataset cleaning (remove duplicates and empty entries)
- Dataset balancing (equal samples of real and fake news)
- Model training and saving with scikit-learn
- Simple Flask web app for user input and prediction

---

## Folder Structure
fake-news-detector/
├── dataset/
│ ├── train.csv # Original dataset
│ ├── train_cleaned.csv # Cleaned dataset (no duplicates or empty rows)
│ └── train_balanced.csv # Balanced dataset (equal real & fake news)
├── model/
│ ├── fake_news_model.pkl
│ └── vectorizer.pkl
├── templates/
│ └── index.html
├── app.py # Flask app to serve prediction UI
├── train_model.py # Script to clean, balance & train the model


---

## Setup Instructions

### 1. Prepare the Dataset
- Download the dataset files (`Fake.csv` and `True.csv`) and combine them into `dataset/train.csv` with columns: `title`, `text`, `label` (0 = fake, 1 = real).

### 2. Clean and Balance Dataset & Train Model
Run the training script which will:
- Remove duplicates and empty rows
- Balance the dataset
- Train and save the model and vectorizer

```bash
python train_model.py

Then run the  flask app by "python app.py" in terminal
