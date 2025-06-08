from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

# Load model & vectorizer
with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("model/fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        news_text = request.form.get("news_text")
        if news_text:
            vec = vectorizer.transform([news_text])
            pred = model.predict(vec)[0]
            prediction = "Real News" if pred == 1 else "Fake News"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
