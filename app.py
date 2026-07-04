from flask import Flask, render_template, request
import pickle
import time

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    email = request.form["email"]

    if email.strip() == "":
        return render_template("index.html",
                               prediction_text="No Input",
                               message="Please enter email content",
                               confidence=0,
                               keywords=[])

    # Simulate AI thinking
    time.sleep(1.2)

    data = vectorizer.transform([email])

    prediction = model.predict(data)[0]
    prob = model.predict_proba(data)[0]
    confidence = round(max(prob) * 100, 2)

    # Extract keywords
    feature_names = vectorizer.get_feature_names_out()
    dense = data.toarray()[0]

    word_scores = list(zip(feature_names, dense))
    sorted_words = sorted(word_scores, key=lambda x: x[1], reverse=True)

    keywords = [word for word, score in sorted_words[:5] if score > 0]

    # Output
    if prediction == 1:
        prediction_text = "Spam Email Detected"
        message = "This email contains patterns commonly associated with spam or phishing."
    else:
        prediction_text = "Legitimate Email"
        message = "This email appears safe and does not contain spam characteristics."

    return render_template("index.html",
                           prediction_text=prediction_text,
                           message=message,
                           confidence=confidence,
                           keywords=keywords)

if __name__ == "__main__":
    app.run(debug=True)