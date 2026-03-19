from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

# Load model
model, vectorizer = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data["text"]

    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]

    result = "Positive 😊" if prediction == 1 else "Negative 😡"

    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True)