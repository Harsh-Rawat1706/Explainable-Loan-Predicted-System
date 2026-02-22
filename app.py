from flask import Flask, render_template, request
from utils import process_input, predict_and_explain

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    form_data = request.form
    df = process_input(form_data)

    pred, prob, suggestions = predict_and_explain(df)

    return render_template(
        "result.html",
        prediction=pred,
        probability=round(prob * 100, 2),
        suggestions=suggestions
    )

if __name__ == "__main__":
    app.run(debug=True)