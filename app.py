import gradio as gr
import joblib
import os

model = joblib.load("spam_classifier_model.joblib")

def classify_email(text):
    if not text.strip():
        return "Please enter email text"

    prediction = model.predict([text])[0]
    prediction = str(prediction).lower().strip()

    if prediction == "spam":
        return "🚨 Spam Email Detected"
    else:
        return "✅ Safe Email (Ham)"

app = gr.Interface(
    fn=classify_email,
    inputs=gr.Textbox(lines=8, placeholder="Enter email text"),
    outputs="text",
    title="Email Spam Classifier"
)

app.launch()
