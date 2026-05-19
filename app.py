import gradio as gr
import joblib

# Load model
model = joblib.load("spam_classifier_model.joblib")

# Prediction function
def classify_email(email_text):

    if not email_text.strip():
        return "Please enter email text."

    prediction = model.predict([email_text])[0]

    if prediction.lower() == "spam":
        return "🚨 Spam Email"

    return "✅ Safe Email"

# Gradio interface
interface = gr.Interface(
    fn=classify_email,
    inputs=gr.Textbox(
        lines=8,
        placeholder="Paste email text here..."
    ),
    outputs="text",
    title="Email Spam Classifier",
    description="AI model to classify spam and safe emails."
)

# Launch app
interface.launch()
