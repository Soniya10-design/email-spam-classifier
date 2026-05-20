import gradio as gr
import joblib
import os

# =========================
# LOAD TRAINED MODEL
# =========================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "spam_classifier_model.joblib"
)

model = joblib.load(MODEL_PATH)

# =========================
# EMAIL CLASSIFICATION FUNCTION
# =========================

def classify_email(text):

    try:

        # Empty input check
        if not text.strip():
            return "⚠️ Please enter email text"

        # Predict
        prediction = model.predict([text])[0]

        # Convert prediction safely
        prediction = str(prediction).lower().strip()

        # Result
        if prediction == "spam":
            return "🚨 Spam Email Detected"

        else:
            return "✅ Safe Email (Ham)"

    except Exception as e:
        return f"Error: {str(e)}"

# =========================
# GRADIO INTERFACE
# =========================

interface = gr.Interface(
    fn=classify_email,

    inputs=gr.Textbox(
        lines=10,
        placeholder="Paste your email text here..."
    ),

    outputs=gr.Textbox(
        label="Prediction Result"
    ),

    title="📧 Email Spam Classifier",

    description="""
This AI model detects whether an email is Spam or Safe (Ham).
Paste any email text and test the prediction.
""",

    theme="soft"
)

# =========================
# LAUNCH APP
# =========================

if __name__ == "__main__":
    interface.launch()
