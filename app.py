import gradio as gr
import joblib
import os

# =========================
# SAFE MODEL LOADING
# =========================

MODEL_PATH = os.path.join(os.path.dirname(__file__), "spam_classifier_model.joblib")

model = joblib.load(MODEL_PATH)

# =========================
# PREDICTION FUNCTION
# =========================

def classify_email(text):
    try:
        if not text or text.strip() == "":
            return "⚠️ Please enter some email text"

        prediction = model.predict([text])[0]

        # Convert everything safely to string first
        prediction = str(prediction).lower().strip()

        if prediction in ["spam", "1", "1.0"]:
            return "🚨 Spam Email Detected"
        else:
            return "✅ Safe Email (Ham)"

    except Exception as e:
        return f"Error: {str(e)}"

        # Normalize output just in case
        prediction = prediction.lower().strip()

        if prediction == "spam":
            return "🚨 Spam Email Detected"
        else:
            return "✅ Safe Email (Ham)"

    except Exception as e:
        return f"Error: {str(e)}"

# =========================
# GRADIO UI
# =========================

interface = gr.Interface(
    fn=classify_email,
    inputs=gr.Textbox(
        lines=8,
        placeholder="Paste email text here..."
    ),
    outputs=gr.Textbox(label="Result"),
    title="Email Spam Classifier",
    description="AI model that detects whether an email is Spam or Safe (Ham).",
    theme="soft"
)

# =========================
# RUN APP
# =========================

if __name__ == "__main__":
    interface.launch()
