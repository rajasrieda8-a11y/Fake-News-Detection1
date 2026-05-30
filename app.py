# ==========================================
# INSTALL GRADIO
# ==========================================
!pip install gradio joblib

# ==========================================
# IMPORT LIBRARIES
# ==========================================
import gradio as gr
import joblib

# ==========================================
# LOAD MODEL
# ==========================================
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ==========================================
# PREDICTION FUNCTION
# ==========================================
def predict_news(news_text):

    if news_text.strip() == "":
        return "⚠️ Please enter news content"

    text_vec = vectorizer.transform([news_text])

    prediction = model.predict(text_vec)[0]

    if prediction == 1:
        return "🟢 Real News"
    else:
        return "🔴 Fake News"

# ==========================================
# CREATE INTERFACE
# ==========================================
app = gr.Interface(
    fn=predict_news,

    inputs=gr.Textbox(
        lines=8,
        placeholder="Paste news article here..."
    ),

    outputs=gr.Textbox(
        label="Prediction"
    ),

    title="🧠 Fake News Detection System",

    description="""
    Enter any news article and check whether
    it is Real News or Fake News.
    """,

    examples=[
        ["Government launches new education scheme."],
        ["Aliens have taken control of India."]
    ]
)

# ==========================================
# LAUNCH APP
# ==========================================
app.launch(share=True)