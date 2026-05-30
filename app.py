import gradio as gr
import joblib


model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def predict_news(news_text):

    if news_text.strip() == "":
        return "⚠️ Please enter news text."

    text_vector = vectorizer.transform([news_text])

    prediction = model.predict(text_vector)[0]

    if prediction == 1:
        return "🟢 Real News"
    else:
        return "🔴 Fake News"


app = gr.Interface(
    fn=predict_news,

    inputs=gr.Textbox(
        lines=8,
        placeholder="Paste news article here..."
    ),

    outputs=gr.Textbox(
        label="Prediction"
    ),

    title="📰 Fake News Detection System",

    description="Check whether a news article is Real or Fake using Machine Learning.",

    examples=[
        ["The Reserve Bank of India kept interest rates unchanged in its latest monetary policy meeting."],
        ["Aliens landed in India and took control of all government offices."]
    ]
)

app.launch(share=True)
