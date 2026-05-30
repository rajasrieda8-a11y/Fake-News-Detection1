import kagglehub
import pandas as pd
import os
import zipfile
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download Dataset
path = kagglehub.dataset_download("bhavikjikadara/fake-news-detection")
print("Dataset downloaded at:", path)

# Extract ZIP files
for file in os.listdir(path):
    if file.endswith(".zip"):
        zip_path = os.path.join(path, file)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(path)

        print("Extracted:", file)

# Find CSV files
fake_path = None
true_path = None

for root, dirs, files in os.walk(path):
    for file in files:

        if file.lower().endswith(".csv"):

            if "fake" in file.lower():
                fake_path = os.path.join(root, file)

            elif "true" in file.lower():
                true_path = os.path.join(root, file)

print("Fake file:", fake_path)
print("True file:", true_path)

# Load Dataset
fake_df = pd.read_csv(fake_path)
true_df = pd.read_csv(true_path)

# Labels
fake_df["label"] = 0
true_df["label"] = 1

# Combine
df = pd.concat([fake_df, true_df], axis=0)

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Content Column
df["content"] = (
    df["title"].fillna("") + " " +
    df["text"].fillna("")
)

# Remove Empty Rows
df = df[df["content"].str.strip() != ""]

# Features & Labels
X = df["content"]
y = df["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = LogisticRegression(max_iter=1000)

model.fit(X_train_vec, y_train)

# Prediction
y_pred = model.predict(X_test_vec)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save Model
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\n✅ Model Saved Successfully")
