# ==========================================
# STEP 1: Install Libraries
# ==========================================
!pip install kagglehub scikit-learn pandas joblib

# ==========================================
# STEP 2: Import Libraries
# ==========================================
import kagglehub
import pandas as pd
import os
import zipfile
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# STEP 3: Download Dataset
# ==========================================
path = kagglehub.dataset_download(
    "bhavikjikadara/fake-news-detection"
)

print("Dataset downloaded at:", path)

# ==========================================
# STEP 4: Extract ZIP Files
# ==========================================
for file in os.listdir(path):
    if file.endswith(".zip"):
        zip_path = os.path.join(path, file)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(path)

        print("Extracted:", file)

# ==========================================
# STEP 5: Locate CSV Files
# ==========================================
fake_path = None
true_path = None

for root, dirs, files in os.walk(path):
    for file in files:
        if file.lower().endswith(".csv"):

            if "fake" in file.lower():
                fake_path = os.path.join(root, file)

            elif "true" in file.lower():
                true_path = os.path.join(root, file)

print("Fake file path:", fake_path)
print("True file path:", true_path)

if fake_path is None or true_path is None:
    raise FileNotFoundError("Dataset CSV files not found.")

# ==========================================
# STEP 6: Load Dataset
# ==========================================
fake_df = pd.read_csv(fake_path)
true_df = pd.read_csv(true_path)

print("Fake Shape:", fake_df.shape)
print("True Shape:", true_df.shape)

# ==========================================
# STEP 7: Add Labels
# ==========================================
fake_df["label"] = 0
true_df["label"] = 1

# ==========================================
# STEP 8: Combine Dataset
# ==========================================
df = pd.concat([fake_df, true_df], axis=0)

df = df.sample(frac=1, random_state=42)
df.reset_index(drop=True, inplace=True)

print("Combined Shape:", df.shape)

# ==========================================
# STEP 9: Create Content Column
# ==========================================
if "title" in df.columns and "text" in df.columns:
    df["content"] = (
        df["title"].fillna("") + " " +
        df["text"].fillna("")
    )
else:
    df["content"] = df["text"].fillna("")

df = df[df["content"].str.strip() != ""]

# ==========================================
# STEP 10: Split Dataset
# ==========================================
X = df["content"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# STEP 11: TF-IDF Vectorization
# ==========================================
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ==========================================
# STEP 12: Train Model
# ==========================================
model = LogisticRegression(max_iter=1000)

model.fit(X_train_vec, y_train)

# ==========================================
# STEP 13: Prediction
# ==========================================
y_pred = model.predict(X_test_vec)

# ==========================================
# STEP 14: Evaluation
# ==========================================
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ==========================================
# STEP 15: Save Model
# ==========================================
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\nModel Saved Successfully!")

# ==========================================
# STEP 16: Test Prediction
# ==========================================
def predict_news(text):
    text_vec = vectorizer.transform([text])

    prediction = model.predict(text_vec)[0]

    if prediction == 1:
        return "🟢 Real News"
    else:
        return "🔴 Fake News"

sample = "Government announces new policy for economic growth"

print("\nPrediction:", predict_news(sample))