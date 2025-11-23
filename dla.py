# Import Libraries
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Download stopwords if not present
nltk.download('stopwords')
stop = set(stopwords.words('english'))

# Load Dataset
data = pd.read_csv("quora_question_pairs.csv")

# Keep only required columns
data = data[['question1', 'question2', 'is_duplicate']].dropna()

# Text Preprocessing Function
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = [w for w in text.split() if w not in stop]
    return " ".join(words)

# Apply preprocessing
data['question1'] = data['question1'].apply(preprocess)
data['question2'] = data['question2'].apply(preprocess)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
tfidf.fit(pd.concat([data['question1'], data['question2']]))

q1_features = tfidf.transform(data['question1'])
q2_features = tfidf.transform(data['question2'])

# Combine both question features
X = np.hstack((q1_features.toarray(), q2_features.toarray()))
y = data['is_duplicate'].values

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model (Logistic Regression)
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("TF-IDF Feature Matrix Shape:", X.shape)
print("Accuracy:", round(acc, 4))
print("F1-Score:", round(f1, 4))

# Sample Predictions
sample = data.sample(5)
for i in range(len(sample)):
    q1 = sample.iloc[i]['question1']
    q2 = sample.iloc[i]['question2']
    pred = model.predict(tfidf.transform([q1]).toarray().reshape(1, -1))
    print(f"Q1: {q1}")
    print(f"Q2: {q2}")
    print("Predicted Duplicate:", "Yes" if pred[0] == 1 else "No")
    print("-" * 50)