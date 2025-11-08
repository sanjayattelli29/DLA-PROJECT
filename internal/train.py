import pandas as pd
import numpy as np
import re
import joblib
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from scipy.sparse import hstack, csr_matrix
from nltk.stem import WordNetLemmatizer
from collections import Counter

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()
stop = set(stopwords.words('english'))

# ✅ 1. Load dataset
print("Loading dataset...")
data = pd.read_csv("quora_local.csv")

# ✅ 2. Improved text preprocessing
def clean_text(text):
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove special characters but keep important punctuation
    text = re.sub(r'[^a-z\s\']', ' ', text)
    
    # Remove extra spaces and split
    words = text.split()
    
    # Remove stopwords and lemmatize
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop]
    
    return " ".join(words)

print("Preprocessing text...")
data["question1"] = data["question1"].apply(clean_text)
data["question2"] = data["question2"].apply(clean_text)

# ✅ 3. Enhanced feature extraction
print("Extracting features...")

# Create TF-IDF vectors with optimized parameters
tfidf = TfidfVectorizer(
    max_features=6000,     # More features
    ngram_range=(1, 2),    # Unigrams and bigrams only (trigrams add noise)
    min_df=3,             # Stricter minimum frequency
    max_df=0.9,          # Stricter maximum frequency
    strip_accents='unicode',
    sublinear_tf=True    # Apply sublinear scaling
)

# Transform questions to TF-IDF vectors
print("Creating TF-IDF vectors...")
q1_vec = tfidf.fit_transform(data["question1"])
q2_vec = tfidf.transform(data["question2"])

# ✅ 4. Create additional features
def get_additional_features(q1, q2):
    # Handle empty strings
    if not q1.strip() or not q2.strip():
        return np.zeros(6)  # Return zero features for empty strings
    
    # Tokenize questions
    q1_tokens = nltk.word_tokenize(q1.lower())
    q2_tokens = nltk.word_tokenize(q2.lower())
    
    # Create word sets and counts
    q1_words = set(q1_tokens)
    q2_words = set(q2_tokens)
    q1_counts = Counter(q1_tokens)
    q2_counts = Counter(q2_tokens)
    
    # 1. Word overlap ratio
    shared_words = q1_words.intersection(q2_words)
    total_words = q1_words.union(q2_words)
    word_overlap = len(shared_words) / len(total_words) if total_words else 0
    
    # 2. Length difference ratio
    len_diff = abs(len(q1_tokens) - len(q2_tokens)) / (len(q1_tokens) + len(q2_tokens)) if q1_tokens and q2_tokens else 1.0
    
    # 3. Word order similarity
    min_len = min(len(q1_tokens), len(q2_tokens))
    word_order = sum(1 for i in range(min_len) if q1_tokens[i] == q2_tokens[i])
    word_order_sim = word_order / min_len if min_len > 0 else 0
    
    # 4. Common word frequency similarity
    freq_sim = 0
    if shared_words:
        freq_diffs = [abs(q1_counts[w] - q2_counts[w]) for w in shared_words]
        freq_sim = 1 - (sum(freq_diffs) / len(shared_words)) / max(len(q1_tokens), len(q2_tokens))
    
    # 5. Question type similarity (what, how, why, etc.)
    q_words = {'what', 'how', 'why', 'when', 'where', 'which', 'who'}
    q1_type = next((w for w in q1_tokens if w.lower() in q_words), '')
    q2_type = next((w for w in q2_tokens if w.lower() in q_words), '')
    type_sim = 1.0 if q1_type and q1_type == q2_type else 0.0
    
    # 6. Content word ratio
    content_words1 = [w for w in q1_tokens if w.lower() not in stop]
    content_words2 = [w for w in q2_tokens if w.lower() not in stop]
    content_ratio = abs(len(content_words1) - len(content_words2)) / (len(content_words1) + len(content_words2)) if content_words1 and content_words2 else 1.0
    
    return np.array([
        word_overlap,      # Basic word overlap
        len_diff,         # Length difference
        word_order_sim,   # Word order similarity
        freq_sim,         # Word frequency similarity
        type_sim,         # Question type similarity
        1 - content_ratio # Content word ratio (inverted)
    ])

print("Computing additional features...")
additional_features = np.array([
    get_additional_features(q1, q2) 
    for q1, q2 in zip(data["question1"], data["question2"])
])

# Scale the additional features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
additional_features_scaled = scaler.fit_transform(additional_features)

# Combine all features
print("Combining features...")
X = hstack([
    q1_vec,
    q2_vec,
    csr_matrix(additional_features_scaled)
])
y = data["is_duplicate"]

# ✅ 5. Split dataset with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ✅ 6. Train model with optimized parameters
print("Training model...")
model = LogisticRegression(
    C=1.0,               # More regularization
    max_iter=2000,       # Even more iterations
    class_weight='balanced',  # Handle class imbalance
    solver='lbfgs',      # Better solver for our case
    random_state=42
)
model.fit(X_train, y_train)

# ✅ 7. Evaluate
print("\nEvaluating model...")
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

print("\nModel Performance:")
print("─" * 40)
print(f"Train Accuracy: {accuracy_score(y_train, train_pred):.4f}")
print(f"Test Accuracy:  {accuracy_score(y_test, test_pred):.4f}")
print(f"Test F1 Score:  {f1_score(y_test, test_pred):.4f}")

# Print confusion matrix
cm = confusion_matrix(y_test, test_pred)
print("\nConfusion Matrix:")
print("─" * 40)
print(f"True Negatives:  {cm[0][0]}")
print(f"False Positives: {cm[0][1]}")
print(f"False Negatives: {cm[1][0]}")
print(f"True Positives:  {cm[1][1]}")

# Save scaler along with model
print("\nSaving model artifacts...")

# ✅ 8. Save all artifacts
joblib.dump(model, "duplicate_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
joblib.dump(scaler, "feature_scaler.pkl")

print("✅ Training completed successfully!")
print("\nReady to detect duplicate questions with improved accuracy!")
