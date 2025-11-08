import re
import joblib
import nltk
from nltk.corpus import stopwords
from scipy.sparse import hstack, csr_matrix
from nltk.stem import WordNetLemmatizer
import numpy as np
from collections import Counter

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()
stop = set(stopwords.words('english'))

# Load all artifacts
print("Loading model and resources...")
model = joblib.load("duplicate_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
scaler = joblib.load("feature_scaler.pkl")

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

def get_similarity_details(prob, features):
    """Get detailed analysis of why questions are similar/different"""
    analysis = []
    
    # Core similarity from model
    if prob > 0.8:
        analysis.append("Very high similarity in content and structure")
    elif prob > 0.6:
        analysis.append("High similarity in content")
    elif prob > 0.4:
        analysis.append("Moderate similarity")
    else:
        analysis.append("Low similarity")
        
    # Word overlap analysis
    if features[0] > 0.7:
        analysis.append("Many common words")
    elif features[0] < 0.3:
        analysis.append("Few common words")
        
    # Question type analysis
    if features[4] == 1.0:
        analysis.append("Same question type")
    else:
        analysis.append("Different question types")
        
    # Word order analysis
    if features[2] > 0.5:
        analysis.append("Similar word order")
    else:
        analysis.append("Different word order")
        
    return analysis

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

while True:
    q1 = input("Enter Question 1: ").strip()
    q2 = input("Enter Question 2: ").strip()

    if not q1 or not q2:
        print("❌ Empty input. Exiting.")
        break

    # Clean and preprocess text
    q1_clean = clean_text(q1)
    q2_clean = clean_text(q2)

    # Get TF-IDF vectors
    q1_vec = tfidf.transform([q1_clean])
    q2_vec = tfidf.transform([q2_clean])

    # Get additional features
    add_features = get_additional_features(q1_clean, q2_clean)
    add_features_scaled = scaler.transform(add_features.reshape(1, -1))

    # Combine all features
    X = hstack([q1_vec, q2_vec, csr_matrix(add_features_scaled)])

    # Get prediction probability
    prob = model.predict_proba(X)[0][1]
    pred = 1 if prob > 0.65 else 0  # Much stricter threshold

    # Get detailed analysis
    analysis = get_similarity_details(prob, add_features)

    # Show detailed analysis
    print("\n📊 Similarity Analysis")
    print("─" * 50)
    print(f"Overall Similarity:     {prob:.2%}")
    print(f"Word Overlap:           {add_features[0]:.2%}")
    print(f"Word Order Match:       {add_features[2]:.2%}")
    print(f"Question Type Match:    {'Yes' if add_features[4] == 1.0 else 'No'}")
    print("\n💡 Analysis:")
    for point in analysis:
        print(f"  • {point}")
    
    if pred == 1:
        print("\n✅ VERDICT: DUPLICATE QUESTIONS")
        print("   These questions are asking the same thing")
    else:
        print("\n❌ VERDICT: DIFFERENT QUESTIONS")
        print("   These questions are asking different things")
