import pandas as pd
import nltk
import string

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

# -----------------------------
# STEP 1: Setup (FAST)
# -----------------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))   # ✅ load once

# -----------------------------
# STEP 2: Load Dataset
# -----------------------------
df = pd.read_csv("customer_support_tickets.csv")

# OPTIONAL: limit size if system slow
# df = df.sample(n=2000, random_state=42)

# -----------------------------
# STEP 3: Fast Text Cleaning
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))  # faster
    words = text.split()
    return " ".join([w for w in words if w not in stop_words])

df['clean_text'] = df['Ticket Description'].apply(clean_text)

# -----------------------------
# STEP 4: Feature Extraction (IMPROVED)
# -----------------------------
vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1,2))
X = vectorizer.fit_transform(df['clean_text'])

# -----------------------------
# STEP 5: Labels
# -----------------------------
le = LabelEncoder()
y = le.fit_transform(df['Ticket Type'])

# -----------------------------
# STEP 6: Train Model (FAST + BETTER)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# -----------------------------
# STEP 7: Evaluation
# -----------------------------
y_pred = model.predict(X_test)

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# STEP 8: Priority Logic
# -----------------------------
def assign_priority(text):
    text = text.lower()
    if any(w in text for w in ["urgent", "failed", "error", "not working"]):
        return "High"
    elif any(w in text for w in ["slow", "delay"]):
        return "Medium"
    return "Low"

# -----------------------------
# STEP 9: Prediction Function
# -----------------------------
def predict_ticket(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    
    category = le.inverse_transform(model.predict(vec))[0]
    priority = assign_priority(text)
    
    return category, priority

# -----------------------------
# STEP 10: Test
# -----------------------------
test = "My payment failed and app is not working"

cat, pri = predict_ticket(test)

print("\nTest Ticket:", test)
print("Predicted Category:", cat)
print("Predicted Priority:", pri)