import pandas as pd
import re
import streamlit as st
import numpy as np

from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import resample

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(page_title="StressSense AI", page_icon="📉")

# -------------------------------
# Setup NLTK
# -------------------------------
@st.cache_resource
def setup_nltk():
    import nltk
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('stopwords')

setup_nltk()
lemmatizer = WordNetLemmatizer()

# -------------------------------
# Text Preprocessing
# -------------------------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.strip()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

# -------------------------------
# Validate Input
# -------------------------------
def is_valid_input(text):
    words = text.split()
    if len(words) < 2:
        return False
    processed = preprocess(text)
    if len(processed.split()) == 0:
        return False
    return True

# -------------------------------
# Gibberish Detector
# -------------------------------
@st.cache_resource
def load_common_words():
    from nltk.corpus import stopwords
    base = set(stopwords.words('english'))
    extra = {
        "work","feel","tired","stress","stressed","exam","test","job",
        "deadline","sleep","eat","angry","sad","happy","afraid","scared",
        "problem","issue","help","need","want","difficult","hard","easy",
        "study","school","college","office","manager","family","friend",
        "assignment","project","pressure","overwhelmed","anxiety","worry",
        "worried","nervous","calm","relax","enjoy","love","hate","like",
        "health","sick","pain","hurt","mind","body","think","thought",
        "today","tomorrow","yesterday","time","long","short","big","small",
        "good","bad","great","okay","fine","terrible","awful","amazing",
        "life","death","hope","dream","goal","fail","success","lose","win",
        "money","debt","bill","late","early","busy","free","alone","lonely",
        "frustrated","exhausted","burden","workload","confused","depressed",
        "panic","fear","tension","uncomfortable","unable","cannot","lot",
        "much","many","every","day","night","week","month","year","hour",
        "people","person","someone","everyone","nobody","thing","nothing"
    }
    return base.union(extra)

def is_meaningful_text(text):
    common_words = load_common_words()
    processed = preprocess(text)
    words = processed.split()
    if len(words) == 0:
        return False
    known_count = sum(1 for w in words if w in common_words)
    return (known_count / len(words)) >= 0.2

# -------------------------------
# Keyword Lists
# -------------------------------
high_stress_words = [
    "panic","terrified","hopeless","unbearable","desperate",
    "breakdown","cannot cope","falling apart","lose my mind",
    "cant breathe","out of control","completely lost","impossible to manage",
"cannot manage",
"cant manage",
"losing control",
"breaking down",
"out of control",
"falling apart",
"mentally exhausted"
]

medium_stress_words = [
    "stress","stressed","pressure","anxious","anxiety",
    "worried","worry","deadline","tired","exhausted",
    "burden","workload","confused","depressed","frustrated",
    "struggle","nervous","tense","uncomfortable","assignment",
    "a lot of work","so much to do","too much","piling up",
    "no time","running out of time","cant sleep","can't sleep",
    "overwhelmed","behind on","falling behind","under pressure"
]

low_stress_words = [
    "happy","joyful","relaxed","calm","peaceful","great",
    "wonderful","excited","love","enjoy","grateful","amazing",
    "fantastic","content","cheerful","delighted","thrilled","going smoothly",
"under control",
"stress free",
"manageable",
"peaceful day",
"doing well"
]

def keyword_stress_level(text):
    text_lower = text.lower()
    processed = preprocess(text)
    for phrase in high_stress_words:
        if phrase in text_lower:
            return "High"
    medium_hits = sum(1 for w in medium_stress_words if w in text_lower or w in processed)
    low_hits = sum(1 for w in low_stress_words if w in processed)
    if medium_hits >= 1:
        return "Medium"
    if low_hits >= 1:
        return "Low"
    return None

# -------------------------------
# Dataset Cleaning (runs inside cache)
# -------------------------------
def clean_and_map(df):
    # Remove duplicates
    df = df.drop_duplicates(subset='text')

    # Remove very short texts
    df = df[df['text'].apply(lambda x: len(str(x).split()) >= 4)].copy()

    # Remove negation-confusion rows (e.g. "not feel afraid" labeled fear)
    negation_patterns = [
        r'not feel (?:afraid|fearful|scared|terrified|frightened)',
        r'do not feel (?:overwhelmed|unsafe|insecure)',
        r'no longer feel (?:afraid|scared|fearful)'
    ]
    mask = pd.Series([False]*len(df), index=df.index)
    for pat in negation_patterns:
        mask |= df['text'].str.contains(pat, case=False, regex=True)
    df = df[~mask]

    # IMPROVED emotion → stress mapping
    # Key change: surprise → Medium (not Low)
    # Surprise texts are confused/shocked, closer to medium stress than calm
    stress_map = {
        'joy':      'Low',
        'love':     'Low',
        'surprise': 'Medium',
        'sadness':  'Medium',
        'fear':     'High',
        'anger':    'High'
    }
    df['stress'] = df['label'].map(stress_map)
    df = df.dropna(subset=['stress'])

    # Keep all classes without aggressive undersampling
    low_df  = df[df['stress'] == 'Low']
    med_df  = df[df['stress'] == 'Medium']
    high_df = df[df['stress'] == 'High']

    df = pd.concat([low_df, med_df, high_df]).sample(frac=1, random_state=42)

    return df

# -------------------------------
# Prepare Model + Vectorizer
# -------------------------------
@st.cache_resource
def prepare_pipeline():
    df = pd.read_csv("train_cleaned.txt", sep=';', names=['text', 'label'])
    df["text"] = df["text"].apply(preprocess)
    df = clean_and_map(df)

    le = LabelEncoder()
    y = le.fit_transform(df["stress"])

    # Bigrams capture phrases like "a lot of work", "too much pressure"
    vectorizer = TfidfVectorizer(
    max_features=2000,
    ngram_range=(1,1),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)
    X = vectorizer.fit_transform(df["text"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # class_weight='balanced' further corrects any remaining imbalance
    base_model = LinearSVC(max_iter=2000)
    model = CalibratedClassifierCV(base_model, cv=3)
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc  = model.score(X_test, y_test)

    return model, vectorizer, le, train_acc, test_acc

best_model, vectorizer, le, train_acc, test_acc = prepare_pipeline()

# -------------------------------
# Hybrid Prediction
# -------------------------------
def predict_stress(user_text):
    # Layer 1: Gibberish / nonsense check
    if not is_meaningful_text(user_text):
        return "Unknown", None

    processed = preprocess(user_text)
    vector = vectorizer.transform([processed])

    # Layer 2: Model with real probability confidence
    proba = best_model.predict_proba(vector)[0]
    max_confidence = max(proba)
    pred_index = np.argmax(proba)
    model_result = le.inverse_transform([pred_index])[0]

    if max_confidence < 0.35:
        return "Unknown", max_confidence

    # Layer 3: Keyword override for known disagreements
    keyword_result = keyword_stress_level(user_text)
    if keyword_result == "Low":
        return "Low", max_confidence
    if keyword_result is not None:
        if keyword_result == "Medium" and model_result == "Low":
            return "Medium", max_confidence
        elif keyword_result == "Medium" and model_result == "High":
            return "Medium", max_confidence
        elif keyword_result == "High" and model_result == "Low":
            return "High", max_confidence
        elif keyword_result == "Low" and model_result == "High":
            return "Low", max_confidence

    return model_result, max_confidence

# -------------------------------
# UI
# -------------------------------
st.title("📉 StressSense AI")
st.caption("Analyze stress levels from text using Machine Learning & NLP")

col1, col2 = st.columns(2)
with col1:
    st.metric("🎯 Training Accuracy", f"{train_acc:.2%}")
with col2:
    st.metric("🧪 Testing Accuracy", f"{test_acc:.2%}")

st.divider()

user_text = st.text_area(
    "Enter your text",
    placeholder="e.g. I have three deadlines tomorrow and haven't slept properly..."
)

if st.button("Predict", type="primary"):
    if user_text.strip() == "":
        st.warning("Please enter some text.")
    elif not is_valid_input(user_text):
        st.info("⚪ Unknown — input too short or invalid.")
    else:
        result, confidence = predict_stress(user_text)

        if result == "Unknown":
            st.info("⚪ Unknown / No stress context detected")
            st.caption("The text doesn't contain recognizable stress-related context.")
        elif result == "High":
            st.error("🔴 High Stress Detected")
        elif result == "Medium":
            st.warning("🟡 Medium Stress Detected")
        else:
            st.success("🟢 Low Stress Detected")

        if confidence is not None:
            st.caption(f"📊 Model Confidence: {confidence:.2%}")

        with st.expander("🔍 Debug Info"):
            st.write(f"**Processed Text:** {preprocess(user_text)}")
            st.write(f"**Keyword Signal:** {keyword_stress_level(user_text)}")
            st.write(f"**Meaningful Text:** {is_meaningful_text(user_text)}")
