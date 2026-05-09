# 📉 StressSense AI
> Text-based stress detection using NLP and Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🔗 Live App

👉 https://text-based-stress-level-detection.streamlit.app/

---

## 📌 What It Does

StressSense AI takes any text input and predicts its stress level in real time:

| Output | Meaning |
|--------|---------|
| 🟢 **Low Stress** | Calm, positive, or neutral tone |
| 🟡 **Medium Stress** | Worry, sadness, or mild pressure |
| 🔴 **High Stress** | Fear, panic, or anger |
| ⚪ **Unknown** | Gibberish or no stress-related context |

---

## 🧠 How It Works

```
User Input
    │
    ▼
Text Preprocessing          ← lowercase, clean, lemmatize
    │
    ▼
Gibberish Filter            ← rejects nonsense like "banana apple table"
    │
    ▼
TF-IDF Vectorization        ← top 2000 features, unigrams
    │
    ▼
Linear SVM (Calibrated)     ← trained on 14,000+ cleaned samples
    │
    ▼
Confidence Check
    ├── < 20%  → Unknown
    ├── 20–50% → Keyword layer can override model result
    └── > 50%  → Model result is trusted directly
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.8+ |
| ML Model | Linear SVM (`LinearSVC` + `CalibratedClassifierCV`) |
| Vectorizer | TF-IDF (`max_features=2000`, `sublinear_tf=True`) |
| NLP | NLTK (lemmatization, stopwords) |
| Data | Pandas |
| UI | Streamlit |

---

## 📂 Dataset

- **Size:** 16,000+ samples → cleaned to ~14,200
- **Source:** Emotion classification dataset (6 classes)
- **Emotion → Stress mapping:**

```
joy, love       →  Low Stress
surprise        →  Medium Stress   (shocked/confused ≠ calm)
sadness         →  Medium Stress
fear, anger     →  High Stress
```

### Cleaning applied:
- Removed duplicate entries
- Removed texts under 4 words
- Removed negation-confused rows (e.g. *"I do not feel afraid"* labeled as `fear`)
- Rebalanced class distribution (Low was 45% of raw data)

---

## 🧪 Example Predictions

| Input | Predicted |
|-------|-----------|
| *"I have three deadlines tomorrow and I haven't slept"* | 🔴 High Stress |
| *"I'm a bit worried about my upcoming interview"* | 🟡 Medium Stress |
| *"Just finished my project, feeling really proud"* | 🟢 Low Stress |
| *"banana apple table car bike phone"* | ⚪ Unknown |

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Training Accuracy | ~96% |
| Testing Accuracy | Displayed live in app |

> `max_features=2000` was used in TF-IDF to reduce overfitting (down from 99% train accuracy).

---

## ⚠️ Limitations

- Trained on emotion-labeled data, not direct stress labels — predictions are approximations
- May misclassify sarcasm or highly contextual sentences
- English only
- Intended for educational and demonstration purposes

---

## 👨‍💻 Author

**Arulmozhi K**

[![GitHub](https://img.shields.io/badge/GitHub-@yourusername-black?logo=github)](https://github.com/yourusername)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/yourprofile)
