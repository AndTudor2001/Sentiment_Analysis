import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from scipy.stats import entropy
import streamlit as st
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import spacy
from transformers import BertTokenizer, BertModel
import torch

# Load data
file_path = "text.csv"  # Replace with your dataset's path
data = pd.read_csv(file_path)

# Drop 'id' column and ensure unique values
data = data.drop(columns=['id']).drop_duplicates()

# Rename columns for convenience
data.columns = ['text', 'sentiment']

data = data.sample(frac=0.1, random_state=42)
# Clean text
def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

data['text'] = data['text'].fillna('').astype(str)
data['text'] = data['text'].apply(clean_text)

# Tokenization and lemmatization
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)
# Function to extract BERT embeddings
def get_bert_embeddings(texts, tokenizer, model):
    embeddings = []
    for text in texts:
        # Tokenize and pad the text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            # Use the CLS token's embedding for classification
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            embeddings.append(cls_embedding.squeeze(0).numpy())
    return np.array(embeddings)

data['text'] = data['text'].apply(preprocess_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['sentiment'], test_size=0.2, random_state=42)

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
# Replace TF-IDF vectorization with BERT embeddings
X_train_bert = get_bert_embeddings(X_train, tokenizer, bert_model)
X_test_bert = get_bert_embeddings(X_test, tokenizer, bert_model)

models = {
    'Logistic Regression': LogisticRegression(max_iter=200),
    'Random Forest': RandomForestClassifier(n_estimators=100, n_jobs=-1),

}

# Function for cross-validation
def cross_validate_model(model, X_train_tfidf, y_train, cv=5, scoring='accuracy'):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train_tfidf, y_train, cv=skf, scoring=scoring, n_jobs=-1)
    return scores.mean(), scores.std()

# Function for evaluation
def evaluate_model_optimized(model, X_train_tfidf, X_test_tfidf, y_train, y_test, all_classes, calc_auc=False):
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    roc_auc = None
    if calc_auc and hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test_tfidf)
        if y_score.shape[1] < len(all_classes):
            aligned_probs = np.zeros((y_score.shape[0], len(all_classes)))
            for i, label in enumerate(model.classes_):
                aligned_probs[:, label] = y_score[:, i]
            y_score = aligned_probs
        y_test_bin = np.zeros((len(y_test), len(all_classes)))
        for i, label in enumerate(y_test):
            y_test_bin[i, label] = 1
        roc_auc = roc_auc_score(y_test_bin, y_score, multi_class='ovr')
    return accuracy, precision, recall, f1, roc_auc

# List of all unique classes
all_classes = np.arange(len(np.unique(y_train)))

# Map numeric labels to emotions
emotion_mapping = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

# Initialize tools
analyzer = SentimentIntensityAnalyzer()
nlp = spacy.load("en_core_web_sm")

# Function for Sentiment Intensity
def get_sentiment_intensity(text):
    return analyzer.polarity_scores(text)['compound']

# Function for Subjectivity
def get_subjectivity(text):
    return TextBlob(text).sentiment.subjectivity

# Function for Aspect-Based Sentiment Analysis
def aspect_sentiment_analysis(text):
    doc = nlp(text)
    aspects = {}
    for chunk in doc.noun_chunks:
        sentiment = TextBlob(chunk.text).sentiment.polarity
        aspects[chunk.text] = sentiment
    return aspects

# Streamlit app
st.title("Evaluarea modelelor de clasificare")

st.write("### Metrici de performanță")
st.write("Rezultatele sunt afișate pentru fiecare model evaluat utilizând datele procesate.")

# Funcție pentru afișarea rezultatelor
def display_results(model_name, accuracy, precision, recall, f1, roc_auc, cv_mean, cv_std):
    with st.expander(f"Rezultate pentru {model_name}"):
        st.subheader(f"Rezultate pentru {model_name}")
        st.write(f"**Acuratețe (Accuracy):** {accuracy:.4f}")
        st.write(f"**Precizie (Precision):** {precision:.4f}")
        st.write(f"**Recall:** {recall:.4f}")
        st.write(f"**F1 Score:** {f1:.4f}")
        st.write(f"**ROC-AUC:** {roc_auc:.4f}" if roc_auc is not None else "**ROC-AUC:** N/A")
        st.write(f"**Acuratețe medie (Cross-Validation Mean):** {cv_mean:.4f}")
        st.write(f"**Deviație standard (Cross-Validation Std Dev):** {cv_std:.4f}")

# Evaluare modele și afișare rezultate
for name, model in models.items():
    cv_mean, cv_std = cross_validate_model(model, X_train_bert, y_train, cv=5, scoring='accuracy')
    results = evaluate_model_optimized(model, X_train_bert, X_test_bert, y_train, y_test, all_classes, calc_auc=True)
    display_results(
        model_name=name,
        accuracy=results[0],
        precision=results[1],
        recall=results[2],
        f1=results[3],
        roc_auc=results[4],
        cv_mean=cv_mean,
        cv_std=cv_std
    )
# Casetă de text pentru introducerea unei propoziții
st.write("### Introdu o propoziție pentru predicția sentimentului")
user_input = st.text_input("Propoziția ta:", value="")
analyze_button = st.button("Analizeaza")

if user_input and analyze_button:
    # Preprocesare text utilizând aceleași reguli ca pentru datele de antrenament
    processed_input = preprocess_text(clean_text(user_input))
    input_vectorized = get_bert_embeddings([processed_input], tokenizer, bert_model)

    # Selectarea modelului (de exemplu, Logistic Regression)
    selected_model = models['Logistic Regression']
    sentiment_pred = selected_model.predict(input_vectorized)[0]
    sentiment_proba = selected_model.predict_proba(input_vectorized)[0]

    # Calculare intensitate, subiectivitate și analiza aspectelor
    sentiment_intensity = get_sentiment_intensity(user_input)
    subjectivity = get_subjectivity(user_input)
    aspect_sentiments = aspect_sentiment_analysis(user_input)
    pred_entropy = entropy(sentiment_proba)
    confidence_interval = (
        round(np.percentile(sentiment_proba, 2.5) * 100, 2),
        round(np.percentile(sentiment_proba, 97.5) * 100, 2)
    )
    sorted_probs = sorted(sentiment_proba, reverse=True)
    likelihood_ratio = sorted_probs[0] / sorted_probs[1]
    # Interpretare label pentru sentiment
    sentiment_label = emotion_mapping[sentiment_pred]
    sentiment_probabilities = {
        emotion_mapping[i]: prob for i, prob in enumerate(sentiment_proba)
    }

    # Selectare Top 3 probabilități
    top3_probabilities = sorted(sentiment_probabilities.items(), key=lambda x: x[1], reverse=True)[:3]

    # Afișarea rezultatului
    st.write("### Rezultatul predicției")
    st.write(f"**Sentiment prezis:** {sentiment_label}")

    # Afișarea intensității, subiectivității și analizelor aspectelor
    st.write(f"**Intensitatea sentimentului:** {sentiment_intensity:.4f}")
    #st.markdown(f"Value: **{sentiment_intensity:.2f}**")
    st.progress(abs(sentiment_intensity))
    st.write(f"**Subiectivitatea:** {subjectivity:.4f}")
    st.progress(subjectivity)

    with st.expander("#### Analiza aspectelor"):
        for aspect, sentiment in aspect_sentiments.items():
            st.write(f"  **Aspect:** {aspect}, **Sentiment:** {sentiment:.2f}")

    # Afișarea probabilităților pentru Top 3 clase
    #st.write("#### Top 3 Probabilități:")
    st.markdown("<h4 style='text-align:center; font-size: 24px; font-weight:bold;'>Top 3 Probabilități:</h4>", unsafe_allow_html=True)
    sentiments, probabilities = zip(*top3_probabilities)
    #for sentiment, prob in top3_probabilities:
        #st.write(f"  {sentiment}: {prob * 100:.2f}%")
    for i, (sentiment, prob) in enumerate(top3_probabilities):
        if i == 0:  # Top prediction (highest probability) in red
            st.markdown(f"<div style='text-align:center;'>{sentiment}: <span style='color:red; font-weight:bold;'>{prob * 100:.2f}%</span></div>", unsafe_allow_html=True)
        elif i == 1:  # Second highest in green
            st.markdown(f"<div style='text-align:center;'>{sentiment}: <span style='color:green; font-weight:bold;'>{prob * 100:.2f}%</span></div>", unsafe_allow_html=True)
        else:  # Third highest in blue
            st.markdown(f"<div style='text-align:center;'>{sentiment}: <span style='color:blue; font-weight:bold;'>{prob * 100:.2f}%</span></div>", unsafe_allow_html=True)

    # Grafic pentru Top 3 probabilități
    fig, ax = plt.subplots()
    ax.bar(sentiments, [p * 100 for p in probabilities], color=['red', 'green', 'blue'])
    ax.set_title("Top 3 Predicted Emotions and Their Probabilities")
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Probability (%)")
    st.pyplot(fig)
    # Calcul metrici specifice propoziției
    predicted_class_prob = sentiment_proba[sentiment_pred]  # Probabilitatea clasei prezise
    precision = predicted_class_prob  # Într-o singură propoziție, precizia poate fi probabilitatea

    # Interpretare label pentru sentiment
    sentiment_label = emotion_mapping[sentiment_pred]

    # Afișarea metricilor pentru propoziție
    with st.expander("#### Metrici specifice propoziției introduse"):
        st.write(f"**Sentiment prezis:** {sentiment_label}")
        st.write(f"**Probabilitatea clasei prezise (Precizie):** {precision * 100:.2f}%")
        st.write(f"**Subiectivitatea:** {subjectivity:.4f}")
        st.write(f"**Entropia predicției:** {pred_entropy:.4f}")
        st.write(f"**Interval de încredere:** {confidence_interval[0]}% - {confidence_interval[1]}%")
        st.write(f"**Raportul de probabilitate pentru clasa prezisă:** {likelihood_ratio:.4f}")
