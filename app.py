
import streamlit as st
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download('stopwords')

model = pickle.load(open('fake_news_model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)
sample_text = "The government announced new policies today."
sample_vec = tfidf.transform([preprocess(sample_text)])
print(model.predict(sample_vec))


# Load trained model and vectorizer
model = pickle.load(open('fake_news_model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Streamlit UI
st.title("Fake News Detector")

user_input = st.text_area("Enter news text")

if st.button("Predict"):
    input_vector = vectorizer.transform([user_input])
    prediction = model.predict(input_vector)[0]
    result = "Fake News" if prediction == 0 else "Real News"
    st.subheader(f"Prediction: {result}")


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

st.title("📰 Fake News Detector")
user_input = st.text_area("Enter a news article or headline to check:", height=200)

if st.button("Check"):
    cleaned = clean_text(user_input)
    vector = tfidf.transform([cleaned]).toarray()
    prediction = model.predict(vector)

    if prediction[0] == 1:
        st.success("✅ This looks like REAL News.")
    else:
        st.error("⚠️ This appears to be FAKE News.")
