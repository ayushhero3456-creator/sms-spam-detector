import io
import zipfile
import urllib.request
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="SMS Spam Detector", page_icon="📩", layout="centered")

DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

@st.cache_data
def load_dataset():
    with urllib.request.urlopen(DATA_URL, timeout=30) as response:
        archive_bytes = response.read()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        raw = archive.read("SMSSpamCollection")
    df = pd.read_csv(
        io.BytesIO(raw), sep="\\t", header=None,
        names=["label", "message"], encoding="utf-8"
    )
    df = df.dropna().drop_duplicates(subset="message")
    return df

@st.cache_resource
def train_model():
    df = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"], test_size=0.2,
        random_state=42, stratify=df["label"]
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english",
                                  ngram_range=(1, 2), sublinear_tf=True)),
        ("classifier", MultinomialNB())
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return model, accuracy, len(df), classification_report(
        y_test, predictions, target_names=["Not spam (ham)", "Spam"],
        output_dict=True, zero_division=0
    )

st.title("📩 SMS Spam Detector")
st.write("Enter a text message and let a machine-learning model classify it as **SPAM** or **NOT SPAM**.")
st.caption("Educational demonstration using TF-IDF and a Naive Bayes classifier.")

with st.spinner("Loading the public SMS dataset and training the model (first run may take a moment)..."):
    try:
        model, accuracy, total_messages, report = train_model()
    except Exception as exc:
        st.error("The model could not load its training dataset. Check your internet connection and refresh the app.")
        st.caption(f"Technical detail: {exc}")
        st.stop()

st.success(f"Model ready · trained on {total_messages:,} labelled SMS examples")
message = st.text_area(
    "Enter an SMS message",
    placeholder="Example: Congratulations! You have won a prize. Click here...",
    height=130
)

if st.button("🔍 Check Message", type="primary", use_container_width=True):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        prediction = model.predict([message])[0]
        probabilities = model.predict_proba([message])[0]
        classes = list(model.classes_)
        score = float(probabilities[classes.index(prediction)]) * 100

        if prediction == "spam":
            st.error("🚨 SPAM MESSAGE")
            st.write("The model considers this message likely to be spam.")
        else:
            st.success("✅ NOT SPAM (HAM)")
            st.write("The model considers this message more likely to be legitimate.")
        st.metric("Model confidence estimate", f"{score:.1f}%")
        st.caption("This score is the model's estimated probability, not a guarantee. Never click suspicious links or share OTPs, passwords, or personal information based only on this result.")

with st.expander("About this project"):
    st.markdown("""
    **How it works**
    1. The app loads the public SMS Spam Collection dataset.
    2. TF-IDF converts message text into numerical features.
    3. Multinomial Naive Bayes learns patterns associated with spam and legitimate messages.
    4. The trained model predicts the label for the message you enter.

    **Dataset:** UCI SMS Spam Collection (5,574 labelled messages in the original collection).
    **Model:** TF-IDF + Multinomial Naive Bayes.

    The dataset is an older English-language collection, so it may not recognise every modern scam, regional-language message, or new spam technique. A prediction is only an aid, not proof that a message is safe.
    """)

with st.expander("Model evaluation"):
    st.metric("Held-out test accuracy", f"{accuracy * 100:.2f}%")
    st.write("Accuracy is measured on a held-out portion of this dataset; real-world performance may differ.")
    st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

st.divider()
st.caption("Student project demonstration · Do not paste private messages or personal information.")
