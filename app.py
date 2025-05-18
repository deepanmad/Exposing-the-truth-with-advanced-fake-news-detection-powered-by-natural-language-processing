import streamlit as st
import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report

st.title("📰 Fake News Detection App")
st.write("Upload **Fake.csv** and **True.csv** datasets to begin.")

uploaded_fake = st.file_uploader("Upload Fake News CSV", type="csv")
uploaded_true = st.file_uploader("Upload True News CSV", type="csv")

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def output_label(n):
    return "Fake News" if n == 0 else "Real News"

if uploaded_fake and uploaded_true:
    df_fake = pd.read_csv(uploaded_fake)
    df_true = pd.read_csv(uploaded_true)

    df_fake["class"] = 0
    df_true["class"] = 1

    df_fake_manual_testing = df_fake.tail(10)
    df_true_manual_testing = df_true.tail(10)
    df_fake = df_fake.iloc[:-10]
    df_true = df_true.iloc[:-10]

    df = pd.concat([df_fake, df_true], axis=0)
    df = df.drop(["title", "subject", "date"], axis=1)
    df = df.sample(frac=1).reset_index(drop=True)
    df["text"] = df["text"].apply(wordopt)

    x = df["text"]
    y = df["class"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    vectorization = TfidfVectorizer()
    xv_train = vectorization.fit_transform(x_train)
    xv_test = vectorization.transform(x_test)

    LR = LogisticRegression()
    DT = DecisionTreeClassifier()
    GBC = GradientBoostingClassifier(random_state=0)
    RFC = RandomForestClassifier(random_state=0)

    LR.fit(xv_train, y_train)
    DT.fit(xv_train, y_train)
    GBC.fit(xv_train, y_train)
    RFC.fit(xv_train, y_train)

    st.subheader("Test News Article")
    user_input = st.text_area("Enter the news text here...")

    if st.button("Predict"):
        if user_input:
            processed = wordopt(user_input)
            vect_input = vectorization.transform([processed])

            preds = {
                "Logistic Regression": output_label(LR.predict(vect_input)[0]),
                "Decision Tree": output_label(DT.predict(vect_input)[0]),
                "Gradient Boosting": output_label(GBC.predict(vect_input)[0]),
                "Random Forest": output_label(RFC.predict(vect_input)[0]),
            }

            for model, result in preds.items():
                st.write(f"**{model}:** {result}")
        else:
            st.warning("Please enter text for prediction.")
else:
    st.info("Please upload both datasets to proceed."
