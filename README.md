# SMS Spam Detector

A small Streamlit web app for an educational machine-learning project. It classifies a message as `spam` or `ham` (not spam) using TF-IDF features and Multinomial Naive Bayes.

## Run on Windows

1. Install Python 3.10 or newer.
2. Extract this project folder.
3. Open Command Prompt in the extracted folder.
4. Install packages:

   ```bash
   py -m pip install -r requirements.txt
   ```

5. Start the app:

   ```bash
   py -m streamlit run app.py
   ```

6. Open the local address Streamlit prints, usually:

   `http://localhost:8501`

The first launch needs an internet connection to download the public UCI SMS Spam Collection dataset and train the model.

## Get a public URL

This project is ready for Streamlit Community Cloud:

1. Sign in to GitHub and create a new repository.
2. Upload `app.py` and `requirements.txt` to the repository.
3. Open https://share.streamlit.io/ and sign in.
4. Choose **Create app**, select your repository, branch, and `app.py`.
5. Deploy. Streamlit will provide a public `https://....streamlit.app` URL.

You must perform the deployment from your own account; this project package does not itself create a public hosted URL.

## Dataset and method

- Dataset: UCI SMS Spam Collection, 5,574 labelled English SMS messages.
- Features: TF-IDF word features (unigrams and bigrams).
- Classifier: Multinomial Naive Bayes.
- Evaluation: stratified 80/20 train/test split.

The original dataset is older and English-language; results on modern or regional-language messages may differ. The model is an educational aid, not a security guarantee. Do not share private messages, OTPs, passwords, or other personal information with the demo.

Dataset source: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
