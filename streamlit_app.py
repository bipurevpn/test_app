import streamlit as st
import pandas as pd
from io import BytesIO
import requests
from gensim.models import Word2Vec
from sentiment_analysis import SentimentAnalysis
import tempfile
import os
import joblib
from bs4 import BeautifulSoup

def download_model_from_google_drive(link):
    # Extract the file ID from the Google Drive link
    file_id = link.split('/d/')[1].split('/')[0]
    initial_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    # Session for maintaining persistent connection
    session = requests.Session()

    # Get the first response
    response = session.get(initial_url, stream=True)
    token = None

    # Check if there is a confirmation token (for large files)
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    # If there is a token, append it to the download URL
    if token:
        model_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={token}"
    else:
        model_url = initial_url

    # Download the file
    response = session.get(model_url, stream=True)
    if response.status_code == 200:
        # Save the content to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp_file:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:  # filter out keep-alive new chunks
                    tmp_file.write(chunk)
            tmp_file_path = tmp_file.name
        
        # Load the model from the temporary file
        model = joblib.load(tmp_file_path)
        return model
    else:
        st.error("Failed to download the model. Please check the link.")
        return None

class StreamlitApp:

    def __init__(self):
        st.title("Sentiment Analysis")
        self.initialize_session_state()
        self.run_app()

    def initialize_session_state(self):
        if 'reset' not in st.session_state:
            st.session_state['reset'] = False
        # Initialize the model components in session state if not already present
        for key in ['label_encoder', 'word2vec_model', 'random_forest_classifier']:
            if key not in st.session_state:
                st.session_state[key] = None

    def run_app(self):
        if st.sidebar.button('Reset'):
            st.session_state['reset'] = True
            # Reset the model components
            for key in ['label_encoder', 'word2vec_model', 'random_forest_classifier']:
                st.session_state[key] = None
            st.experimental_rerun()

        # Always show the file uploader components regardless of the download button status
        uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"], key='file_uploader')

        # Only show uploaders for the model components if they haven't been uploaded yet
        if st.session_state['label_encoder'] is None:
            uploaded_label_encoder = st.sidebar.file_uploader("Upload the Label Encoder file", type=["pkl"], key='label_encoder_uploader')
            if uploaded_label_encoder is not None:
                st.session_state['label_encoder'] = joblib.load(uploaded_label_encoder)

#        if st.session_state['random_forest_classifier'] is None:
#            uploaded_random_forest_classifier = st.sidebar.file_uploader("Upload the Classifier file", type=["joblib"], key='random_forest_classifier_uploader')
#            if uploaded_random_forest_classifier is not None:
#                st.session_state['random_forest_classifier'] = joblib.load(uploaded_random_forest_classifier)

#        if st.session_state['word2vec_model'] is None:
#            uploaded_word2vec_model = st.sidebar.file_uploader("Upload the Word2Vec Model file", type=["model"], key='word2vec_model_uploader')
#            if uploaded_word2vec_model is not None:
#                # Save the uploaded file to a temporary file and load it
#                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
#                    tmp_file.write(uploaded_word2vec_model.read())
#                    st.session_state['word2vec_model'] = Word2Vec.load(tmp_file.name)
#                    os.unlink(tmp_file.name)

        if st.session_state['word2vec_model'] is None:
            uploaded_word2vec_model = st.sidebar.file_uploader("Upload the Word2Vec Model file", type=["joblib"], key='word2vec_model_uploader')
            if uploaded_word2vec_model is not None:
                # Load the uploaded joblib file directly into the session state
                st.session_state['word2vec_model'] = joblib.load(uploaded_word2vec_model)

        
#        if st.session_state['encryption_key'] is None:
#            encryption_key = st.sidebar.text_input("Enter the encryption key", type="password", key='encryption_key_input')
#            if encryption_key:
#                st.session_state['encryption_key'] = encryption_key

        
#        if st.session_state['random_forest_classifier'] is None:
#            uploaded_random_forest_classifier = st.sidebar.file_uploader("Upload the Classifier file", type=["joblib"], key='random_forest_classifier_uploader')
#            if uploaded_random_forest_classifier is not None:
#                st.session_state['random_forest_classifier'] = uploaded_random_forest_classifier

         # Input for the Google Drive link
        if st.session_state['random_forest_classifier'] is None:
            model_link = st.sidebar.text_input("Enter the Google Drive link for the Classifier file", key='model_link_input')
            if model_link:
                st.session_state['random_forest_classifier'] = download_model_from_google_drive(model_link)
                                                                                                
        if uploaded_file is not None and all(st.session_state[key] is not None for key in ['label_encoder', 'word2vec_model', 'random_forest_classifier']):
            comments_file = pd.read_csv(uploaded_file)
            st.subheader("Uploaded Data:")
            st.dataframe(comments_file)
            comments_file['Predicted Tag'], comments_file['Sentiment'] = zip(*comments_file.iloc[:, 0].apply(self.process_and_analyze_comment))
            st.subheader("Processed Data:")
            st.dataframe(comments_file)
            csv = self.to_csv(comments_file)
            st.sidebar.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='processed_comments.csv',
                mime='text/csv',
            )

    @staticmethod
    def process_and_analyze_comment(comment):
        predicted_tag, sentiment = SentimentAnalysis.load_models_and_predict(comment)
        return predicted_tag, sentiment

    @staticmethod
    def to_csv(df):
        output = BytesIO()
        df.to_csv(output, index=False)
        processed_data = output.getvalue()
        return processed_data

if __name__ == "__main__":
    StreamlitApp()
