import streamlit as st
import pandas as pd
from io import BytesIO
from gensim.models import Word2Vec
from sentiment_analysis import SentimentAnalysis
import tempfile
import os
import joblib

class StreamlitApp:

    def __init__(self):
        st.title("Sentiment Analysis")
        self.initialize_session_state()
        self.run_app()

    def initialize_session_state(self):
        if 'reset' not in st.session_state:
            st.session_state['reset'] = False
        # Initialize the model components in session state if not already present
        for key in ['label_encoder', 'random_forest_classifier', 'word2vec_model']:
            if key not in st.session_state:
                st.session_state[key] = None

    def run_app(self):
        if st.sidebar.button('Reset'):
            st.session_state['reset'] = True
            # Reset the model components
            for key in ['label_encoder', 'random_forest_classifier', 'word2vec_model']:
                st.session_state[key] = None
            st.experimental_rerun()

        # Always show the file uploader components regardless of the download button status
        uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"], key='file_uploader')

        # Only show uploaders for the model components if they haven't been uploaded yet
        if st.session_state['label_encoder'] is None:
            uploaded_label_encoder = st.sidebar.file_uploader("Upload the Label Encoder file", type=["pkl"], key='label_encoder_uploader')
            if uploaded_label_encoder is not None:
                st.session_state['label_encoder'] = joblib.load(uploaded_label_encoder)

        if st.session_state['random_forest_classifier'] is None:
            uploaded_random_forest_classifier = st.sidebar.file_uploader("Upload the Classifier file", type=["joblib"], key='random_forest_classifier_uploader')
            if uploaded_random_forest_classifier is not None:
                st.session_state['random_forest_classifier'] = joblib.load(uploaded_random_forest_classifier)

        if st.session_state['word2vec_model'] is None:
            uploaded_word2vec_model = st.sidebar.file_uploader("Upload the Word2Vec Model file", type=["model"], key='word2vec_model_uploader')
            if uploaded_word2vec_model is not None:
                # Save the uploaded file to a temporary file and load it
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(uploaded_word2vec_model.read())
                    st.session_state['word2vec_model'] = Word2Vec.load(tmp_file.name)
                    os.unlink(tmp_file.name)

        if uploaded_file is not None and all(st.session_state[key] is not None for key in ['label_encoder', 'random_forest_classifier', 'word2vec_model']):
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
