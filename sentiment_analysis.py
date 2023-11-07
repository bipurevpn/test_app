from transformers import pipeline
import joblib
from gensim.models import Word2Vec
import numpy as np
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import streamlit as st

# w2v_model = Word2Vec.load("app/resources/word2vec.model")
# clf = joblib.load('app/resources/random_forest_classifier.pkl')
# le = joblib.load('app/resources/label_encoder.pkl')


sentiment_pipeline = pipeline("sentiment-analysis")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

class SentimentAnalysis:

    def get_wordnet_pos(treebank_tag):
        from nltk.corpus import wordnet
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN  
        
    def preprocess_text(text):
        tokens = word_tokenize(text)
        filtered_tokens = [word.lower() for word in tokens if word.isalpha()]
        filtered_tokens = [word for word in filtered_tokens if word not in stop_words]
        pos_tags = pos_tag(filtered_tokens)
        tokens = [lemmatizer.lemmatize(word, SentimentAnalysis.get_wordnet_pos(pos)) for word, pos in pos_tags]
        return tokens

    def comment_to_vec(comment, model):
        vector = np.zeros(model.vector_size)
        num_words = 0
        for word in comment:
            if word in model.wv:
                vector += model.wv[word]
                num_words += 1
        if num_words > 0:
            vector /= num_words
        return vector

    def get_sentiment(text):
        result = sentiment_pipeline(text)[0]
        return result['label']
            
    
    @staticmethod
    def load_models_and_predict(comment):
        # Load models from the session state instead of disk
        le = st.session_state['label_encoder']
        clf = st.session_state['random_forest_classifier']
        w2v_model = st.session_state['word2vec_model']
        
        sentiment = SentimentAnalysis.get_sentiment(comment)
        preprocessed_comment = SentimentAnalysis.preprocess_text(comment)
        comment_vector = SentimentAnalysis.comment_to_vec(preprocessed_comment, w2v_model)
        predicted_tag_encoded = clf.predict([comment_vector])
        predicted_tag = le.inverse_transform(predicted_tag_encoded)
        return predicted_tag[0], sentiment