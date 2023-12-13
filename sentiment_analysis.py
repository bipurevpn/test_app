import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import re
from transformers import pipeline
import joblib
from gensim.models import Word2Vec
import streamlit as st

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


class SentimentAnalysis:
    
    tags_keywords = {
        "Addon issue": [
            "dedicated ip", "static ip", "fixed ip", "private server", 
            "port", "port forwarding", "pf", "already in use", 
            "reserved ip", "permanent ip", "specific port", "port allocation", "dedicated vpn",
            "dedicate ip", "dedicatedip", "ports",
        ],
        "App Issue": [
            "asking me for feedback", "doesn't seem to work", "not working", "same server", 
            "same ip", "fetching location", "slow app launch", "new app", 
            "new version", "don't work", "doesn't work", "not working", 
            "hard to connect", "surveys", "nothing happen", "this interface", 
            "slow to load", "long to connect", "stop asking", "no ip address", 
            "ui issues", "user interface", "application crash", "app not responding", 
            "loading issue", "connection delay", "crash", "app crash", "app issue",
            "update", "latest update", 'apple tv-app', 'apple tv app', "latest version", "appletv"
        ],
        "Disconnection": [
            "disconnect", "disconnection", "disconnecting", "cut off", 
            "cut down", "reconnect", "unstable connection", "instable", 
            "shut", "shutdown", "unstable", "connection drops", "drops",
            "network drop", "loss of connection", "network failure", "connection loss", 
            "connection drop", "network instability", "connection dropped", 
            "connection down"
        ],
        "Unable to login": [
            "password", "log in", "sign in", "logout", "user id", 
            "pw", "unable to login", "fetch details", "user details", 
            "9000", "8005", "login", "login issue", "cannot login", "can not login"
            "signing in", "authentication problem", "login issue", "account access", 
            "password issue", "username problem"
        ],
        "UTB": [
            "browse", "internet", "network", "net", "wifi", "unable to browse",
            "lan", "mobile data", "hotspot", "browsing", "web", "webpage",
            "web browsing", "network access", "not loading", "site not loading", "utb"
        ],
        "Torrenting": [
            "download", "down", "torrent", "utorrent", "bittorrent", 
            "qbittorrent", "p2p", "file share", "peers", "transfer",
            "file downloading", "peer to peer", "p2p sharing", "file transfer", 
            "peer sharing", "download speed", "upload speed", "torrent speed"
        ],
        "Feature": [
            "split tunneling", "ipv6", "dns", "kill switch", "connect automatic", 
            "start automatic", "login automatic", "auto launch", "language", 
            "webrtc", "on demand", "ad block", "security features", "privacy features", 
            "app preferences", "auto-connect", "auto connect", "auto-launch", "iks", "auto-login",
            "auto login", "automatically connect", "automatically login", "autoconnect", "autologin"
            "automatically", "login automatically", "connect automatically", "shortcuts"
        ],
        "Unable to Setup/ Knowledge base": [
            "unable to setup", "don't know", "how to use", "how to access", 
            "can't figure out", "can't find", "how to connect", "how to choose", 
            "why", "where", "what", "unable", "don't understand", "cannot", 
            "didn't work", "install", "not sure", "how do", "setup issue", "installation problem", "usage guide", "accessibility", 
            "configuration problem", "setup guide"
        ],
        "Server Location": [
            "more cities", "more servers", "more locations", "more countries", 
            "costa rica", "kuwait", "taiwan", "indonesia", "malaysia", 
            "bangalore", "thailand", "mexico", "no china", "columbia", 
            "new zealand", "quebec", "nz", "venezuela", "server location", "server"
            "server options", "location availability", "server availability", 
            "regional servers", "server choice", "location options"
        ],
        "Invalid IP Location": [
            "wrong location", "wrong country", "incorrect server", 
            "ip locations doesn't match", "false location", "location not correct",
            "incorrect location", "wrong ip", "location error", "server mismatch", 
            "geolocation error", "ip discrepancy", "incorrect ip", "ip location",
            "invalid ip location", "ip location error", "ip location mismatch",
        ],
        "IP Exposed": [
            "public ip", "local ip", "real ip", "location detected", 
            "my location", "ip no mask", "whatismyip", "my ip address", 
            "current location", "doesn't change location", "detect real location",
            "exposed address", "visible ip", "detectable location", "location exposure", 
            "ip leak", "ip visibility", "ip detected"
        ],
        "Support": [
            "support", "customer service", "router", "tech support", 
            "representative", "remote desktop", "rdp", "no help", "no assistance", 
            "chat", "automated answer", "bot", "human", "chromecast", 
            "casting", "complicated", "mirror",
            "customer care", "technical assistance", "help desk", "tech help", 
            "customer support", "service support", "technical support", "live chat"
        ],
        "Others": [
            "same ip", "ip not changing", 
            "more ip", "ip blacklist", "fraud",
            "others"
        ],
        "Gaming": [
            "game", "xbox", "wrestling", "call of duty", "pubg", 
            "minecraft", "gaming", "playstation", "video games", "online gaming", 
            "multiplayer games", "games"
        ],
        "Geo restricted": [
            "china", "iran", "russia", "uae", "turkey", "egypt",
            "geoblocking", "location restriction", "access denied", "region locked", 
            "regional block", "country restriction"
        ],
        "Slow Speed": [
            "slow", "speed", "kbps", "upload", "lag", "buffer", 
            "browsing stuck frequently", "slow speed",
            "low bandwidth", "network lag", "slow connection", "internet lag", 
            "slow browsing", "slow download", "not fast", "high ping", "slow internet",
            "high latency", "lagging","speed issue", "speed problem", "speed test", "speedtest", "speedtest.net",
            "freeze", "freezes", "freezing", "slowing"
        ],
        "Content": [
            ".com", "content", "stream", "access", "video", "play", 
            "bbc", "itv", "youtube", "hotstar", "iptv", "netflix", "prime",
            "media access", "streaming services", "video streaming", "online video", 
            "media streaming", "digital content", "streaming content", "online content", 
            "peacock", "hulu", "disney", "disney+", "hbo", "hbo max", "hbo+", "hbomax", 
            "hboplus", "hboplus", "hboplusmax", "hboplus+", "hboplusmax", "hboplus+", "cannot watch"
            "tv shows", "tv", 'buffering', "buffer", "buffers", "watch", "watchable", "instagram",
            "facebook", "twitter", "throttling", "throttled", "throttle", "throttles", "bbc", 
            "news", "streaming", "stream", "page", "site", "brave search", "google", "bing", "yahoo",
            "skybet", "channel", "sports"
        ],
        "UTC": [
            "unable to connect", "cannot connect", "connection issue", "connection problem",
            "connection error", "connection failed", "connection failure", "connection timeout",
            "connection lost", "connection loss", "can't connect", "cannot connect", "not connecting", 
            "not able to connect", "won't connect", "cant connect"
        ],
        "Cancellation": [
            "cancel", "cancellation", "refund", "money back", "refunded", "cancelled"
        ],
        "Suggestions": [
            "suggestion", "recommendation", "feature request", "feature suggestion",
            "suggest", "recommend", "request", "suggestion", "recommendations"
        ],
    }

    priority_order = ['Addon issue', 'Gaming', 'Feature', 'Content', 'UTB', 'Disconnection', 'Slow Speed', 'UTC', 'App Issue', 'Geo restricted', 'Unable to login', 'Torrenting', 'Unable to Setup/ Knowledge base', 'Server Location', 'Invalid IP Location', 'IP Exposed', 'Support', 'Others', 'Suggestions', 'Cancellation']


    sentiment_pipeline = pipeline("sentiment-analysis")
    stop_words = set(nltk.corpus.stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    @staticmethod
    def get_wordnet_pos(treebank_tag):
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

    @staticmethod
    def preprocess_text(text):
        tokens = word_tokenize(text)
        filtered_tokens = [word.lower() for word in tokens if word.isalpha()]
        filtered_tokens = [word for word in filtered_tokens if word not in SentimentAnalysis.stop_words]
        pos_tags = pos_tag(filtered_tokens)
        tokens = [SentimentAnalysis.lemmatizer.lemmatize(word, SentimentAnalysis.get_wordnet_pos(pos)) for word, pos in pos_tags]
        return tokens

    @staticmethod
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

    @staticmethod
    def get_sentiment(text):
        result = SentimentAnalysis.sentiment_pipeline(text)[0]
        return result['label']

    @staticmethod
    def preprocess_and_tag(dataframe, tags_keywords, priority_order):
        dataframe['Content'] = dataframe['Content'].str.lower()
        dataframe['Initial_Tag'] = 'Unassigned'

        slow_speed_regex = r'\d+\s?(mb|kb|mbps|kbps|mps|kbs)'  # Regex for 'Slow Speed'

        for index, row in dataframe.iterrows():
            # Check for 'Slow Speed' using the regex
            if re.search(slow_speed_regex, row['Content'], re.IGNORECASE):
                dataframe.at[index, 'Initial_Tag'] = "Slow Speed"
                continue  # Move to the next row

            # If 'Slow Speed' regex does not match, proceed with other checks
            for tag, keywords in tags_keywords.items():
                for keyword in keywords:
                    if SentimentAnalysis.contains_keyword(row['Content'], keyword):
                        dataframe.at[index, 'Initial_Tag'] = tag
                        break

        return dataframe

    @staticmethod
    def contains_keyword(text, keyword):
        pattern = r'\b' + re.escape(keyword) + r'\b'
        return re.search(pattern, text, re.IGNORECASE) is not None

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













# from transformers import pipeline
# import joblib
# from gensim.models import Word2Vec
# import numpy as np
# from transformers import pipeline
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from nltk import pos_tag
# import streamlit as st
# import nltk

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

# sentiment_pipeline = pipeline("sentiment-analysis")

# stop_words = set(stopwords.words('english'))
# lemmatizer = WordNetLemmatizer()

# class SentimentAnalysis:

#     def get_wordnet_pos(treebank_tag):
#         from nltk.corpus import wordnet
#         if treebank_tag.startswith('J'):
#             return wordnet.ADJ
#         elif treebank_tag.startswith('V'):
#             return wordnet.VERB
#         elif treebank_tag.startswith('N'):
#             return wordnet.NOUN
#         elif treebank_tag.startswith('R'):
#             return wordnet.ADV
#         else:
#             return wordnet.NOUN  
        
#     def preprocess_text(text):
#         tokens = word_tokenize(text)
#         filtered_tokens = [word.lower() for word in tokens if word.isalpha()]
#         filtered_tokens = [word for word in filtered_tokens if word not in stop_words]
#         pos_tags = pos_tag(filtered_tokens)
#         tokens = [lemmatizer.lemmatize(word, SentimentAnalysis.get_wordnet_pos(pos)) for word, pos in pos_tags]
#         return tokens

#     def comment_to_vec(comment, model):
#         vector = np.zeros(model.vector_size)
#         num_words = 0
#         for word in comment:
#             if word in model.wv:
#                 vector += model.wv[word]
#                 num_words += 1
#         if num_words > 0:
#             vector /= num_words
#         return vector

#     def get_sentiment(text):
#         result = sentiment_pipeline(text)[0]
#         return result['label']
            
    
#     @staticmethod
#     def load_models_and_predict(comment):
#         # Load models from the session state instead of disk
#         le = st.session_state['label_encoder']
#         clf = st.session_state['random_forest_classifier']
#         w2v_model = st.session_state['word2vec_model']
        
#         sentiment = SentimentAnalysis.get_sentiment(comment)
#         preprocessed_comment = SentimentAnalysis.preprocess_text(comment)
#         comment_vector = SentimentAnalysis.comment_to_vec(preprocessed_comment, w2v_model)
#         predicted_tag_encoded = clf.predict([comment_vector])
#         predicted_tag = le.inverse_transform(predicted_tag_encoded)
#         return predicted_tag[0], sentiment
