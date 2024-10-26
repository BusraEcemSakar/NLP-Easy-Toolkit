import nltk
import re
import spacy
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from collections import Counter

# Download necessary resources for nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

class NLPToolkit:
    def __init__(self):
        # Text Preprocessing components
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.nlp_spacy = spacy.load("en_core_web_sm")  # SpaCy for tokenization and NER
        
        # Sentiment Analysis components
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.bert_analyzer = pipeline("sentiment-analysis")

    # ----- TEXT PREPROCESSING METHODS -----
    def clean_text(self, text):
        """Clean text by removing URLs, emails, and non-alphanumeric characters."""
        text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
        return text.lower()

    def tokenize(self, text, use_spacy=True):
        """Tokenize text using spaCy or nltk."""
        if use_spacy:
            doc = self.nlp_spacy(text)
            return [token.text for token in doc if not token.is_punct]
        else:
            return nltk.word_tokenize(text)

    def remove_stopwords(self, tokens):
        """Remove stopwords from token list."""
        return [word for word in tokens if word.lower() not in self.stop_words]

    def stem(self, tokens):
        """Apply stemming to token list."""
        return [self.stemmer.stem(word) for word in tokens]

    def lemmatize(self, tokens):
        """Apply lemmatization to token list."""
        return [self.lemmatizer.lemmatize(word) for word in tokens]

    # ----- SENTIMENT ANALYSIS METHODS -----
    def analyze_sentiment_textblob(self, text):
        """Analyze sentiment using TextBlob."""
        blob = TextBlob(text)
        return blob.sentiment.polarity, blob.sentiment.subjectivity

    def analyze_sentiment_vader(self, text):
        """Analyze sentiment using Vader."""
        return self.vader_analyzer.polarity_scores(text)

    def analyze_sentiment_bert(self, text):
        """Analyze sentiment using BERT model."""
        return self.bert_analyzer(text)[0]

    def get_sentiment_category(self, text, method="textblob"):
        """Categorize sentiment as Positive, Negative, or Neutral."""
        if method == "textblob":
            polarity, _ = self.analyze_sentiment_textblob(text)
        elif method == "vader":
            polarity = self.analyze_sentiment_vader(text)['compound']
        elif method == "bert":
            bert_result = self.analyze_sentiment_bert(text)
            return bert_result['label'] if bert_result else "Neutral"
        else:
            raise ValueError("Method should be 'textblob', 'vader', or 'bert'.")

        if polarity > 0:
            return 'Positive'
        elif polarity < 0:
            return 'Negative'
        else:
            return 'Neutral'

    # ----- NAMED ENTITY RECOGNITION (NER) METHODS -----
    def get_entities(self, text):
        """Extract named entities from text using spaCy."""
        doc = self.nlp_spacy(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities

    def plot_entities(self, text):
        """Plot a bar chart of entity types in text."""
        doc = self.nlp_spacy(text)
        labels = [ent.label_ for ent in doc.ents]
        label_counts = Counter(labels)
        
        # Plot entity distribution
        plt.figure(figsize=(10, 5))
        plt.bar(label_counts.keys(), label_counts.values())
        plt.title("Entity Distribution")
        plt.xlabel("Entity Type")
        plt.ylabel("Frequency")
        plt.show()
