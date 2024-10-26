
# NLP Easy Toolkit

An open-source NLP toolkit designed to simplify text processing and analysis. This toolkit includes essential NLP functionalities, from text preprocessing to sentiment analysis and named entity recognition (NER), all with easy-to-use functions that can be integrated directly into your workflows.

---

## 🌟 Features

1. **Text Preprocessing**
   - Tokenization (using both NLTK and spaCy)
   - Stopword Removal
   - Stemming and Lemmatization
   - Text Cleaning (removal of URLs, emails, punctuation, etc.)

2. **Sentiment Analysis**
   - Sentiment analysis using TextBlob, Vader, and BERT-based models.
   - Get sentiment polarity (Positive, Neutral, Negative) with fine-grained control.

3. **Named Entity Recognition (NER)**
   - Extract and identify entities (e.g., people, organizations, locations).
   - Visualize entity distribution with a simple bar chart.

---

## 🛠️ Installation

### Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/NLP-Easy-Toolkit.git
cd NLP-Easy-Toolkit
```

### Install Dependencies
Install the necessary Python libraries by running:
```bash
pip install -r requirements.txt
```

### Download spaCy Language Model
For NER and advanced tokenization, install the spaCy language model:
```bash
python -m spacy download en_core_web_sm
```

---

## 🚀 Usage

Below are examples of how to use various modules within the toolkit. Import the toolkit and initialize it as shown below:

```python
from nlp_toolkit import NLPToolkit

# Initialize Toolkit
toolkit = NLPToolkit()
```

### 1. Text Preprocessing

```python
text = "Hello! This is a test for NLP Easy Toolkit. Visit https://example.com for more info."
cleaned_text = toolkit.clean_text(text)
tokens = toolkit.tokenize(cleaned_text)
print("Tokens:", tokens)

# Remove stopwords, apply stemming and lemmatization
filtered_tokens = toolkit.remove_stopwords(tokens)
stemmed_tokens = toolkit.stem(filtered_tokens)
lemmatized_tokens = toolkit.lemmatize(filtered_tokens)
print("Lemmatized Tokens:", lemmatized_tokens)
```

### 2. Sentiment Analysis

```python
text = "I love this tool! It's so helpful and user-friendly."

# Analyze sentiment using different methods
sentiment_textblob = toolkit.get_sentiment_category(text, method="textblob")
sentiment_vader = toolkit.get_sentiment_category(text, method="vader")
sentiment_bert = toolkit.get_sentiment_category(text, method="bert")

print("TextBlob Sentiment:", sentiment_textblob)
print("Vader Sentiment:", sentiment_vader)
print("BERT Sentiment:", sentiment_bert)
```

### 3. Named Entity Recognition (NER)

```python
text = "Apple is looking at buying a UK startup for $1 billion."

# Extract and visualize entities
entities = toolkit.get_entities(text)
print("Entities:", entities)
toolkit.plot_entities(text)
```

---

## 📊 Examples

Check the **examples** folder for Jupyter Notebooks that demonstrate each functionality of the toolkit. These include:
- **Sentiment Analysis**: Examples using TextBlob, Vader, and BERT.
- **Text Preprocessing**: Steps from cleaning text to tokenizing and lemmatizing.
- **Named Entity Recognition**: Entity extraction with spaCy and visualization of entity distributions.

---

## 🤝 Contributing

Contributions are welcome! 

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

For any questions, feel free to reach out or open an issue. Happy coding!
