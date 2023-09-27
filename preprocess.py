# preprocess.py
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora
import gensim
import pandas as pd

# # Sample documents (replace with your data)
# documents = [
#     "Topic modeling is a powerful technique for analyzing text data.",
#     "Natural language processing is an exciting field in AI.",
#     "Gensim is a Python library for topic modeling.",
#     "Latent Dirichlet Allocation is a popular topic modeling algorithm.",
#     "NLP helps computers understand and generate human language.",
# ]

# Read data from the CSV file
data = pd.read_csv("reviews_data.csv")

# Extract the "Review" column as your documents
documents = data["Review"].tolist()

# Preprocess the documents
nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalpha() and word not in stop_words]
    return words

tokenized_documents = [preprocess_text(doc) for doc in documents]

# Create a dictionary and a corpus
dictionary = corpora.Dictionary(tokenized_documents)
corpus = [dictionary.doc2bow(doc) for doc in tokenized_documents]
