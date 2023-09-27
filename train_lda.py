# train_lda.py
from preprocess import tokenized_documents  # Import tokenized_documents from the preprocessing step
from gensim import corpora
import gensim

# Create a dictionary and a corpus
dictionary = corpora.Dictionary(tokenized_documents)
corpus = [dictionary.doc2bow(doc) for doc in tokenized_documents]

# Train the LDA model and print topics
lda_model = gensim.models.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=15)

for topic_id, topic_words in lda_model.print_topics():
    print(f"Topic {topic_id + 1}: {topic_words}")
