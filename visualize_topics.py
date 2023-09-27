# visualize_topics.py
from train_lda import lda_model, corpus  # Import lda_model and corpus from train_lda.py
import matplotlib.pyplot as plt


# Get the topic distribution for each document
topic_distribution = [lda_model[doc] for doc in corpus]

# Extract the topic weights
topic_weights = [[weight for _, weight in topics] for topics in topic_distribution]

# Transpose to have documents as rows and topics as columns
topic_weights_transposed = list(zip(*topic_weights))

# Plot the topic distribution for each document
for i, weights in enumerate(topic_weights_transposed):
    plt.bar(range(len(weights)), weights, tick_label=[f"Topic {i + 1}" for i in range(len(weights))])
    plt.title(f"Document {i + 1}")
    plt.xlabel("Topics")
    plt.ylabel("Topic Weight")
    plt.show()
