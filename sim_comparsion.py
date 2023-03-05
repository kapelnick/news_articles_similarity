import pandas as pd
from top2vec import Top2Vec
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
import seaborn as sns
import matplotlib.pyplot as plt

# load model
model = Top2Vec.load("t2v.model")
topic_words, _, _ = model.get_topics()
topic_sizes, topic_nums = model.get_topic_sizes()

# read the normalized scores from the file and store them in a dictionary
with open('normalized_scores.txt', encoding='utf-8') as f:
    normalized_scores = {}
    for line in f:
        if not line.startswith('Topic'):
            topic, score = line.strip().split('\t')
            normalized_scores[int(topic)] = float(score)

# calculate the similarity between topic words
topic_word_vectors = model.topic_vectors
topic_word_cosine_similarity = cosine_similarity(topic_word_vectors)
topic_word_euclidean_similarity = 1 / (1 + euclidean_distances(topic_word_vectors))
topic_word_manhattan_similarity = 1 / (1 + manhattan_distances(topic_word_vectors))

# calculate the similarity between sentiments
sentiment_vectors = pd.DataFrame.from_dict(normalized_scores, orient='index').to_numpy()
sentiment_cosine_similarity = cosine_similarity(sentiment_vectors)
sentiment_euclidean_similarity = 1 / (1 + euclidean_distances(sentiment_vectors))
sentiment_manhattan_similarity = 1 / (1 + manhattan_distances(sentiment_vectors))

# test different combinations of similarity measures and alpha values
for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
    for topic_word_similarity, sentiment_similarity, similarity_name in [
        (topic_word_cosine_similarity, sentiment_cosine_similarity, "Cosine Similarity"),
        (topic_word_euclidean_similarity, sentiment_euclidean_similarity, "Euclidean Similarity"),
        (topic_word_manhattan_similarity, sentiment_manhattan_similarity, "Manhattan Similarity")
    ]:
        overall_similarity = alpha * topic_word_similarity + (1 - alpha) * sentiment_similarity

        # plot the heatmap
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(overall_similarity, cmap='YlGnBu')
        plt.title(f"Topic-Sentiment Similarity (Alpha={alpha}, {similarity_name})")
        plt.xlabel("Sentiments")
        plt.ylabel("Topics")
        plt.show()