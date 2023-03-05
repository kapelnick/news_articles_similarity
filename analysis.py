import pandas as pd
from top2vec import Top2Vec

#load model
model = Top2Vec.load("t2v.model")
topic_words, _, _ = model.get_topics()
topic_sizes, topic_nums = model.get_topic_sizes()

# Read the normalized scores from the file and store them in a dictionary
with open('normalized_scores.txt', encoding='utf-8') as f:
    normalized_scores = {}
    for line in f:
        if not line.startswith('Topic'):
            topic, score = line.strip().split('\t')
            normalized_scores[int(topic)] = float(score)

# Count the number of positive and negative scores
positive_count = sum(score > 0 for score in normalized_scores.values())
negative_count = sum(score < 0 for score in normalized_scores.values())

# Find the topic with the highest and minimum score
highest_topic = max(normalized_scores, key=normalized_scores.get)
minimum_topic = min(normalized_scores, key=normalized_scores.get)

# Calculate the average and standard deviation of the scores
average_score = sum(normalized_scores.values()) / len(normalized_scores)
standard_deviation = pd.Series(list(normalized_scores.values())).std()

max_words = topic_words[highest_topic]
min_words = topic_words[minimum_topic]


# Write the analysis results to a text file
with open('analysis_results.txt', 'w', encoding='utf-8') as f:
    f.write(f'Number of positive scores: {positive_count}\n')
    f.write(f'Number of negative scores: {negative_count}\n')
    f.write(f'Highest score: {normalized_scores[highest_topic]:.3f} (Topic {highest_topic})\n')
    f.write(f'Minimum score: {normalized_scores[minimum_topic]:.3f} (Topic {minimum_topic})\n')
    f.write(f'Average score: {average_score:.3f}\n')
    f.write(f'Standard deviation: {standard_deviation:.3f}\n')
    f.write(f'Top words for max sentiment (Topic {highest_topic}):\n {max_words}\n')
    f.write(f'Top words for min sentiment (Topic {minimum_topic}):\n {min_words}\n')
    




