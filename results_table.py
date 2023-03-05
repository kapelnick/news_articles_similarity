import matplotlib.pyplot as plt
import pandas as pd
import statistics

# Read the sentiment scores from the file and store them in a dictionary
with open('sentiment_results2.txt') as f:
    topic_scores = {}
    for line in f:
        if line.startswith('Sentiment for top words in Topic'):
            topic = int(line.split(':')[0].split()[-1])
            score = float(line.split(':')[2].split()[0])
            topic_scores[topic] = score
            
# Compute the mean and standard deviation of the scores
mean_score = statistics.mean(topic_scores.values())
stddev_score = statistics.stdev(topic_scores.values())

# Normalize the scores using z-score normalization
normalized_scores = {topic: (score - mean_score) / stddev_score for topic, score in topic_scores.items()}

# Sort the normalized scores in descending order of score
sorted_scores = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)

# Write the sorted scores to a table in txt format
with open('sorted_scores.txt', 'w') as f:
    f.write('Topic\tNormalized Score\n')
    for topic, score in sorted_scores:
        f.write(f'{topic}\t{score:.3f}\n')

# Write the normalized scores to a table in txt format
with open('normalized_scores.txt', 'w') as f:
    f.write('Topic\tNormalized Score\n')
    for topic, score in normalized_scores.items():
        f.write(f'{topic}\t{score:.3f}\n')

# # Plot the results as a scatter plot
# plt.scatter(normalized_scores.keys(), normalized_scores.values(), color='#21618C', alpha=0.8)
# plt.xlabel('Topic')
# plt.ylabel('Normalized Score')
# plt.title('Sentiment Scores by Topic')
# plt.show()

# Plot the distribution of the normalized scores as a histogram
plt.figure(figsize=(8, 6))
plt.hist(normalized_scores.values(), bins=20, alpha=0.5, color='#5EB1BF', edgecolor='black')
plt.xlabel('Normalized Score')
plt.ylabel('Frequency')
plt.title('Distribution of Normalized Sentiment Scores')
plt.show()


'''
This code computes the mean and standard deviation of the sentiment scores using the mean() and stdev() functions from the statistics module, and then uses these values to perform z-score normalization on the scores.
'''
'''
The frequency axis in the histogram represents the number of topics that fall into each bin. The histogram bins represent ranges of normalized scores, and the height of each bin represents the number of topics that have normalized scores within that range. The specific values on the frequency axis will depend on the data being plotted, but generally, the height of each bin will be proportional to the number of topics in that range. For example, if there are 100 topics with normalized scores between -2 and -1, and 50 topics with normalized scores between -1 and 0, the height of the bin representing the range -2 to -1 will be twice the height of the bin representing the range -1 to 0. By looking at the histogram, you can get a sense of the distribution of normalized scores and how many topics fall into each range.
'''