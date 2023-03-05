from top2vec import Top2Vec
import os
import csv
import numpy as np

#load model
model = Top2Vec.load("t2v.model")
        
# Open a file for writing
with open('output.txt', 'w', encoding='utf-8') as outfile:
    
    # Write the number of topics
    num_of_topics = model.get_num_topics()
    outfile.write('Number of topics in the data: {}\n\n'.format(num_of_topics))
    
    # Write the number of documents most similar to each topic
    topic_sizes, topic_nums = model.get_topic_sizes()
    outfile.write('Number of documents most similar to each topic:\n')
    
    for i, size in enumerate(topic_sizes):
        outfile.write('Topic {}: {}\n'.format(topic_nums[i], size))
        
    outfile.write('\n')
    
    # Write the top words for each topic
    outfile.write('Top words for each topic:\n')
    topic_words, _, _ = model.get_topics()
    
    for i, words in enumerate(topic_words):
        outfile.write('Topic {}: {}\n'.format(topic_nums[i], ', '.join(words)))
        
    outfile.write('\n')
