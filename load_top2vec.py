from top2vec import Top2Vec
import os

#load model
model = Top2Vec.load("t2v.model")

#returns number of topics that top2vec found in the data
num_of_topics = model.get_num_topics()
print('Number of topics in the data:')
print(num_of_topics)
print('----------------------')

#number of documents most similar to each topic. in decreasing order of size
#topic_sizes: The number of documents most similar to each topic.
#topic_nums: The unique index of every topic will be returned.
num_sim_doc_per_topic, topic_index = topic_sizes, topic_nums = model.get_topic_sizes()
print('The number of documents most similar to each topic:')
print(num_sim_doc_per_topic)
print('----------------------')

print('The unique index of every topic:')
print(topic_index)
print('----------------------')

#return topics in decreasing size
#topic_words: For each topic the top 50 words are returned, in order of semantic similarity to topic.
#word_scores: For each topic the cosine similarity scores of the top 50 words to the topic are returned.
#topic_nums: The unique index of every topic will be returned.
topic_50_words, cos_sim_score, topic_index = topic_words, word_scores, topic_nums = model.get_topics(4)
print('For each topic the top 50 words, in order of semantic similarity to topic:')
print(topic_50_words)
print('----------------------')

print('For each topic the cosine similarity scores of the top 50 words to the topic:')
print(cos_sim_score)
print('----------------------')

print('The unique index of every topic:')
print(topic_index)
print('----------------------')

#generate word clouds for the top 5 most similar topics to covid topic
# topic_words, word_scores, topic_scores, topic_nums = model.search_topics(keywords=["covid"], num_topics=4)
# for topic in topic_nums:
    # model.generate_topic_wordcloud(topic)
    
# Search documents for content semantically similar to covid and ουκρανια.
# documents, document_scores, document_ids = model.search_documents_by_keywords(keywords=["covid", "ουκρανια"], num_docs=5)
# for doc, score, doc_id in zip(documents, document_scores, document_ids):
    # print(f"Document: {doc_id}, Score: {score}")
    # print("-----------")
    # print(doc)
    # print("-----------")
    # print()