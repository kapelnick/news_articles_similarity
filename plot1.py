import os
from top2vec import Top2Vec
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load model
model = Top2Vec.load("t2v.model")

# Get document vectors
doc_vectors = model.topic_vectors

# Reduce dimensionality using t-SNE
tsne = TSNE(n_components=2, random_state=42)
doc_vectors_2d = tsne.fit_transform(doc_vectors)

# Get topic labels for each document
#topic_labels, _ = model.get_topics()
topic_sizes, topic_nums = model.get_topic_sizes()

# Plot the reduced vectors
fig, ax = plt.subplots(figsize=(10, 10))
scatter = ax.scatter(doc_vectors_2d[:, 0], doc_vectors_2d[:, 1], c=topic_nums, cmap='rainbow')
legend = ax.legend(*scatter.legend_elements(), title='Topics', loc='lower right')
ax.add_artist(legend)
plt.show()

'''
The x and y axes of the scatter plot represent the two dimensions obtained by reducing the dimensionality of the topic vectors using t-SNE.

The x and y axis in the scatter plot show the two-dimensional coordinates obtained from the t-SNE dimensionality reduction of the document vectors. The t-SNE algorithm maps high-dimensional data points to low-dimensional space while preserving the local structure of the data points as much as possible. This means that data points that are close together in the high-dimensional space are likely to be close together in the low-dimensional space as well.

In the case of Top2Vec, the document vectors for each topic are high-dimensional (they have as many dimensions as there are unique words in the corpus), but t-SNE reduces them to two dimensions for visualization purposes. Therefore, each point in the scatter plot represents a document, and the x and y coordinates of the point represent the two-dimensional coordinates obtained from the t-SNE dimensionality reduction of the document vector.

Since the scatter plot is colored by topic number, the user can see which documents belong to which topic and how they are distributed in the low-dimensional space. This can help with understanding the relationships between topics and identifying clusters of related documents.

The TSNE class is initialized with n_components=2, which means that it will produce a two-dimensional representation of the topic vectors.

The fit_transform() method of the TSNE object is then called on the topic vectors, which returns a two-dimensional array of shape (n_documents, 2) where n_documents is the number of documents in the corpus. The two columns of this array represent the two dimensions obtained by t-SNE.

The scatter() function is then used to plot the two-dimensional points on the scatter plot, with the x-axis corresponding to the first column of the doc_vectors_2d array (doc_vectors_2d[:, 0]) and the y-axis corresponding to the second column of the array (doc_vectors_2d[:, 1]).
The length of the x-axis and y-axis in the scatter plot depends on the range of values in the two columns of the doc_vectors_2d array, which are the two dimensions obtained by t-SNE.

The fit_transform() method of the TSNE object maps high-dimensional vectors to a two-dimensional space such that the distances between the vectors in the high-dimensional space are preserved as much as possible in the two-dimensional space. However, the absolute values of the distances are not preserved, which means that the scale of the two-dimensional plot does not necessarily correspond to the scale of the original high-dimensional space.

In other words, the length of the x-axis and y-axis in the scatter plot does not have any specific interpretation in terms of the original data. Instead, the positions of the points relative to each other are what matters in interpreting the plot.

The topics are plotted by hundreds because the topic_nums array contains integer values that correspond to the topic numbers assigned by the Top2Vec model.

When the scatter plot is created using scatter = ax.scatter(doc_vectors_2d[:, 0], doc_vectors_2d[:, 1], c=topic_nums, cmap='rainbow'), the c parameter is set to topic_nums, which assigns a color to each point based on its topic number. The cmap parameter specifies the color map to use, which in this case is 'rainbow'.

By default, the color map is scaled to cover the range of values in the topic_nums array. Since the topic numbers in the topic_nums array are integers, and there are typically fewer topics than documents, the color map is plotted with a relatively small number of distinct colors, usually in the hundreds. Each topic is then represented by a group of points with the same color, allowing the viewer to see the distribution of documents across topics in the scatter plot.

The plot represents the high-dimensional document vectors produced by the Top2Vec model, reduced to two dimensions using the t-SNE algorithm. Each dot on the plot represents a document, and the color of the dot corresponds to the topic to which the document was assigned by the model. The legend on the right side of the plot shows the topics and their corresponding colors.

The goal of the plot is to provide a visual representation of the distribution of documents in the high-dimensional vector space, and to allow for easy identification of groups of documents that belong to the same topic. The plot can be used for exploratory data analysis, to gain insights into the distribution of topics in the document collection, and to identify any potential anomalies or outliers.

'''