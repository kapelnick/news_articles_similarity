from top2vec import Top2Vec
import os

# Specify the directory containing the text files
dir_path = r'C:\Users\kapel\Desktop\dataset'

# Create an empty list to store the text data
documents = []

# Iterate through each file in the directory
for file_name in os.listdir(dir_path):

    # Check if the file is a text file
    if file_name.endswith('.txt'):

        # Construct the full path to the file
        file_path = os.path.join(dir_path, file_name)

        # Open the file, read its contents, and add it to the list of documents
        with open(file_path, encoding='utf-8') as f:
            documents.append(f.read())

# Train the Top2Vec model on the list of documents
model = Top2Vec(documents=documents, speed='deep-learn', workers=16)

# Save the model to disk
model.save('t2v.model')
