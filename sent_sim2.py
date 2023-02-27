'''this script works with files from one dir
it calculates sentiment similarity and text similarity
and writes them in a file result2.csv while also visualizing the results'''

import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt


def preprocess_text(text):
    text = text.strip()
    text = text.replace('\n', '')
    text = text.replace('\r', '')
    return text


def load_files(dir_path):
    files = []
    for subdir, _, filenames in os.walk(dir_path):
        for filename in filenames:
            file_path = os.path.join(subdir, filename)
            if file_path.endswith('.txt'):
                files.append(file_path)
    return files


def get_sentiment_score(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.last_hidden_state.mean(dim=1).squeeze()
        softmax = torch.nn.functional.softmax(logits, dim=0)
    return softmax[1].item() - softmax[0].item()


def get_similarity_score(text1, text2, model, tokenizer):
    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)
    inputs = tokenizer([text1, text2], return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return 1 - cosine(embeddings[0], embeddings[1])


def main():
    tokenizer = AutoTokenizer.from_pretrained('nlpaueb/bert-base-greek-uncased-v1')
    model = AutoModel.from_pretrained('nlpaueb/bert-base-greek-uncased-v1')

    dir_path = r'C:\Users\kapel\Desktop\test_1'
    files = load_files(dir_path)

    sentiment_scores = defaultdict(float)
    similarity_scores = defaultdict(float)
    for i in range(len(files)):
        file1 = files[i]
        with open(file1, 'r', encoding='utf-8') as f1:
            text1 = f1.read()
            sentiment_score1 = get_sentiment_score(text1, model, tokenizer)
        for j in range(i+1, len(files)):
            file2 = files[j]
            with open(file2, 'r', encoding='utf-8') as f2:
                text2 = f2.read()
                sentiment_score2 = get_sentiment_score(text2, model, tokenizer)
                
            sentiment_similarity_score = 1 - abs(sentiment_score1 - sentiment_score2)
            text_similarity_score = get_similarity_score(text1, text2, model, tokenizer)
            sentiment_scores[(file1, file2)] = sentiment_similarity_score
            similarity_scores[(file1, file2)] = text_similarity_score

    # write the results to a CSV file
    df = pd.DataFrame.from_dict({'file1': [x[0] for x in sentiment_scores.keys()],
                                 'file2': [x[1] for x in sentiment_scores.keys()],
                                 'sentiment_similarity_score': list(sentiment_scores.values()),
                                 'text_similarity_score': list(similarity_scores.values())})
    df.to_csv('results2.csv', index=False)

    # visualize the results using a heatmap
    heatmap_data = pd.pivot_table(df, values='text_similarity_score', index=['file1'], columns='file2')
    plt.figure(figsize=(10, 10))
    sns.heatmap(heatmap_data, annot=True, cmap='Blues')
    plt.show()

if __name__ == '__main__':
    main()