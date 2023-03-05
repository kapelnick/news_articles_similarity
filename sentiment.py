from textblob import TextBlob
from top2vec import Top2Vec
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch

#load model
model = Top2Vec.load("t2v.model")
topic_words, _, _ = model.get_topics()
topic_sizes, topic_nums = model.get_topic_sizes()

#load GERT
# Load BERT model and tokenizer
model_name = 'nlpaueb/bert-base-greek-uncased-v1'
model2 = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def get_sentiment_score(text, model2, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model2(**inputs)
        logits = outputs.last_hidden_state.mean(dim=1).squeeze()
        softmax = torch.nn.functional.softmax(logits, dim=0)
    return softmax[1].item() - softmax[0].item()

# Open a file for writing
with open('sentiment_results2.txt', 'w', encoding='utf-8') as outfile:

    # Loop over the top words for each topic and calculate sentiment score
    for i, words in enumerate(topic_words):
        outfile.write('Sentiment for top words in Topic {}: '.format(topic_nums[i]))
        sentiment_scores = []
        for word in words:
            sentiment_score = get_sentiment_score(word, model2, tokenizer)
            sentiment_scores.append(sentiment_score)
        mean_sentiment = np.mean(sentiment_scores)
        outfile.write('Mean Sentiment Score: {}\n'.format(mean_sentiment))