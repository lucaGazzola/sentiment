# Sentiment

A simple sentiment analysis python project. It analyzes the sentiment that emerges from the last tweets containing user specified keywords

## Getting Started

First of all you need some training data to train the Naive Bayes Classifier used in sentiment-analyzer.py.
Unfortunately, twitter policy does not allow to distribute tweet text (although i suspect if you look hard enough you can still find something online).
The solution is to classify some tweets manually, the more tweets you classify, the better your classifier will be (roughly speaking).
I wrote a script that lets you do just this. First of all you need to get a twitter API key, which you can find at https://apps.twitter.com/ , so that you can search through the most recent tweets for a specific topic. Once you obtained the needed keys, put them in the config file.
Now you can manually classify tweets running:

```
manual-classifier.py <number of tweets to classify> <keywords> <destination csv file for the classification> 
```
example:
```
manual-classifier.py 1000 "bitcoin, price" training_data.csv
```
at this point you can run the analyzer with the training data you classified (or some better dataset you found on the internet).
The csv format is <tweet, sentiment> where tweet is a string and sentiment is an integer:

0: negative sentiment
1: neutral sentiment
2: positive sentiment

example: <"Great day at work!",2>

run the analyzer with:

```
sentiment-analyzer.py <number of tweets to classify> <keywords> <input csv file needed to train the classifier> 
```
example:
```
sentiment-analyzer.py 100 "bitcoin, price" training_data.csv
```

you should see in the output how many tweets have been classified as positive, negative and neutral.

### Prerequisites

python 3.6

python libs:

csv
tweepy
re
urllib
nltk
config
sys
