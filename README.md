# Sentiment

A simple python project to analyze the sentiment that emerges from the last tweets containing some keywords

## Getting Started

First of all you need some training data to train the Naive Bayes Classifier used in sentiment-analyzer.py.
Unfortunately, twitter policy does not allow to distribute tweet text (although i suspect if you look hard enough you can still find something online).
The solution is to classify some tweets manually, the more tweets you classify, the better your classifier will be (roughly speaking).
I wrote a script that lets you do just this: run it with

```
manual-classifier.py <number of tweets to classify> <keywords> <destination csv file for the classification> 
```
example:
```
manual-classifier.py 1000 "bitcoin, price" training_data.csv
```
at this point you can run the analyzer with the training data you classified (or some better dataset you found on the internet):

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