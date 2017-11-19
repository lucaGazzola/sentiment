'''
Created on 19 Nov 2017

Trains a Naive Bayes Classifier on some training data and then assesses the sentiment shown in the last 'num_tweets' tweets containing 'keywords'
part of the code is taken from the example found at: https://www.ravikiranj.net/posts/2012/code/how-build-twitter-sentiment-analyzer/

@author: Luca G.
'''

import csv
import tweepy
import re
import urllib
import nltk
import config
import sys

feature_list = []
stop_words = []


def main():

    # Setup
    
    if len(sys.argv) < 4:
        print('usage: sentiment-analyzer.py <number of tweets to analyze -- example: 100> <keywords -- example: "bitcoin, price"> <input training data -- example: training_data.csv')
        sys.exit()

    num_tweets = sys.argv[1]
    keywords = sys.argv[2]
    training_data_file = sys.argv[3]

    # Training

    nb_classifier = train(training_data_file)

    # Classification

    classification = classify(num_tweets, keywords, nb_classifier)

    print('positive tweets: {0}'.format(classification.get('positive')))
    print('negative tweets: {0}'.format(classification.get('negative')))
    print('neutral tweets: {0}'.format(classification.get('neutral')))
    if classification.get('positive') > classification.get('negative'):
        print('Overall sentiment: positive')
    elif classification.get('positive') < classification.get('negative'):
        print('Overall sentiment: negative')
    else:
        print('Overall sentiment: neutral')


def train(training_data_file):

    global feature_list
    global stop_words

    # Read the tweets in the training dataset one by one and process them
    input_tweets = csv.reader(open(training_data_file, 'r'), delimiter=',')
    tweets = []
    stop_words = get_stop_word_list()
    next(input_tweets, None)
    for row in input_tweets:
        if row[0] == '0':
            sentiment = 'negative'
        elif row[0] == '1':
            sentiment = 'neutral'
        else:
            sentiment = 'positive'
        tweet = row[1]
        processed_tweet = process_tweet(tweet)
        feature_vector = get_feature_vector(processed_tweet)
        feature_list.extend(feature_vector)
        tweets.append((feature_vector, sentiment))

    # remove duplicates
    feature_list = list(set(feature_list))

    # Extract feature vector for all tweets in one shot
    training_set = nltk.classify.util.apply_features(extract_features, tweets)

    # Train the classifier
    naive_bayes_classifier = nltk.NaiveBayesClassifier.train(training_set)

    print(naive_bayes_classifier.show_most_informative_features(50))

    return naive_bayes_classifier


def classify(num_tweets, keywords, classifier):

    # Twitter API setup
    auth = tweepy.OAuthHandler(config.twitter_api['consumer_key'], config.twitter_api['consumer_secret'])
    auth.set_access_token(config.twitter_api['access_token'], config.twitter_api['access_token_secret'])
    twitter_api = tweepy.API(auth)

    # get the last 'num_tweets' tweets containing 'keywords'
    unprocessed_tweets = twitter_api.search(q=[keywords], count=num_tweets)

    count_positive = 0
    count_negative = 0
    count_neutral = 0
    total = 0
    
    # classifies each tweet with its sentiment
    for tweet in unprocessed_tweets:
        encoded = tweet.text.encode("utf-8")
        processed = process_tweet(encoded)
        print(processed)
        if classifier.classify(extract_features(get_feature_vector(processed))) == 'positive':
            count_positive += 1
            print('positive')
        elif classifier.classify(extract_features(get_feature_vector(processed))) == 'negative':
            count_negative += 1
            print('negative')
        else:
            count_neutral += 1
            print('neutral')
        total += 1

    return dict((('positive', count_positive), ('negative', count_negative), ('neutral', count_neutral)))


# start extract_features
def extract_features(tweet):

    global feature_list
    # remove duplicates
    feature_list = list(set(feature_list))
    tweet_words = set(tweet)
    features = {}
    for word in feature_list:
        features['contains(%s)' % word] = (word in tweet_words)
    return features


# processes tweets before they can be analyzed
def process_tweet(tweet):

    # Convert to lower case
    tweet = tweet.lower()
    # Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', str(tweet))
    # Convert @username to AT_USER
    tweet = re.sub('@[^\s]+', 'AT_USER', str(tweet))
    # Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', str(tweet))
    # Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', str(tweet))
    # trim
    tweet = tweet.strip('\'"')
    return tweet


# replace two+ letter occurrences
def replace_two_or_more(s):

    # look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1+}", re.DOTALL)
    return pattern.sub(r"\1\1", s)


# get list of stop words
def get_stop_word_list():

    global stop_words

    # add fixed stopwords
    stop_words = ['AT_USER', 'URL']

    # read stopwords from url
    # noinspection PyUnresolvedReferences
    data = urllib.request.urlopen('https://github.com/ravikiranj/twitter-sentiment-analyzer/blob/master/data/feature_list/stopwords.txt')

    for line in data:
        word = line.strip()
        stop_words.append(word)
    return stop_words


# get the feature vector
def get_feature_vector(tweet):

    global stop_words
    feature_vector = []
    # split tweet into words
    words = tweet.split()
    for w in words:
        # replace two or more with two occurrences
        w = replace_two_or_more(w)
        # strip punctuation
        w = w.strip('\'"?,.')
        # check if the word stats with an alphabet
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
        # ignore if it is a stop word
        if w in stop_words or val is None:
            continue
        else:
            feature_vector.append(w.lower())
    return feature_vector


if __name__ == '__main__':

    main()
