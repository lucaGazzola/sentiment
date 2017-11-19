'''
Created on 19 Nov 2017

Gets the last 'num_tweets' tweets containing 'keywords' and lets you manually classify them, saves the classification in training_data.csv

@author: Luca G.
'''
import csv
import tweepy
import urllib
import sys
import config


def main():

    if len(sys.argv) < 3:
        print('usage: manual-classifier.py <number of tweets to classify -- example: 100> <keywords -- example: "bitcoin, price">')
        sys.exit()

    num_tweets = sys.argv[1]
    keywords = sys.argv[2]

    tweets = get_last_tweets(num_tweets, keywords)
    classification = {}

    for tweet in tweets:
        encoded = tweet.text.encode("utf-8")
        print(encoded)
        sentiment = input('is the tweet positive(2), neutral(1) or negative(0) with respect to {0}? (enter 3 if you want to skip)'.format(keywords))
        if sentiment != '3':
            classification[encoded] = sentiment

    with open('training_data.csv', 'w', newline='') as f:
        fieldnames = ['Tweet', 'Sentiment']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        data = [dict(zip(fieldnames, [k, v])) for k, v in classification.items()]
        writer.writerows(data)


def get_last_tweets(number, keywords):

    auth = tweepy.OAuthHandler(config.twitter_api['consumer_key'], config.twitter_api['consumer_secret'])
    auth.set_access_token(config.twitter_api['access_token'], config.twitter_api['access_token_secret'])
    twitter_api = tweepy.API(auth)

    return twitter_api.search(q=[keywords], count=number)


if __name__ == '__main__':

    main()
