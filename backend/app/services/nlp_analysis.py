import nltk
from textblob import TextBlob
from newsapi import NewsApiClient
import tweepy
from app.core.config import settings

nltk.download('punkt')

class NLPAnalyzer:
    def __init__(self):
        self.newsapi = NewsApiClient(api_key=settings.NEWS_API_KEY)
        self.twitter_auth = tweepy.OAuthHandler(settings.TWITTER_API_KEY, settings.TWITTER_API_SECRET_KEY)
        self.twitter_auth.set_access_token(settings.TWITTER_ACCESS_TOKEN, settings.TWITTER_ACCESS_TOKEN_SECRET)
        self.twitter_api = tweepy.API(self.twitter_auth)

    def analyze_sentiment(self, text):
        blob = TextBlob(text)
        return blob.sentiment.polarity

    def get_news_sentiment(self, query):
        articles = self.newsapi.get_everything(q=query, language='en', sort_by='publishedAt', page_size=100)
        sentiments = [self.analyze_sentiment(article['title'] + ' ' + article['description']) for article in articles['articles']]
        return sum(sentiments) / len(sentiments) if sentiments else 0

    def get_social_media_sentiment(self, query):
        tweets = self.twitter_api.search_tweets(q=query, lang='en', count=100)
        sentiments = [self.analyze_sentiment(tweet.text) for tweet in tweets]
        return sum(sentiments) / len(sentiments) if sentiments else 0

    def get_overall_sentiment(self, symbol):
        news_sentiment = self.get_news_sentiment(symbol)
        social_sentiment = self.get_social_media_sentiment(symbol)
        return (news_sentiment + social_sentiment) / 2

nlp_analyzer = NLPAnalyzer()
