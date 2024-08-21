from newsapi import NewsApiClient
from app.core.config import settings
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

class NLPAnalyzer:
    def __init__(self):
        self.newsapi = NewsApiClient(api_key=settings.NEWS_API_KEY)
        nltk.download('vader_lexicon')
        self.sia = SentimentIntensityAnalyzer()

    def get_news(self, query):
        news = self.newsapi.get_everything(q=query, language='en', sort_by='relevancy', page_size=10)
        return news['articles']

    def analyze_sentiment(self, text):
        return self.sia.polarity_scores(text)

    def analyze_news_sentiment(self, query):
        news = self.get_news(query)
        sentiments = []
        for article in news:
            sentiment = self.analyze_sentiment(article['title'] + ' ' + article['description'])
            sentiments.append({
                'title': article['title'],
                'sentiment': sentiment
            })
        return sentiments

nlp_analyzer = NLPAnalyzer()
