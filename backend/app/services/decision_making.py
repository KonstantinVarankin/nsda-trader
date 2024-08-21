from app.services.nlp_analysis import nlp_analyzer
from app.services.technical_analysis import technical_analyzer
import numpy as np

class DecisionMaker:
    def __init__(self):
        self.nlp_analyzer = nlp_analyzer
        self.technical_analyzer = technical_analyzer

    def get_sentiment_score(self, symbol):
        sentiment = self.nlp_analyzer.get_overall_sentiment(symbol)
        if sentiment > 0.2:
            return 1  # Bullish
        elif sentiment < -0.2:
            return -1  # Bearish
        else:
            return 0  # Neutral

    def get_technical_score(self, symbol):
        signals = self.technical_analyzer.generate_signals(symbol)
        score = 0

        if signals['sma_crossover'] == 'buy':
            score += 1
        elif signals['sma_crossover'] == 'sell':
            score -= 1

        if signals['rsi'] == 'oversold':
            score += 1
        elif signals['rsi'] == 'overbought':
            score -= 1

        if signals['macd'] == 'bullish_crossover':
            score += 1
        elif signals['macd'] == 'bearish_crossover':
            score -= 1

        return score

    def make_decision(self, symbol):
        sentiment_score = self.get_sentiment_score(symbol)
        technical_score = self.get_technical_score(symbol)

        total_score = sentiment_score + technical_score

        if total_score >= 2:
            return "Strong Buy"
        elif total_score == 1:
            return "Buy"
        elif total_score == 0:
            return "Hold"
        elif total_score == -1:
            return "Sell"
        else:
            return "Strong Sell"

    def get_confidence(self, symbol):
        sentiment = abs(self.nlp_analyzer.get_overall_sentiment(symbol))
        technical_indicators = self.technical_analyzer.get_latest_indicators(symbol)
        
        # Use some key indicators for confidence calculation
        rsi = technical_indicators.get('momentum_rsi', 50)
        macd = technical_indicators.get('trend_macd_diff', 0)
        
        # Normalize indicators
        rsi_confidence = 1 - abs(50 - rsi) / 50  # RSI closer to 50 means less confidence
        macd_confidence = min(abs(macd), 1)  # Larger absolute MACD means more confidence, cap at 1
        
        # Combine confidences
        confidence = np.mean([sentiment, rsi_confidence, macd_confidence])
        return round(confidence * 100, 2)  # Return as percentage

decision_maker = DecisionMaker()
