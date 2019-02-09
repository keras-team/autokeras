from autokeras.pretrained.text_classifier import SentimentAnalysis


def test_sentiment_analysis():
    sentiment_analyzer = SentimentAnalysis()

    positive_polarity = sentiment_analyzer.predict("The model is working really well.")
    if positive_polarity <= 0.5:
        raise AssertionError()

    negative_polarity = sentiment_analyzer.predict("The university intake has reduced drastically this year.")
    if negative_polarity >= 0.5:
        raise AssertionError()
