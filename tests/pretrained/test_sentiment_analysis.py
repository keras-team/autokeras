from autokeras.pretrained.text_sentiment import TextSentiment

def test_sentiment_analysis():
    sentiment_analyzer = TextSentiment()

    positive_polarity = sentiment_analyzer.predict("The model is working really well.")
    if positive_polarity <= 0.5:
        raise AssertionError()

    negative_polarity = sentiment_analyzer.predict("The university intake has reduced drastically this year.")
    if negative_polarity >= 0.5:
        raise AssertionError()
