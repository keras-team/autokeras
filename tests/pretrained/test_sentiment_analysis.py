from autokeras.pretrained.text_sentiment import TextSentiment

def test_sentiment_analysis():
    sentiment_analyzer = TextSentiment()

    positive_polarity = sentiment_analyzer.predict("The model is working really well.")
    assert positive_polarity > 0.5

    negative_polarity = sentiment_analyzer.predict("The university intake has reduced drastically this year.")
    assert negative_polarity < 0.5
