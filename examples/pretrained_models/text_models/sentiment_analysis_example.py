from autokeras.pretrained.text_classifier import SentimentAnalysis

sentiment_cls = SentimentAnalysis()

polarity = sentiment_cls.predict("The model is working well..")

print("Polarity of the input sentence is (sentiment is +ve if polarity > 0.5) : ", polarity)

