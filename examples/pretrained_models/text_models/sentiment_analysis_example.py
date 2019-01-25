from autokeras.pretrained.sentiment_analysis import SentimentAnalysis

text_cls = SentimentAnalysis()

polarity = text_cls.predict("The model is working well..")

print("Polarity of the input sentence is (sentiment is +ve if polarity > 0.5) : ", polarity)

