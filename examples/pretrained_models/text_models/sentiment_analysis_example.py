from autokeras.pretrained.text_sentiment import TextSentiment

text_cls = TextSentiment()

polarity = text_cls.predict("The model is working well..")

print("Polarity of the input sentence is (sentiment is +ve if polarity > 0.5, otherwise -ve) : ", polarity)

