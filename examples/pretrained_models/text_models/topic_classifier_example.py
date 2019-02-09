from autokeras.pretrained.text_classifier import TopicClassifier

topic_classifier = TopicClassifier()

topic_name = topic_classifier.predict("With some more practice, they will definitely make it to finals..", )

print("The input sentence belongs to the class : ", topic_name)

