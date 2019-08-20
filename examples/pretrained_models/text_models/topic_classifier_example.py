from autokeras import TopicClassifier

topic_classifier = TopicClassifier()

to_predict = "With some more practice, they will definitely make it to finals.."
topic_name = topic_classifier.predict(to_predict)

print("The input sentence belongs to the class : ", topic_name)

