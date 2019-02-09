from autokeras.pretrained.text_classifier import TopicClassifier


def test_topic_classifier():
    topic_classifier = TopicClassifier()

    topic_name = topic_classifier.predict(
        "Risk mitigation is the pursuit of opportunities where the potential upside is far greater than the potential "
        "downside", )

    if topic_name != "Business":
        raise AssertionError()

    topic_name = topic_classifier.predict(
        "With a tap on the screen the app will recognise your face and bring up the filter menu", )

    if topic_name != "Sci/Tech":
        raise AssertionError()

    topic_name = topic_classifier.predict(
        "Anthony received a loud ovation when he was shown on the overhead videoboard in the first quarter", )

    if topic_name != "Sports":
        raise AssertionError()

    topic_name = topic_classifier.predict("The soviet union was created about five years after Russian Revolution.", )

    print(topic_name)
    if topic_name != "World":
        raise AssertionError()
