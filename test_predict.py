# test_predict.py
from predict import predict_sentiment

def test_positive_sentence():
    assert predict_sentiment("I love this good and awesome day!") == "positive"

def test_negative_sentence():
    assert predict_sentiment("I hate this terrible and bad experience.") == "negative"

def test_neutral_sentence():
    assert predict_sentiment("The day was fine.") == "neutral"
