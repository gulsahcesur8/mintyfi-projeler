# predict.py (refactor edilmiş hali)

POSITIVE_WORDS = ['good', 'great', 'happy', 'love', 'awesome']
NEGATIVE_WORDS = ['bad', 'sad', 'terrible', 'hate', 'angry']

def count_sentiment_words(text, word_list):
    return sum(word in text for word in word_list)

def predict_sentiment(text):
    """
    Metindeki duyguyu pozitif/negatif/nötr olarak tahmin eder.
    """
    text = text.lower()
    pos_count = count_sentiment_words(text, POSITIVE_WORDS)
    neg_count = count_sentiment_words(text, NEGATIVE_WORDS)

    if pos_count > neg_count:
        return 'positive'
    elif neg_count > pos_count:
        return 'negative'
    else:
        return 'neutral'
