from textblob import TextBlob

class SpellCorrector:
    def __init__(self):
        pass

    def correct(self, text):
        blob = TextBlob(text)
        return str(blob.correct())
