import re
import nltk
import pandas as pd

from html.parser import HTMLParser
from nltk.corpus import stopwords


try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

    
_english_stopwords = set(stopwords.words("english"))


class HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)

    def get_text(self):
        return ''.join(self.parts)

# def strip_html(html):
#     stripper = HTMLStripper()
#     stripper.feed(html)
#     return stripper.get_text()

# html = "<p>Hello <b>world</b> &amp; universe</p>"
# text = strip_html(html)
# print(text)  # Hello world & universe


@pd.api.extensions.register_series_accessor("nlp")
class KaggleWord2VecAccessor:
    def __init__(self, series):
        self._s = series                

    def review_to_wordlist(self):    
        return self._s.apply(self._review_to_wordlist)

    @staticmethod
    def _review_to_wordlist(text):
        html_stripper = HTMLStripper()
        html_stripper.feed(text)

        # 1. Remove HTML
        review_text = html_stripper.get_text()

        # 2. Remove non-letters
        review_text = re.sub("[^a-zA-Z]"," ", review_text)

        # 3. Convert words to lower case and split them
        words = review_text.lower().split()

        # 4. Optionally remove stop words (false by default)
        # if remove_stopwords:
        #     stops = set(stopwords.words("english"))
        words = [w for w in words if w not in _english_stopwords]

        return words    
