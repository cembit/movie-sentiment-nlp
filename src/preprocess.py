"""Text cleaning for movie reviews."""

import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

_LEMMATIZER = WordNetLemmatizer()
_STOP_WORDS: set[str] | None = None

_HTML_TAG = re.compile(r"<[^>]+>")
_NON_ALPHA = re.compile(r"[^a-z\s]")


def _ensure_nltk_data() -> None:
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


def _get_stop_words() -> set[str]:
    global _STOP_WORDS
    if _STOP_WORDS is None:
        _ensure_nltk_data()
        _STOP_WORDS = set(stopwords.words("english"))
    return _STOP_WORDS


def clean_text(text: str) -> str:
    """Lowercase, strip HTML, keep letters, remove stopwords, lemmatize."""
    _ensure_nltk_data()
    text = _HTML_TAG.sub(" ", text)
    text = text.lower()
    text = _NON_ALPHA.sub(" ", text)
    tokens = [t for t in text.split() if t and t not in _get_stop_words()]
    return " ".join(_LEMMATIZER.lemmatize(t) for t in tokens)
