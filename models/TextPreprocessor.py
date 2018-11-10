"""
Cleans the Figure Eight Multilingual Disaster Response Messages data

https://www.figure-eight.com/dataset/combined-disaster-response-data/

CAUTION: 
-------
This class was placed in a separate *.py file in order to run scikit-learn's
GridSearch in parallel.

https://github.com/scikit-learn/scikit-learn/issues/10054
"""
import contractions
import nltk
from nltk import word_tokenize, pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
import pandas as pd
import re
from sklearn.base import BaseEstimator, TransformerMixin
import string

class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Applies the folowing transformations to text data:
    
    1. Removes web addresses

    2. Removes 'RT" (retweet) and 'yr old'

    3. Tokenizes text into words

    4. Removes punctunation

    5. Split word tokens separated by punctuation and transforms them to 
    lower case
    
    6. Removes punctunation

    7. Tag Part of Speech (POS) & lemmatize

    8. Remove english stopwords

    9. Remove word tokens that start with a number
    """
    def __init__(self):
        """TextPreprocessor class constructor

        INPUT:
            self: TextPreprocessor class object handle

        OUTPUT:
            self: TextPreprocessor class object handle"""
        self.english_stopwords = stopwords.words('english')

        self.lemmatizer = WordNetLemmatizer()

        self.translator = str.maketrans('', '', string.punctuation)

    def fit(self, X, y=None):
        """Transformer fit method

        INPUT:
            self: TextPreprocessor class object handle

            X: Matrix that stores text data

            y: (Optional) Text data label(s)

        OUTPUT:
            self: TextPreprocessor class object handle"""
        return self

    def transform(self, X):
        """Transforms text into a set of word tokens

        INPUT:
            X: Matrix that stores Figure Eight Multilingual Disaster 
               Response Messages data

        OUTPUT:
            X_clean: Matrix that stores preprocessed Figure Eight Multilingual
                     Disaster Response Messages data"""
        Tokens = pd.Series(X).apply(lambda elem: self.tokenize(elem))

        return Tokens.apply(lambda elem: " ".join(elem)).values

    def tokenize(self,
                 text):
        """Applies the folowing transformations to text data:

        1. Removes web addresses

        2. Removes 'RT" (retweet) and 'yr old'

        3. Tokenizes text into words

        4. Removes punctunation

        5. Split word tokens separated by punctuation and transforms them to 
        lower case
        
        6. Removes punctunation

        7. Tag Part of Speech (POS) & lemmatize

        8. Remove english stopwords

        9. Remove word tokens that start with a number

        INPUT:
            text: String that stores a Figure Eight Multilingual Disaster
                  Response message

        OUTPUT:
            tokens: List of Figure Eight Multilingual Disaster Response
                    message word tokens
        """

        # Remove web addresses
        cleaned_text = TextPreprocessor.clean_text(text)

        # Remove 'RT' (retweet) and 'yr old'
        cleaned_text = re.sub('RT', ' ', cleaned_text)
        cleaned_text = re.sub('yr old', ' ', cleaned_text)

        # Word tokenize
        tokens = [elem for elem in word_tokenize(cleaned_text)]

        # Remove punctuation
        pattern_obj = re.compile('[' + string.punctuation + ']+')
        tokens = [elem for elem in tokens if pattern_obj.match(elem) is None]

        # Split word tokens separated by punctuation and transforms them to
        # lower case
        tokens = [elem.lower() for elem in TextPreprocessor.split_tokens(tokens)]

        # Remove punctuation
        tokens = [elem.translate(self.translator) for elem in tokens]
        tokens = [elem for elem in tokens if len(elem) > 0]

        # Tag Part of Speech (POS) & lemmaitze
        tokens = [self.lemmatize(elem) for elem in pos_tag(tokens)]

        # Remove english stopwords
        tokens = [elem for elem in tokens if elem not in self.english_stopwords]

        # Remove word tokens that start with a number
        pattern_obj = re.compile('^[0-9].*$')
        return [elem for elem in tokens if pattern_obj.match(elem) is None]

    def lemmatize(self,
                  tagged_token):
        """
        This method transforms a part of speech (POS) tagged word token into 
        its root form via application of the WordNet lemmatizer

        INPUT:
            self: TextPreprocessor object handle

            tagged_token: Tuple that stores a POS tagged token

        OUTPUT:
            root_form: Root form of a POS tagged word token

        Reference:
        ---------
        https://textminingonline.com/
        dive-into-nltk-part-iv-stemming-and-lemmatization

        https://stackoverflow.com/questions/15586721/
            wordnet-lemmatization-and-pos-tagging-in-python
        """
        if tagged_token[1].startswith('J'):
            pos = wordnet.ADJ
        #---------------------------------------------------------
        elif tagged_token[1].startswith('V'):
            pos = wordnet.VERB
        #---------------------------------------------------------
        elif tagged_token[1].startswith('N'):
            pos = wordnet.NOUN
        #---------------------------------------------------------
        elif tagged_token[1].startswith('R'):
            pos = wordnet.ADV
        #---------------------------------------------------------
        else:
            pos = None

        if pos is None:
            return self.lemmatizer.lemmatize(tagged_token[0])
        else:
            return self.lemmatizer.lemmatize(tagged_token[0], pos)

    @staticmethod
    def clean_text(text):
        """
        This static method removes web addresses

        INPUT:
            text: String that may contain web addresses

        OUTPUT:
            cleaned_text: String after the removal of web addresses

        Reference:
        ---------
        Udacity Data Scientist Nanodegree course code example
        """
        cleaned_text = contractions.fix(text)

        url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]' +\
                    '|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

        cleaned_text = re.sub(url_regex, ' ', cleaned_text)

        cleaned_text = re.sub(r'http[a-zA-Z\s&\.]+\s', ' ', cleaned_text)
        cleaned_text = re.sub(r'http\s+:\s+[/a-zA-Z0-9\.]+', ' ', cleaned_text)
        return re.sub(r'[w]{3}\.([A-Za-z0-9-]+)\.com', ' ', cleaned_text)

    @staticmethod
    def split_tokens(tokens):
        """
        This static method splits word tokens separated by punctuation

        INPUT:
            tokens: List of word tokens that may include words separated by
                    punctuation

        OUTPUT:
            clean_tokens: List of word tokens
        """
        clean_tokens = []

        pattern = '[' + string.punctuation + ']'

        for elem in tokens:
            clean_tokens.extend(re.split(pattern, elem))

        return clean_tokens