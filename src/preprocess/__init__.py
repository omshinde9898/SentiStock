from abc import ABC , abstractmethod
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.tokenize import word_tokenize , sent_tokenize
from nltk.corpus import stopwords
from typing import Literal

class PreprocessStep(ABC):
    """
    Abstract class to implement Preprocessing steps and dynamically change sequence and requirements of preprocessing involved
    """

    @abstractmethod
    def fit_transform(self, data:list) -> list:
        """
        Abstract Method which on implementation fits data into Preprocess instance and returns transformed data

        Args : 
            data : list[records]

        output: transformed list
        """

    @abstractmethod
    def transform(self, data:list ) -> list:
        """
        Abstract function which on implementation should return transformed data wrt trained instance

        Args:
            data: list
        
        output : transformed list
        """


class CountVectTransformer(PreprocessStep):
    def __init__(self) -> None:
        """
        PreprocessStep Implemented to include CountVectorizer into preproccesing steps for NLP data
        """
        
        super().__init__()
        
        self.vectorizer = CountVectorizer(lowercase=True,stop_words="english")

    def fit_transform(self, data:list ) -> list:
        """
        trains a count vectorizer on given data and return transformed data 
        """
        try:
            data = self.vectorizer.fit_transform(data)
        except Exception as e:
            raise e
        
        return data
        

    def transform(self,data:list) -> list:
        """
        Returns transformed sequences of given data

        Args:
            data : list of sentences

        output: list
        """
        try:
            data = self.vectorizer.transform(data)
        except Exception as e:
            raise e
        
        return data


class TokenTransformer(PreprocessStep):
    def __init__(self,level:Literal['word','sentence']) -> None:
        """
        Preprocess step implementation for tokenizing data into token
        Args: 
            level str : word | sentence
        """
        super().__init__()
        nltk.download('punkt_tab')
        if level == 'word':
            self.tokenizer = word_tokenize
        elif level == 'sentence':
            self.tokenizer = sent_tokenize

    def fit_transform(self,data:list) -> list:
        """
        """
        try:
            data = self.tokenizer(data)
        except Exception as e:
            raise e
        
        return data

    def transform(self,data:list) -> list:
        """
        """
        try:
            data = self.tokenizer(data)
        except Exception as e:
            raise e
        
        return data