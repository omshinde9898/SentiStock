from abc import ABC , abstractmethod
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import nltk
import pickle
import os
from pathlib import Path
from nltk.tokenize import word_tokenize , sent_tokenize
from nltk.corpus import stopwords
from typing import Literal
from src import logger
from src.utils import read_yaml_config

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
    
    @abstractmethod
    def inverse_transform(self, data:list ) -> list:
        """
        Abstract function which on implementation should return transformed data wrt trained instance

        Args:
            data: list
        
        output : transformed list
        """


class CountVectTransformer(PreprocessStep):
    def __init__(self, save:bool = True ) -> None:
        """
        PreprocessStep Implemented to include CountVectorizer into preproccesing steps for NLP data
        """
        
        super().__init__()
        
        self.vectorizer = None
        self.save = save
        self.load_path = 'artifacts/CountVectTransformer/latest.pkl'
        self.config = read_yaml_config('config/preprocess_config.yaml')['CountVectTransformer']

    def fit_transform(self, data:list ) -> list:
        """
        trains a count vectorizer on given data and return transformed data 
        """
        try:
            self.vectorizer = CountVectorizer(
                input = self.config['input'],
                encoding = self.config['encoding'],
                decode_error = self.config['decode_error'],
                strip_accents = None if self.config['strip_accents'] == 'None' else self.config['strip_accents'],
                lowercase = self.config['lowercase'],
                preprocessor = None if self.config['preprocessor'] == 'None' else self.config['preprocessor'],
                tokenizer = None if self.config['tokenizer'] == 'None' else self.config['tokenizer'],
                stop_words = self.config['stop_words'],
                token_pattern = None if self.config['token_pattern'] == 'None' else self.config['token_pattern'],
                analyzer = self.config['analyzer'],
                max_df = self.config['max_df'],
                min_df = self.config['min_df'],
                max_features = None if self.config['max_features'] == 'None' else self.config['max_features'],
                vocabulary = None if self.config['vocabulary'] == 'None' else self.config['vocabulary'],
                binary = self.config['binary'],
            )
            data = self.vectorizer.fit_transform(data)

            if self.save:
                logger.info(f"Saving Vectorizer to : {self.load_path}")
                if not os.path.exists(self.load_path):
                    path = self.load_path.split('/')[:-1]
                    os.makedirs('/'.join(path),exist_ok=True)
                with open(self.load_path,'wb') as f:
                    pickle.dump(self.vectorizer,f)
                
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
            logger.info(f"Reading vectorizer binary from : {self.load_path}")
            with open(self.load_path,'rb') as f:
                self.vectorizer = pickle.load(f)

            data = self.vectorizer.transform(data)
        
        except Exception as e:
            raise e
        
        return data


    def inverse_transform(self,data:list) -> list:
        """
        Returns transformed sequences of given data

        Args:
            data : list of sentences

        output: list
        """
        try:
            logger.info(f"Reading vectorizer binary from : {self.load_path}")
            with open(self.load_path,'rb') as f:
                self.vectorizer = pickle.load(f)

            data = self.vectorizer.inverse_transform(data)
        
        except Exception as e:
            raise e
        
        return data


class LabelTransformer(PreprocessStep):
    def __init__(self,save : bool = True) -> None:
        """
        PreprocessStep Implemented to include LabelEncoder into preproccesing steps for categorical data
        """
        
        super().__init__()
        
        self.encoder = None
        self.save = save
        self.load_path = f'artifacts/LabelTransformer/latest.pkl'

    def fit_transform(self, data:list ) -> list:
        """
        trains a count vectorizer on given data and return transformed data 
        """
        try:
            self.encoder = LabelEncoder()
            data = self.encoder.fit_transform(data)

            if self.save:
                logger.info(f"Saving Encoder to : {self.load_path}")
                if not os.path.exists(self.load_path):
                    path = self.load_path.split('/')[:-1]
                    os.makedirs('/'.join(path),exist_ok=True)
                with open(self.load_path,'wb') as f:
                    pickle.dump(self.encoder,f)


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

            logger.info(f"Reading encoder binary from : {self.load_path}")
            with open(self.load_path,'rb') as f:
                self.encoder = pickle.load(f)

            data = self.encoder.transform(data)
        except Exception as e:
            raise e
        
        return data

    def inverse_transform(self,data:list) -> list:
        """
        Returns transformed sequences of given data

        Args:
            data : list of sentences

        output: list
        """
        try:

            logger.info(f"Reading encoder binary from : {self.load_path}")
            with open(self.load_path,'rb') as f:
                self.encoder = pickle.load(f)

            data = self.encoder.inverse_transform(data)
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
    
    def inverse_transform(self, data: list) -> list:
        raise NotImplementedError