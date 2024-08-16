from abc import ABC , abstractmethod
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import pickle
import os
from typing import Literal
from src import logger
from src.utils import read_yaml_config

class ClassifierModel(ABC):
    """
    """

    @abstractmethod
    def train_model(self, x_train , y_train ,save:bool):
        """
        method to train and save model on given conditions
        """
        raise NotImplementedError

    @abstractmethod
    def test_model(self,x_test , y_test):
        """
        Method to test model on given data
        """
        raise NotImplementedError


    @abstractmethod
    def predict_sentiment(self,headline):
        """
        run prediction on single headline
        """
        raise NotImplementedError

    @abstractmethod
    def predict_onFrame(self,headlines):
        """
        run prediction on multiple headlines
        """
        raise NotImplementedError


class DecisionTreeClassifier(ClassifierModel):
    def __init__(self,filepath : str) -> None:
        """
        """

        self.model = None
        self.accuracy = None
        self.filepath = read_yaml_config('config/model_config.yaml')['FILEPATH']

        # TODO : Create function to update model parameters
        self.params = read_yaml_config('config/model_config.yaml')['DecisionTreeClassifier']

        if os.path.exists(filepath) :
            with open(filepath, 'rb') as file:
                logger.info(f"Loading model file from : {filepath}")

                # TODO : Implement exception handling for wrong filepath
                self.model = pickle.load(file)
        else:
            logger.info(f"Using untrained model, filepath not provided")

            self.model = DT(
                criterion = self.params['criterion'],
                splitter = self.params['splitter'],
                max_depth = None if self.params['max_depth'] == 'None' else self.params['max_depth'],
                min_samples_split = self.params['min_samples_split'],
                min_samples_leaf = self.params['min_samples_leaf'],
                min_weight_fraction_leaf = self.params['min_weight_fraction_leaf'],
                max_features = None if self.params['max_features'] == 'None' else self.params['max_features'],
                random_state = None if self.params['random_state'] == 'None' else self.params['random_state'],
                max_leaf_nodes = None if self.params['max_leaf_nodes'] == 'None' else self.params['max_leaf_nodes'],
                min_impurity_decrease = self.params['min_impurity_decrease'],
                class_weight = None if self.params['class_weight'] == 'None' else self.params['class_weight'],
                ccp_alpha = self.params['ccp_alpha'],
            )


    def train_model(self,x_train, y_train ,save:bool = True):
        """
        method to train model or train and save it as pickel file on given filepath if save == True 

        Args:
            x_train : training data
            y_train : labels
            save : (True | False) True

        output : model history

        """
        try:
            
            logger.info("Training of classifier model has started!")        
            hist = self.model.fit(x_train , y_train)    
            logger.info("Training of classifier model has Finished!")

        except Exception as e:
            
            logger.error(f"Cannot train classifier model")
            raise e

        if save:
            with open(self.filepath, 'wb') as file:
                logger.info(f"Saving model file on path : {self.filepath}")
                pickle.dump(self.model, file)
        
        return hist
    
    def test_model(self,x_test, y_test):
        """
        method to test model on given test data 

        Args:
            x_test : test data
            y_test : true labels

        output : (accuracy , precision , recall, f1 score)
        """
        
        logger.info(f"Model evaluation has initiated")
        y_pred = self.model.predict(x_test)
        accuracy_scored = accuracy_score(y_test,y_pred)
        precision_scored = precision_score(y_test,y_pred,average='weighted')
        recall_scored = recall_score(y_test,y_pred,average='weighted')
        f1_scored = f1_score(y_test,y_pred,average='weighted')

        logger.info(f"Model evaluation complete with accuracy : {accuracy_scored}")
        self.accuracy = accuracy_scored
        print(accuracy_scored)
        print(precision_scored)
        print(recall_scored)
        print(f1_scored)

        return ( accuracy_scored , precision_scored , recall_scored , f1_scored )
        
    
    def predict_sentiment(self,headline):
        """
        Method to run inference on single record i.e headline : str 

        Args:
            headline : str

        output : predicted label
        """
        logger.info(f"Running inference for headline : {headline}")
        headline = [headline]

        return self.model.predict(headline)


    def predict_onFrame(self, headlines):
        """
        Method to run inference on multiple headlines 

        Args:
            headlines : list

        output : list[ predicted labels ]
        """
        logger.info(f"Running inference for multiple headlines")
        return self.model.predict(headlines)
    
    def get_params(self):
        return self.model.get_params()
    