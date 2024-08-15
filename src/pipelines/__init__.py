from dataclasses import dataclass
from abc import ABC , abstractmethod
from src.data.database import DatabaseHandler , PostgreSqlDatabaseHandler
from src.preprocess import PreprocessStep , CountVectTransformer ,LabelTransformer
from src.model import DecisionTreeClassifier ,ClassifierModel
import pandas as pd
import mlflow
from mlflow.sklearn import log_model
import os
from src import logger


class Pipeline(ABC):
    """
    """
    
    @abstractmethod
    def run_pipeline(self) -> None:
        """
        """
        raise NotImplementedError
    

class TrainingPipeline(Pipeline):

    def __init__(
            self,
            db_config: dict,
            model_filepath: str,
            d_handler: DatabaseHandler = PostgreSqlDatabaseHandler, 
            x_steps: list[PreprocessStep] = [CountVectTransformer],
            y_steps: list[PreprocessStep] = [LabelTransformer],
            classifier: ClassifierModel = DecisionTreeClassifier,
        ) -> None:
        """
        """
        logger.info(f"Initiating training pipeline for model : {classifier.__name__} and {d_handler.__name__}")
        self.data_handler = d_handler(db_config)
        self.steps_on_x = [i() for i in x_steps]
        self.steps_on_y = [i() for i in y_steps]
        self.classifier = classifier(model_filepath)

    def run_pipeline(self) -> None:
        logger.info(f"Executing trainig pipeling with {self.data_handler.__class__.__name__}")
        data = self.data_handler.load_from_database('traindata')
        X = data['headline']
        y = data['label']
        for i in self.steps_on_x:
            logger.info(f"Executing trainig pipeling with {i.__class__.__name__} for features")
            X = i.fit_transform(X)
        for i in self.steps_on_y:
            logger.info(f"Executing trainig pipeling with {i.__class__.__name__} for labels")
            y = i.fit_transform(y)
        
        return self.classifier.train_model(X,y,save=True)



class EvaluationPipeline(Pipeline):

    def __init__(
            self,
            db_config: dict,
            model_filepath: str,
            d_handler: DatabaseHandler = PostgreSqlDatabaseHandler, 
            x_steps: list[PreprocessStep] = [CountVectTransformer],
            y_steps: list[PreprocessStep] = [LabelTransformer],
            classifier: ClassifierModel = DecisionTreeClassifier,
        ) -> None:
        """
        """
        logger.info(f"Initiating evaluation pipeline for model : {classifier.__name__} and {d_handler.__name__}")
        self.data_handler = d_handler(db_config)
        self.steps_on_x = x_steps
        self.steps_on_y = [i() for i in y_steps]
        self.classifier = classifier(model_filepath)

    def run_pipeline(self,log_path) -> None:
        logger.info(f"Executing evaluation pipeling with {self.data_handler.__class__.__name__}")
        data = self.data_handler.load_from_database('testdata')
        X = data['headline']
        y = data['label']
        for i in self.steps_on_x:
            logger.info(f"Executing evaluation pipeling with {i.__class__.__name__} for features")
            X = i().transform(X)
        for i in self.steps_on_y:
            logger.info(f"Executing evaluation pipeling with {i.__class__.__name__} for features")
            y = i.transform(y)

        try:
            logger.info(f"Executing evaluation pipeling with MLFlow experiment tracking")
            with mlflow.start_run():

                mlflow.log_params(
                    self.classifier.get_params()
                )

                ( accuracy_scored , precision_scored , recall_scored , f1_scored ) = self.classifier.test_model(X,y)

                mlflow.log_metrics({
                    'accuracy' : accuracy_scored,
                    'precision': precision_scored,
                    'recall' : recall_scored,
                    'f1_score' : f1_scored
                })

            if not os.path.exists(log_path):
                os.makedirs(log_path)
        
            logger.info(f"Experiment ran successfully saving model artifacts at : {log_path}")
            mlflow.log_artifact(log_path)
            
            log_model(self.classifier,'model')

        except Exception as e:
            logger.info(f"Failed experiment with MLFlow tracking")
            raise e

