from abc import ABC , abstractmethod
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import pickle
from src import logger

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
    def __init__(self,filepath : str = None) -> None:
        """
        """

        self.model = None
        self.accuracy = None
        self.filepath = None

        # TODO : Create function to update model parameters
        # self.params = {
        #     'criterion':'gini',
        #     'splitter': 'best',

        # }

        # class DT(
        #     *,
        #     criterion: Literal['gini', 'entropy', 'log_loss'] = "gini",
        #     splitter: Literal['best', 'random'] = "best",
        #     max_depth: Int | None = None,
        #     min_samples_split: float | int = 2,
        #     min_samples_leaf: float | int = 1,
        #     min_weight_fraction_leaf: Float = 0,
        #     max_features: float | int | Literal['auto', 'sqrt', 'log2'] | None = None,
        #     random_state: Int | RandomState | None = None,
        #     max_leaf_nodes: Int | None = None,
        #     min_impurity_decrease: Float = 0,
        #     class_weight: Mapping | str | Sequence[Mapping] | None = None,
        #     ccp_alpha: float = 0
        # )
        

        if filepath :
            self.filepath = filepath
            with open(filepath, 'rb') as file:
                logger.info(f"Loading model file from : {filepath}")

                # TODO : Implement exception handling for wrong filepath
                self.model = pickle.load(file)
        else:
            logger.info(f"Using untrained model, filepath not provided")
            self.model = DT()


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
            if self.filepath == None:
                raise Exception('Filepath Not given while instantiating')
            else:
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
        precision_scored = precision_score(y_test,y_pred)
        recall_scored = recall_score(y_test,y_pred)
        f1_scored = f1_score(y_test,y_pred)

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
    