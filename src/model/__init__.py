from abc import ABC , abstractmethod
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import pickle

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

        if filepath :
            self.filepath = filepath
            with open(filepath, 'rb') as file:
                self.model = pickle.load(file)


    def train_model(self,x_train, y_train ,save:bool = True):
        """
        method to train model or train and save it as pickel file on given filepath if save == True 

        Args:
            x_train : training data
            y_train : labels
            save : (True | False) True

        output : model history

        """
        
        hist = self.model.fit(x_train , y_train)

        if save:
            if self.filepath == None:
                raise Exception('Filepath Not given while instantiating')
            else:
                with open(self.filepath, 'wb') as file:
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
        
        y_pred = self.model.predict(x_test)
        accuracy_scored = accuracy_score(y_test,y_pred)
        precision_scored = precision_score(y_test,y_pred)
        recall_scored = recall_score(y_test,y_pred)
        f1_scored = f1_score(y_test,y_pred)

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
        headline = [headline]

        return self.model.predict(headline)


    def predict_onFrame(self, headlines):
        """
        Method to run inference on multiple headlines 

        Args:
            headlines : list

        output : list[ predicted labels ]
        """
        return self.model.predict(headlines)
    