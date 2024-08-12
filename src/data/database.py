import pandas as pd
from abc import ABC , abstractmethod
from sqlalchemy import create_engine ,text


class DatabaseHandler(ABC):
    def __init__(self, config : dict) -> None:
        """
        Args:
            config : {
                "host" : host,
                "user" : user,
                "password" : password,
                "database" : database
            }

        return DatabaseHandler
        """
        self.host = config['host']
        self.user = config['user']
        self.password = config['password']
        self.database = config['database']
        self.conn = None
    
    @abstractmethod
    def execute(self,query:str):
        """s
        executes a given query for choosen database handler instance and returns output

        Args:
            query : str
        Output:
            any
        """
    
    @abstractmethod
    def frame_to_database(self, frame: pd.DataFrame, name:str ) -> None:
        """
        loads a dataframe into database
        """

    @abstractmethod
    def load_from_database(self) -> pd.DataFrame:
        """
        loads data from database instance to pandas Dataframe
        """

    @abstractmethod
    def add_to_database(self, headline: str, outcome: bool, name:str) -> None:
        """
        this method adds new records to already existing database
        """


class PostgreSqlDatabaseHandler(DatabaseHandler):
    """
    This class handles all procedures related to SQL database 
    """

    def __init__(self, config: dict) -> None:
        """
        To create SQL Database Instance and setup a connection with remote database

        Args:
            config : {
                "host" : host,
                "user" : user,
                "password" : password,
                "database" : database
            }

        return PostgreSqlDatabaseHandler
        """
        super().__init__(config)

        self.conn = create_engine(f"postgresql://{self.user}:{self.password}@{self.host}:5432/{self.database}")

    def execute(self, query: str) -> list:
        """
        To execute custom SQL query
        """

        connection = self.conn.connect()
        
        try:
        
            cursor_result = connection.execute(text(query))
        
            return cursor_result.all()
        
        except Exception as e:
        
            raise e

    def load_from_database(self,table_name:str,) -> pd.DataFrame:
        """
        This function loads dataset from givem database into pd.Dataframe

        Args:
            table_name = 'Train_Data'|'Test_Data'

        return : Dataframe
        """
        try:
        
            return pd.read_sql_table(table_name=table_name,con=self.conn).drop(['index'],axis=1)
        
        except Exception as e:
            raise e
        
    def frame_to_database(self, frame: pd.DataFrame, name:str ) -> None:
        """
        loads data to 'Train_Data' or 'Test_Data' tables specify name accordingly

        Args:
            frame : Dataframe to load
            name : 'traindata' | 'testdata'

        Output : None
        """
        try:

            frame.to_sql(name, con=self.conn ,if_exists='replace')
        
        except Exception as e:
            raise e

    def add_to_database(self, headline: str, outcome: bool, name:str) -> None:
        """
        loads single record to 'Train_Data' or 'Test_Data' tables specify name accordingly

        Args:
            headline : news headline
            outcome : actual or predicted outcome
            name (Table Name) : 'traindata' | 'testdata' | 'infer_data'

        Output : None
        """
        try:

            fr = pd.DataFrame([[headline][outcome]] , columns=['headline','label'])
            
            fr.to_sql(con=self.conn, name=name, if_exists='append')
        
        except Exception as e:
            raise e


class MongoDatabaseHandler(DatabaseHandler):
    """

    Args:
            config : {
                "host" : host,
                "user" : user,
                "password" : password,
                "database" : database
            }

        return MongoDatabaseHandler
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)

        self.conn = self.user

    