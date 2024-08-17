import pandas as pd
from abc import ABC , abstractmethod
import pymongo.mongo_client
from sqlalchemy import create_engine ,text
import pymongo
from src.utils import read_yaml_config
from src import logger


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
    def frame_to_database(self, frame: pd.DataFrame, name:str ) -> None:
        """
        loads a dataframe into database
        """

    @abstractmethod
    def load_from_database(self, name : str) -> pd.DataFrame:
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
        super().__init__(
            read_yaml_config('config/database_config.yaml')['PostgresConfig']
        )

        self.conn = create_engine(f"postgresql://{self.user}:{self.password}@{self.host}:5432/{self.database}")
        logger.info(f"Using PostgreSQL Database handler on host : {self.host}")

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
            logger.info(f"Requesting data from databse table : {table_name}")

            return pd.read_sql_table(table_name=table_name,con=self.conn).drop(['index'],axis=1)
        
        except Exception as e:
            logger.error(f"Failed to load data from database table: {table_name}")
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
            logger.info(f"Updating table : {name} with new dataframe")
            frame.to_sql(name, con=self.conn ,if_exists='replace')
        
        except Exception as e:
            logger.error(f"Failed to update data on table : {name}")
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
            
            logger.info(f'Adding new data on table : {name}')
            fr.to_sql(con=self.conn, name=name, if_exists='append')
        
        except Exception as e:
            logger.error(f"Failed to add new data on table : {name}")
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

        self.conn = pymongo.MongoClient(f"mongodb://{self.user}:{self.password}@{self.host}:27017/")
        self.conn = self.conn[self.database]


    def get_db_connection(self):
        """
        Returns a MongoDB connection to given database
        """
        try:
            return self.conn
        except Exception as e:
            raise e


    def load_from_database(self,name:str,) -> pd.DataFrame:
        """
        This function loads dataset from givem database into pd.Dataframe

        Args:
            name = 'traindata'|'testdata'

        return : Dataframe
        """
        try:
        
            #TODO implementatioin pending
            pass
        
        except Exception as e:
            raise e
        
    def frame_to_database(self, frame: pd.DataFrame, name:str ) -> None:
        """
        loads data to 'Train_Data' or 'Test_Data' tables specify name accordingly

        Args:
            frame : Dataframe to load
            name : 'traindata' | 'testdata' collection name

        Output : None
        """
        try:
            records = []
            for i in range(frame.shape[0]):
                records.append({'headline':frame.iloc[i]['headline'],'label':frame.iloc[i]['label']})
            
            self.conn[name].insert_many(records)
        
        except Exception as e:
            raise e

    def add_to_database(self, headline: str, outcome: bool, name:str) -> None:
        """
        loads single record to 'Train_Data' or 'Test_Data' tables specify name accordingly

        Args:
            headline : news headline
            outcome : actual or predicted outcome
            name (Collection Name) : 'traindata' | 'testdata' | 'infer_data'

        Output : None
        """
        try:

            #TODO implementatioin pending
            pass
        
        except Exception as e:
            raise e

    