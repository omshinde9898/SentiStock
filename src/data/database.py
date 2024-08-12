import pandas as pd

class DatabaseHandler():
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

        self.conn = conn

    def execute(self,query:str):
        pass

    def frame_to_database(self,frame: pd.DataFrame) -> None:
        pass

    def load_from_database(self) -> pd.DataFrame:
        pass

    def add_to_database(self, headline:str, outcome:bool ) -> None:
        pass
